import time
from math import ceil
import warnings
from collections import defaultdict
import json
import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
import torchmetrics
import whisper
import os, re
import torch.nn.functional as F
import torchaudio
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
from whisper.normalizers import EnglishTextNormalizer


def load_transcripts_words(transcripts_path: str):
    """Return a sorted list of unique, lowercased words across all transcripts."""
    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    words = set()
    for _, text in transcripts.items():
        # keep contractions (you're) as a single token
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
        words.update(tokens)
    return sorted(words)

def build_allowed_token_id_set_with_tok(transcripts_path, tok):
    """
    Return a sorted list of token IDs that are globally allowed at any step.
    Reads full sentences from transcripts and collects all tokens Whisper's
    tokenizer may emit (both sentence-initial and space-prefixed variants).
    """
    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    allowed = set()

    # (0) Special/control tokens
    allowed.update(tok.sot_sequence_including_notimestamps)  # SOT + language/task + no-timestamps
    allowed.add(tok.eot)

    # (1) whitespace / separators the tokenizer may emit
    for s in [" ", "  ", "\n", "\n\n", "\t"]:
        allowed.update(tok.encode(s))

    # (2) common punctuation (both with and without leading space)
    for punct in [".", ",", "!", "?", "'", '"', ":", ";", "-", "—", "…", "(", ")", "[", "]"]:
        for s in (punct, " " + punct):
            allowed.update(tok.encode(s))

    # (3) tokenize each transcript sentence both as-is and with leading space
    for _, text in transcripts.items():
        allowed.update(tok.encode(text))       # sentence-initial tokens
        allowed.update(tok.encode(" " + text)) # space-prefixed tokens

    # (4) safety net: encode the entire concatenated corpus once
    corpus = " " + " ".join(transcripts.values())
    allowed.update(tok.encode(corpus))
    corpus_nospace = "".join(transcripts.values())
    allowed.update(tok.encode(corpus_nospace))

    return sorted(allowed)

def parse_category_from_stem(stem: str) -> str:
    """
    Extracts the text after 'category_' in the filename stem.

    Example:
      '00000_0.51_category_emo_adoration_sentences' -> 'emo_adoration_sentences'
      '00001_0.44_category_emo_anger_sentences'     -> 'emo_anger_sentences'
    """
    m = re.search(r'category_(.*)$', stem)
    return m.group(1) if m else stem

class WhisperGuidedScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-6, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=-1, 
                            help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="score_matching", help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--pesq_weight", type=float, default=0.0, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=48000, help="The sample rate of the audio files.")
        parser.add_argument("--lang", type=str, default="en", help="Language for Whisper decoding")
        parser.add_argument("--debug", type=bool, default=False, help="Whether to enable debug visualization during validation")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, data_module_cls=None, whisper_lang='en', model_mode="regular",
        whisper_name="base", guidance_scale=1.0, distillation_weight=1.0, debug=False,
        transcripts_path: str = "", **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize WER metric
        self.wer_metric = torchmetrics.WordErrorRate()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr
        self.debug = debug
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.options = whisper.DecodingOptions(language=whisper_lang, without_timestamps=True)
        self.whisper = whisper.load_model(whisper_name)
        self.multilingual_tokenizer = whisper.tokenizer.get_tokenizer(True, language=whisper_lang, task=self.options.task)
        if hasattr(self.whisper, 'alignment_heads') and self.whisper.alignment_heads.is_sparse:
        # Convert the sparse buffer to a dense buffer
            self.whisper.alignment_heads = self.whisper.alignment_heads.to_dense()    
        # Freeze Whisper parameters but keep gradient computation enabled
        for param in self.whisper.parameters():
            param.requires_grad = False
            
        # Cross entropy loss for ASR
        self.whisper_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        with open(transcripts_path, "r", encoding="utf-8") as f:
            self.transcripts = json.load(f)
        # self.allowed_words = load_transcripts_words(transcripts_path)
        tok = self.multilingual_tokenizer
        self.allowed_toks = build_allowed_token_id_set_with_tok(transcripts_path, tok)
        self.text_normalizer = EnglishTextNormalizer()

        # Initialize EMA with current parameter values
        self._initialize_ema()
    
    def _initialize_ema(self):
        """Initialize EMA with current parameter values to avoid random initialization"""
        with torch.no_grad():
            for p, avg in zip(self.dnn.parameters(), self.ema.shadow_params):
                avg.copy_(p.data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    
    def eval_without_ema(self):
        """Evaluate without using EMA parameters for direct comparison"""
        return self.train(False, no_ema=True)

    def _loss(self, forward_out, x_t, z, t, mean, x, masks):
        """
        Different loss functions can be used to train the score model, see the paper: 
        
        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """

        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z))  # [B,1,F,T]
            else:
                raise ValueError(f"Invalid loss weighting for loss_type=score_matching: {self.loss_weighting}")

            # masks: [B,1,1,T]  -> broadcast to [B,1,F,T]
            m_tf = masks.to(x.device, dtype=losses.dtype).expand(-1, 1, x.shape[2], -1).contiguous()

            B = x.size(0)
            loss = 0.5 * (losses * m_tf).sum(dim=(1, 2, 3)).mean()

        elif self.loss_type == "denoiser":
            score = forward_out
            D = score * sigma.pow(2) + x_t # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean)) # Eq. (8)
            if self.loss_weighting == "1":
                losses = losses
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                losses = ((sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2))[:, None, None, None] * losses
            else:
                raise ValueError("Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))     
        elif self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))

            # losses using PESQ
            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(losses_pesq)
                # combine the losses
                loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq 
            else:
                loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss

    def _step(self, batch, batch_idx):
        specs, masks = batch["specs"], batch["time_mask"]
        y = specs[:, 0:1]        # (B, 1, F, T)
        x = specs[:, 1:2]        # (B, 1, F, T)

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x, masks)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            ds = getattr(self.data_module, "test_set", None) if getattr(self.trainer, "testing", False) else self.data_module.valid_set
            if ds is None:
                ds = self.data_module.valid_set

            if self.num_eval_files < 0:
                eval_total = len(ds.clean_files)
            else:
                eval_total = min(self.num_eval_files, len(ds.clean_files))

            # Split the evaluation files among the GPUs
            eval_files_per_gpu = ceil(eval_total / world_size)

            clean_files = ds.clean_files[:eval_total]
            noisy_files = ds.noisy_files[:eval_total]

            # Select the files for this GPU
            if rank == world_size - 1:
                clean_files = clean_files[rank*eval_files_per_gpu:]
                noisy_files = noisy_files[rank*eval_files_per_gpu:]
            else:
                clean_files = clean_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]
                noisy_files = noisy_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]

            # -------------------- metrics containers --------------------
            pesq_sum = 0; pesq_cnt = 0; si_sdr_sum = 0; estoi_sum = 0
            o_list = []                             # enhanced 
            o_list_clean, o_list_corrupted = [], [] # clean/noisy 
            debug_seen_per_label = defaultdict(int)

            # -------------------- decode helpers ------------------------
            device = next(self.whisper.parameters()).device
            tok = self.multilingual_tokenizer
            n_vocab = tok.encoding.n_vocab

            allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=device)

            # a mask we will add to logits each step: allowed=0, disallowed=-inf
            neg_inf = -1e9
            base_mask = torch.full((n_vocab,), neg_inf, device=device)
            base_mask[allowed_ids_t] = 0.0

            # maximum tokens we’ll allow to emit (you can expose this as a hyperparam)
            max_decode_len = 224

            def decode_constrained(signal_16k):
                """
                Greedy, closed-set decode: at each step, mask logits to allowed set.
                Returns (pred_text, mel) where pred_text is the decoded string.
                """

                wav = torch.tensor(signal_16k, dtype=torch.float32, device=device)
                wav = self._pad_or_trim(wav)  # (T,)
                mel = self._log_mel_spectrogram(wav).unsqueeze(0)  # (1, 80, Tm)
                feats = self.whisper.encoder(mel)  # (1, T', C)

                # start with SOT + <|notimestamps|>
                prefix = torch.tensor([tok.sot_sequence_including_notimestamps],
                                    dtype=torch.long, device=device)  # (1, L0)
                out = prefix.clone()
                constrained_tokens = []

                for _ in range(max_decode_len):
                    # logits over full vocab for each position so far
                    logits = self.whisper.decoder(out, feats).squeeze(0)  # (L, V)
                    next_logits = logits[-1]  # (V,)
                    # record only the allowed positions' logits for this step -> (|A|,)
                    best_allowed_idx = int(torch.argmax(next_logits.index_select(0, allowed_ids_t)).item())
                    constrained_token_id = allowed_ids_t[best_allowed_idx].item()
                    constrained_tokens.append(constrained_token_id)

                    # greedy pick
                    next_id = int(torch.argmax(next_logits).item())
                    out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)

                    if constrained_token_id == tok.eot:
                        break

                if len(constrained_tokens) > 0 and constrained_tokens[-1] == tok.eot:
                    constrained_tokens = constrained_tokens[:-1]
                text = tok.decode(constrained_tokens)
                return text, mel
            
            def expand_contractions(s: str) -> str:
                """Convert only 'eight pm' to '8pm' and 'seventy five percent' to '75%'."""
                result = s
                # Convert "eight pm" -> "8pm" (case insensitive)
                result = re.sub(r'\beight\s*pm\b', '8pm', result, flags=re.IGNORECASE)
                # Convert "seventy five percent" -> "75%" (case insensitive)
                result = re.sub(r'\bseventy\s*five\s*percent\b', '75%', result, flags=re.IGNORECASE)
                # Also handle if there's already a % symbol: "seventy five %" -> "75%"
                result = re.sub(r'\bseventy\s*five\s*%', '75%', result, flags=re.IGNORECASE)
                return result

            # -------------------- main loop -----------------------------
            for i, (clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)):
                # Load the clean and noisy speech
                x, sr_x = load(clean_file)
                x = x.squeeze().numpy()
                y, sr_y = load(noisy_file)
                y = y.squeeze().numpy()
                assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

                # Resample for Whisper front-end
                if sr_x != 16000:
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
                    y_16k = resample(y, orig_sr=sr_y, target_sr=16000).squeeze()
                else:
                    x_16k = x
                    y_16k = y

                # # ground truth sentence
                stem = os.path.splitext(os.path.basename(clean_file))[0]
                category = parse_category_from_stem(stem)

                # closed-set ASR on noisy and clean
                corrupted_transcript, corrupted_mel = decode_constrained(y_16k)
                clean_transcript, clean_mel = decode_constrained(x_16k)

                # Enhance the noisy speech (keep native sr for enhancement)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
                x_hat = self.enhance(y_tensor.unsqueeze(0), N=self.sde.N, snr=0.5)

                # Resample enhanced to 16k for Whisper + metrics that expect 16k
                if sr_y != 16000:
                    x_hat_16k = resample(x_hat, orig_sr=sr_y, target_sr=16000).squeeze()
                else:
                    x_hat_16k = x_hat.detach().cpu().numpy() if torch.is_tensor(x_hat) else x_hat

                # closed-set ASR on enhanced
                enhanced_transcript, enhanced_mel = decode_constrained(x_hat_16k)
                enhanced_transcript_processed = expand_contractions(self.text_normalizer(enhanced_transcript))
                corrupted_transcript_processed = expand_contractions(self.text_normalizer(corrupted_transcript))
                clean_transcript_processed = expand_contractions(self.text_normalizer(clean_transcript))

                # --- collect WER strings ---
                o_list.append(enhanced_transcript_processed)
                o_list_corrupted.append(corrupted_transcript_processed)
                o_list_clean.append(clean_transcript_processed)

                # --- signal metrics ---
                try:
                    pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb')
                    pesq_cnt += 1
                except Exception:
                    pass

                si_sdr_sum += si_sdr(torch.tensor(x_16k), torch.tensor(x_hat_16k))
                estoi_sum += stoi(x_16k, x_hat_16k, self.sr, extended=True)

                # --- optional debug plots ---
                if self.debug and debug_seen_per_label[category] < 1:
                    debug_seen_per_label[category] += 1
                    self._plot_and_save_debug_info(
                        clean_mel=clean_mel,
                        corrupted_mel=corrupted_mel,
                        enhanced_mel=enhanced_mel,
                        clean_transcript=clean_transcript_processed,
                        corrupted_transcript=corrupted_transcript_processed,
                        enhanced_transcript=enhanced_transcript_processed,
                        ground_truth=clean_transcript_processed,
                        category=category,
                        index=debug_seen_per_label[category]
                    )

            # -------------------- aggregate + log ------------------------
            si_sdr_avg = si_sdr_sum / len(clean_files)
            estoi_avg = estoi_sum / len(clean_files)
            wer_enhanced  = self.wer_metric(o_list, o_list_clean)
            wer_corrupted = self.wer_metric(o_list_corrupted, o_list_clean)

            self.log('pesq',  (pesq_sum / pesq_cnt) if pesq_cnt > 0 else float('nan'), on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi',  estoi_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_enhanced',  wer_enhanced,  on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_corrupted', wer_corrupted, on_step=False, on_epoch=True, sync_dist=True)

        # keep your training loss path unchanged
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # If your validation_step returns logs/metrics, just reuse it
        return self.validation_step(batch, batch_idx)

    def forward(self, x_t, y, t):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat
                
        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)            
            score = -self.dnn(dnn_input, t)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
    
    def _pad_or_trim(self, x: torch.Tensor, length: int = 30 * 16_000):
        """
        Make every waveform exactly `length` samples (default = 30 s at 16 kHz).

        • Pads at the END with zeros if x is shorter.
        • Hard‑truncates if x is longer.
        """
        if x.size(-1) < length:
            x = F.pad(x, (0, length - x.size(-1)))
        else:
            x = x[..., :length]
        return x
    
    def _log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_fft: int = 400,
        hop: int = 160,
        n_mels: int = 80,
        sr: int = 16_000,
        f_min: float = 0.0,
        f_max: float = 8_000.0,
    ):
        """
        Parameters
        ----------
        audio : (T,) 1‑D float32 tensor in −1 … 1
        Returns
        -------
        logmel : (n_mels, 1 + ⌊(T‑n_fft)/hop⌋) tensor ‑‑ identical shape Whisper expects
        """
        # stereo → mono, dtype guard
        if audio.dim() == 2:
            audio = audio.mean(0)
        if audio.dtype != torch.float32:
            audio = audio.float()

        window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)

        # STFT → power
        stft = torch.stft(
            audio,
            n_fft,
            hop,
            window=window,
            win_length=n_fft,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().pow(2.0)

        # Mel filterbank (Whisper uses Slaney mel & Slaney norm)
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            sample_rate=sr,          # make it explicit
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            norm="slaney",
            mel_scale="slaney",
        ).to(power.device) 

        if fb.shape[0] != n_mels:    
            fb = fb.t().contiguous()

        mel = fb @ power
        mel = mel[:, :-1]
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))  
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == 'OUVESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False,
                    **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y.cuda(), sampler_type=self.sde.sampler_type)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(self.sde.__class__.__name__))

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
    
    def _plot_and_save_debug_info(self, clean_mel, corrupted_mel, enhanced_mel, clean_transcript, corrupted_transcript, enhanced_transcript, ground_truth, category, index):
        """
        Plot and save mel spectrograms and transcripts for debugging
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create debug directory if it doesn't exist
        debug_dir = "debug_plots"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Plot mel spectrograms
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        TIME_FRAMES = 300
        
        # Clean mel
        axes[0].imshow(clean_mel[:, :TIME_FRAMES].cpu().numpy(), aspect='auto', origin='lower')
        axes[0].set_title('Clean Mel Spectrogram')
        axes[0].set_ylabel('Mel Frequency Bands')
        
        # Corrupted mel
        axes[1].imshow(corrupted_mel[:, :TIME_FRAMES].cpu().numpy(), aspect='auto', origin='lower')
        axes[1].set_title('Corrupted Mel Spectrogram')
        axes[1].set_ylabel('Mel Frequency Bands')
        
        # Enhanced mel
        axes[2].imshow(enhanced_mel[:, :TIME_FRAMES].cpu().numpy(), aspect='auto', origin='lower')
        axes[2].set_title('Enhanced Mel Spectrogram')
        axes[2].set_ylabel('Mel Frequency Bands')
        axes[2].set_xlabel('Time Frames')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(debug_dir, f"mel_spectrograms_{category}_{index}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Save transcripts to a text file
        transcript_path = os.path.join(debug_dir, f"transcripts_{category}_{index}.txt")
        with open(transcript_path, 'w') as f:
            f.write(f"Ground Truth: {ground_truth}\n")
            f.write(f"Clean Transcript: {clean_transcript}\n")
            f.write(f"Corrupted Transcript: {corrupted_transcript}\n")
            f.write(f"Enhanced Transcript: {enhanced_transcript}\n")
