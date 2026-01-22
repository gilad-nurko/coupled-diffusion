import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
import torchmetrics
import whisper
import os

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss


class WhisperGuidedScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=1000, # was 20
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
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model variant to use (tiny, base, small, medium, large)")
        parser.add_argument("--lang", type=str, default="en", help="Language for Whisper decoding")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, data_module_cls=None, whisper_lang='en', model_mode="regular",
        whisper_name="base", guidance_scale=1.0, distillation_weight=1.0, **kwargs
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
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.allowed_words = ["down","go","left","no","off","on","right","stop","up","yes"]
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

        ids = [self.multilingual_tokenizer.encode(" " + w) for w in self.allowed_words]
        for w, tok in zip(self.allowed_words, ids):
            if len(tok) != 1:
                print(f"⚠️  '{w}' tokenises to {tok} – handle multi-token keywords!")
        self.allowed_toks = torch.tensor([tok[0] for tok in ids])

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

    def _loss(self, forward_out, x_t, z, t, mean, x):
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
                losses = torch.square(torch.abs(score * sigma + z)) # Eq. (7)
            else:
                raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
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
        specs, multilingual_tokens, labels = batch["specs"], batch["multilingual_tokens"], batch["labels"]
        y = specs[:, 0:1]        # (B, 1, F, T)
        x = specs[:, 1:2]        # (B, 1, F, T)

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss


    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     """
    #     1.  First GPU‑only batch (batch_idx == 0) → run objective metrics on
    #         `num_eval_files` held‑out wav files:
    #         • PESQ, SI‑SDR, ESTOI            (already present)
    #         • WER for clean / noisy / enhanced (new)
    #     2.  Always run the diffusion loss on the incoming batch for logging.
    #     """

    #     # ------------------------------------------------------------------ #
    #     # 1) OBJECTIVE METRICS ON HELD‑OUT AUDIO FILES (ONLY ON THE FIRST BATCH)
    #     # ------------------------------------------------------------------ #
    #     if batch_idx == 0 and self.num_eval_files != 0:
    #         rank, world_size = dist.get_rank(), dist.get_world_size()
    #         eval_per_gpu     = self.num_eval_files // world_size

    #         clean_files = self.data_module.valid_set.clean_files[:self.num_eval_files]
    #         noisy_files = self.data_module.valid_set.noisy_files[:self.num_eval_files]

    #         # shard across GPUs
    #         if rank == world_size - 1:          # last GPU takes the remainder
    #             clean_files = clean_files[rank*eval_per_gpu:]
    #             noisy_files = noisy_files[rank*eval_per_gpu:]
    #         else:
    #             start, end  = rank*eval_per_gpu, (rank+1)*eval_per_gpu
    #             clean_files = clean_files[start:end]
    #             noisy_files = noisy_files[start:end]

    #         # running sums
    #         pesq_sum = si_sdr_sum = estoi_sum = 0.0
    #         wer_clean_err = wer_noisy_err = wer_enh_err = 0      # wrong predictions
    #         total_cnt, pesq_cnt = 0, 0
    #         preds_clean, preds_noisy, preds_enh, refs = [], [], [], []

    #         device = next(self.whisper.parameters()).device      # where Whisper lives

    #         for clean_fp, noisy_fp in zip(clean_files, noisy_files):
    #             # ------------ load & (optionally) resample wavs ------------
    #             x, sr_x = load(clean_fp) ; x = x.squeeze().numpy()
    #             y, sr_y = load(noisy_fp) ; y = y.squeeze().numpy()
    #             assert sr_x == sr_y, "Mismatched sample‑rates!"

    #             # mono @ 16 kHz for metrics + ASR
    #             if sr_x != 16000:
    #                 x_16 = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
    #                 y_16 = resample(y, orig_sr=sr_y, target_sr=16000).squeeze()
    #             else:
    #                 x_16, y_16 = x, y

    #             # ------------ enhancement ------------
    #             y_torch = torch.tensor(y_16, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]
    #             x_hat = self.enhance(y_torch, N=self.sde.N, snr=0.33)         # returns wav @ self.sr
    #             if self.sr != 16000:
    #                 x_hat_16 = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
    #             else:
    #                 x_hat_16 = x_hat

    #             # ========== objective audio metrics ==========
    #             # pesq_sum   += pesq(16000, x_16, x_hat_16, 'wb')
    #             try:
    #                 pesq_sum  += pesq(16000, x_16, x_hat_16, 'wb')
    #                 pesq_cnt+=1
    #             except Exception:
    #                 pass
    #             si_sdr_sum += si_sdr(torch.tensor(x), torch.tensor(x_hat))
    #             estoi_sum  += stoi(x, x_hat, self.sr, extended=True)

    #             # ========== ASR‑based keyword prediction ==========
    #             # file name pattern:  "{keyword}_XX.wav"
    #             label_str = os.path.splitext(os.path.basename(noisy_fp))[0].split('_')[0].lower()

    #             # -- helper to get keyword probability vector for a wav ----------
    #             def _keyword_logits(wav16):
    #                 wav_padded = whisper.pad_or_trim(torch.tensor(wav16).flatten())
    #                 mel       = whisper.log_mel_spectrogram(wav_padded).to(device)
    #                 feats     = self.whisper.encoder(mel.unsqueeze(0))
    #                 # conditioning sequence built *once* from ground‑truth
    #                 return self.whisper.decoder(self._keyword_tokens[label_str], feats).squeeze()

    #             # build conditioning tokens once (cached as attribute)
    #             if not hasattr(self, "_keyword_tokens"):
    #                 self._keyword_tokens = {}
    #                 for w in self.allowed_words:
    #                     toks = [*self.tokenizer.sot_sequence_including_notimestamps] \
    #                         + self.tokenizer.encode(" " + w)
    #                     self._keyword_tokens[w] = torch.tensor(toks, device=device).unsqueeze(0)

    #             # logits for clean / noisy / enhanced
    #             log_clean = _keyword_logits(x_16)
    #             log_noisy = _keyword_logits(y_16)
    #             log_enh   = _keyword_logits(x_hat_16)

    #             # we care only about the *keyword* position:
    #             keyword_idx = len(self.tokenizer.sot_sequence_including_notimestamps)
    #             probs_clean = log_clean[keyword_idx][self.allowed_toks.to(device)]
    #             probs_noisy = log_noisy[keyword_idx][self.allowed_toks.to(device)]
    #             probs_enh   = log_enh  [keyword_idx][self.allowed_toks.to(device)]

    #             pred_clean = self.allowed_words[probs_clean.argmax().item()]
    #             pred_noisy = self.allowed_words[probs_noisy.argmax().item()]
    #             pred_enh   = self.allowed_words[probs_enh  .argmax().item()]

    #             # --------- WER for a single‑word reference -----------
    #             # wer_clean_err += int(pred_clean != label_str)
    #             # wer_noisy_err += int(pred_noisy != label_str)
    #             # wer_enh_err   += int(pred_enh   != label_str)
    #             preds_clean.append(pred_clean)
    #             preds_noisy.append(pred_noisy)
    #             preds_enh.append(pred_enh)
    #             refs.append(label_str)

    #             total_cnt += 1
    #         # -------------------- averages & logging --------------------
    #         n = len(clean_files)
    #         self.log('pesq',       pesq_sum  / pesq_cnt if pesq_cnt>0 else float('nan'), on_step=False, on_epoch=True, sync_dist=True)
    #         self.log('si_sdr',     si_sdr_sum    / n, on_step=False, on_epoch=True, sync_dist=True)
    #         self.log('estoi',      estoi_sum     / n, on_step=False, on_epoch=True, sync_dist=True)
    #         self.log('wer_clean',     self.wer_metric(preds_clean, refs),     on_step=False, on_epoch=True, sync_dist=True)
    #         self.log('wer_corrupted',     self.wer_metric(preds_noisy, refs),     on_step=False, on_epoch=True, sync_dist=True)
    #         self.log('wer_enhanced',      self.wer_metric(preds_enh,   refs),      on_step=False, on_epoch=True, sync_dist=True)

    #     # ------------------------------------------------------------------ #
    #     # 2) REGULAR DIFFUSION LOSS ON THE CURRENT BATCH
    #     # ------------------------------------------------------------------ #
    #     loss = self._step(batch, batch_idx)
    #     self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    #     return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # Split the evaluation files among the GPUs
            eval_files_per_gpu = self.num_eval_files // world_size

            # clean_files = self.data_module.valid_set.clean_files[:self.num_eval_files]
            # noisy_files = self.data_module.valid_set.noisy_files[:self.num_eval_files]
            if not hasattr(self, "eval_perm"):               # compute only the first time
                num_total = len(self.data_module.valid_set.clean_files)
                num_eval  = min(self.num_eval_files, num_total)

                g = torch.Generator().manual_seed(0)     # ← any fixed integer seed you like
                perm = torch.randperm(num_total, generator=g)[:num_eval].tolist()

                self.eval_perm = perm                        # cache for future epochs

            clean_files = [self.data_module.valid_set.clean_files[i] for i in self.eval_perm]
            noisy_files = [self.data_module.valid_set.noisy_files[i] for i in self.eval_perm]

            # Select the files for this GPU     
            if rank == world_size - 1:
                clean_files = clean_files[rank*eval_files_per_gpu:]
                noisy_files = noisy_files[rank*eval_files_per_gpu:]
            else:   
                clean_files = clean_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]
                noisy_files = noisy_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]  

            # Evaluate the performance of the model
            pesq_sum = 0; pesq_cnt = 0; si_sdr_sum = 0; estoi_sum = 0; 
            # lists for WER (exactly as before)
            o_list, l_list = [], []
            o_list_clean, o_list_corrupted = [], []
            # extra containers
            correct_cnt_clean, correct_cnt_corrupted, correct_cnt_enhanced, total_cnt = 0, 0, 0, 0
            self.allowed_toks = self.allowed_toks.to(self.whisper.device)

            # helper: fast mask of special tokens
            special_ids = {
                self.multilingual_tokenizer.eot,
                self.multilingual_tokenizer.encode("<|en|>",           allowed_special={"<|en|>"})[0],
                self.multilingual_tokenizer.encode("<|transcribe|>",   allowed_special={"<|transcribe|>"})[0],
                self.multilingual_tokenizer.encode("<|notimestamps|>", allowed_special={"<|notimestamps|>"})[0],
                self.multilingual_tokenizer.encode("<|nospeech|>",     allowed_special={"<|nospeech|>"})[0],
            }
            for (clean_file, noisy_file) in zip(clean_files, noisy_files):
                # Load the clean and noisy speech
                x, sr_x = load(clean_file)
                x = x.squeeze().numpy()
                y, sr_y = load(noisy_file) 
                assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

                # Resample if necessary
                if sr_x != 16000:
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
                    y_16k = resample(y, orig_sr=sr_y, target_sr=16000).squeeze()
                else:
                    x_16k = x
                    y_16k = y
                
                label_str = os.path.splitext(os.path.basename(clean_file))[0].split('_')[0]
                tokens = [*self.multilingual_tokenizer.sot_sequence_including_notimestamps] \
                        + self.multilingual_tokenizer.encode(" " + label_str)
                labels = tokens[1:] + [self.multilingual_tokenizer.eot]
                multilingual_tokens = torch.tensor(tokens, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.long)

                multilingual_tokens = multilingual_tokens.to(self.whisper.device)
                labels = labels.to(self.whisper.device)

                # ---- locate keyword position once per utterance ----------
                keyword_idx = next(j for j, t in enumerate(labels.tolist()) if t not in special_ids)

                def run_whisper(signal_16k, multilingual_tokens, labels, keyword_idx):
                    # 1) move the waveform to the same device Whisper lives on
                    device = next(self.whisper.parameters()).device
                    signal  = torch.tensor(signal_16k, dtype=torch.float32, device=device)

                    # 2) Whisper’s utility helpers
                    signal_padded = whisper.pad_or_trim(signal).flatten()      # (T,)
                    mel           = whisper.log_mel_spectrogram(signal_padded).to(device)  # (80, T)

                    # 3) Encoder‑decoder forward
                    feats  = self.whisper.encoder(mel.unsqueeze(0))            # (1, …, T’)
                    tokens = multilingual_tokens.to(device).unsqueeze(0)       # (1, L)
                    logits = self.whisper.decoder(tokens, feats).squeeze(0)    # (L, vocab)

                    # 4) Cross‑entropy w.r.t. the ground‑truth label sequence
                    ce = self.whisper_loss(logits.view(-1, logits.size(-1)),
                                    labels.to(device).view(-1)).detach().cpu()

                    # 5) Closed‑set keyword probabilities
                    kw_logits = logits[keyword_idx]            # vector over entire vocab
                    kw_probs  = kw_logits.softmax(-1)          # turn into probabilities
                    kw_probs  = kw_probs[self.allowed_toks.to(device)]  # length‑10

                    pred_id = kw_probs.argmax().item()
                    return ce, pred_id, kw_probs

                
                ce_loss_corrupted, pred_id_cor, kw_probs_cor = run_whisper(y_16k, multilingual_tokens, labels, keyword_idx)
                ce_loss_clean, pred_id_cln, kw_probs_cln = run_whisper(x_16k, multilingual_tokens, labels, keyword_idx)

                # Enhance the noisy speech
                device = next(self.whisper.parameters()).device
                y_tensor = torch.tensor(y_16k, dtype=torch.float32, device=device)
                x_hat = self.enhance(y_tensor, N=self.sde.N)
                if self.sr != 16000:
                    x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
                else:
                    x_hat_16k = x_hat    
                
                ce_loss_enhanced, pred_id_enh, kw_probs_enh = run_whisper(x_hat_16k, multilingual_tokens, labels, keyword_idx)
                true_id = (self.allowed_toks == labels[keyword_idx]).nonzero(as_tuple=True)[0].item()
                ground_truth = self.allowed_words[true_id]
                # --- closed-set accuracy ----------------------------------
                total_cnt   += 1
                correct_cnt_enhanced += int(pred_id_enh == true_id)

                # --- WER lists (as close to original as possible) ---------
                enhanced_transcript  = self.allowed_words[pred_id_enh]
                corrupted_transcript = self.allowed_words[pred_id_cor]
                clean_transcript     = self.allowed_words[pred_id_cln]

                o_list.append(enhanced_transcript)
                o_list_corrupted.append(corrupted_transcript)
                o_list_clean.append(clean_transcript)
                l_list.append(ground_truth)
        
                # pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb') 
                try:
                    pesq_sum  += pesq(16000, x_16k, x_hat_16k, 'wb')
                    pesq_cnt+=1
                except Exception:
                    pass
                # si_sdr_sum += si_sdr(x, x_hat)
                si_sdr_sum += si_sdr(torch.tensor(x), torch.tensor(x_hat))
                estoi_sum += stoi(x, x_hat, self.sr, extended=True)

            pesq_avg = pesq_sum / len(clean_files)
            si_sdr_avg = si_sdr_sum / len(clean_files)
            estoi_avg = estoi_sum / len(clean_files)
            # --------------- metrics (names kept) -------------------------
            wer_clean     = self.wer_metric(o_list_clean, l_list)
            wer_enhanced  = self.wer_metric(o_list, l_list)
            wer_corrupted = self.wer_metric(o_list_corrupted, l_list)
            keyword_acc = torch.tensor(correct_cnt_enhanced / max(total_cnt, 1), device=self.device)

            self.log('pesq',pesq_sum / pesq_cnt if pesq_cnt>0 else float('nan'), on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi', estoi_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_enhanced',  wer_enhanced,  on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_clean',     wer_clean,     on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_corrupted', wer_corrupted, on_step=False, on_epoch=True, sync_dist=True)
            self.log('keyword_acc',  keyword_acc, on_step=False, on_epoch=True, sync_dist=True)

        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss




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
