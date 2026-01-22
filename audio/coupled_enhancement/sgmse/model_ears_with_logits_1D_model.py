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
from torchaudio.functional import resample as ta_resample
import torchmetrics
import whisper
import os, re
import torch.nn.functional as F
import torchaudio
from typing import List, Tuple, Optional
from tqdm import tqdm
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr, build_allowed_token_id_set_with_tok, normalize_logits, parse_category_from_stem
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
# from sgmse.backbones.logits_model import LogitsDenoiser
from sgmse.backbones.residual_logits_model import LogitsDenoiser
from whisper.normalizers import EnglishTextNormalizer


class WhisperGuidedScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-6, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=-1, # was 20
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
        parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model variant to use (tiny, base, small, medium, large)")
        parser.add_argument("--lang", type=str, default="en", help="Language for Whisper decoding")
        parser.add_argument("--debug", type=bool, default=False, help="Whether to enable debug visualization during validation")
        parser.add_argument("--sampling_mode", type=str, default="nested", choices=["parallel", "full", "nested"],
                              help="Sampling mode: 'parallel' for parallel sampling, 'full' for full x then full y sampling, 'nested' for nested sampling.")
        parser.add_argument("--num_iterations", type=int, default=5,
                              help="Number of iterations for sampling in full mode.")
        parser.add_argument("--logits_diffusion_steps", type=int, default=5,
                              help="Number of diffusion steps for the logits in the nested mode.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, data_module_cls=None, whisper_lang='en', model_mode="regular",
        whisper_name="base", guidance_scale=1.0, distillation_weight=1.0, logits_weight=1.0, sampling_mode="full", 
        num_iterations=5, logits_diffusion_steps=5, debug=False, transcripts_path: str = "/mlspeech/data/gilad/paper_ears_reverbed/transcripts.json",
        logits_pretrain_ckpt: str="/mlspeech/data/gilad/logs/ASR_diffusion_ears/with_logits/logits_model_pretrain.pt", **kwargs
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
        # Initialize allowed token IDs for Whisper
        self.options = whisper.DecodingOptions(language=whisper_lang, without_timestamps=True)
        self.whisper = whisper.load_model(whisper_name)
        self.multilingual_tokenizer = whisper.tokenizer.get_tokenizer(True, language=whisper_lang, task=self.options.task)
        tok = self.multilingual_tokenizer
        self.allowed_toks = build_allowed_token_id_set_with_tok(transcripts_path, tok)
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        kwargs["logits_dim"] = len(self.allowed_toks)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
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
        self.logits_weight = logits_weight
        self.sr = sr
        self.debug = debug
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
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
        self.text_normalizer = EnglishTextNormalizer()
        self.sampling_mode = sampling_mode
        self.num_iterations = num_iterations
        self.logits_diffusion_steps = logits_diffusion_steps
        self.logits_pretrain_ckpt = logits_pretrain_ckpt
        self._load_logits_on_fit_start = True 
        self._did_load_logits_ckpt = False
        self.logits_model = LogitsDenoiser(
            logits_size=len(self.allowed_toks)
        )
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        # # Initialize EMA with current parameter values
        # self._initialize_ema()
    
    def _initialize_ema(self):
        """Initialize EMA with current parameter values (including logits_model, which will be random)."""
        with torch.no_grad():
            for p, avg in zip(self.parameters(), self.ema.shadow_params):
                avg.copy_(p.data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

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
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    
    def extract_logits(self, y, lengths, target_mean=None, target_std=None):
        """
        Regular (unconstrained) Whisper decode for control, with closed-set readout.
        Stop criterion: constrained stream's EOT only.

        Returns:
            mel:             [B, 80, Tm]
            allowed_logits:  [B, S, |A|]   (per-step scores over allowed set; finished rows one-hot @EOT)
            mask:            [B, S]        (True where step is valid; ends at constrained EOT)
            audio_embedding: [B, Tenc', D]
            transcripts:     List[str]     (decoded from constrained tokens only)
        """
        whisper_dev = next(self.whisper.parameters()).device
        self.whisper.eval()

        # -------------------- tokenizer / vocab --------------------
        tok = self.multilingual_tokenizer
        n_vocab = tok.encoding.n_vocab

        # Allowed token ids tensor (closed set)
        allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=whisper_dev)

        # EOT must be in the closed set (to be able to stop)
        eot_in_allowed = (allowed_ids_t == tok.eot).nonzero(as_tuple=True)[0]
        if eot_in_allowed.numel() == 0:
            raise ValueError("tok.eot is not in allowed_ids_t; include EOT in the closed set.")
        eot_pos_in_allowed = int(eot_in_allowed.item())

        max_decode_len = 224
        B = y.shape[0]

        # -------------------- (1) build batch of 16k audios --------------------
        y_audio_list_padded, true_audio_lengths_16k = [], []
        for i in range(B):
            spec_single = y[i].squeeze(0)  # (F, T)
            audio_single = self.to_audio(spec_single, lengths[i].item()).to(dtype=torch.float32)
            if self.sr != 16000:
                audio_single_16k = ta_resample(audio_single.unsqueeze(0), self.sr, 16000).squeeze(0).to(whisper_dev)
            else:
                audio_single_16k = audio_single.to(whisper_dev)
            true_audio_lengths_16k.append(int(audio_single_16k.shape[-1]))
            y_audio_list_padded.append(self._pad_or_trim(audio_single_16k))

        y_audio_padded = torch.stack(y_audio_list_padded, dim=0).to(whisper_dev)  # (B, T_audio)

        # -------------------- (2) mel + encoder --------------------
        mel   = self._log_mel_spectrogram(y_audio_padded)   # (B, 80, Tm) on whisper_dev
        feats = self.whisper.encoder(mel)                   # (B, Tenc, D)

        hop = getattr(self, "mel_hop_length", 160)
        pad_target_len = y_audio_padded.shape[-1]
        true_audio_lengths_16k = [min(L, pad_target_len) for L in true_audio_lengths_16k]
        true_mel_lengths  = [(L + hop - 1) // hop for L in true_audio_lengths_16k]

        enc_down = 2  # Whisper downsamples time by 2
        true_enc_lengths = [(m + enc_down - 1) // enc_down for m in true_mel_lengths]
        max_true_enc = max(true_enc_lengths)
        audio_embedding = feats[:, :max_true_enc, :]       # (B, max_true_enc, D)

        # -------------------- (3) init tokens (batched) --------------------
        prefix_1 = torch.tensor([tok.sot_sequence_including_notimestamps],
                                dtype=torch.long, device=whisper_dev)  # (1, Lp)
        out = prefix_1.repeat(B, 1)                                     # (B, Lp)

        # Trackers
        active_control     = torch.ones(B, dtype=torch.bool, device=whisper_dev)
        active_constrained = torch.ones(B, dtype=torch.bool, device=whisper_dev)

        first_ctrl_eot_step   = torch.full((B,), -1, dtype=torch.long, device=whisper_dev)
        first_constr_eot_step = torch.full((B,), -1, dtype=torch.long, device=whisper_dev)

        constrained_tokens = [[] for _ in range(B)]
        step_allowed = []  # list of (B, |A|) per step

        # -------------------- (4) autoregressive loop --------------------
        for s in range(max_decode_len):
            # logits over vocab (control path = unconstrained)
            logits_full = self.whisper.decoder(out, feats)      # (B, cur_len, V)
            next_logits_full = logits_full[:, -1, :]            # (B, V)

            # closed-set slice for readout/logging
            allowed_raw  = next_logits_full[:, allowed_ids_t]   # (B, |A|)
            allowed_next = normalize_logits(allowed_raw, target_mean, target_std)  # (B, |A|)

            # ----- control greedy pick (regular decode) -----
            next_ids = torch.argmax(next_logits_full, dim=1)    # (B,)
            out = torch.cat([out, next_ids.unsqueeze(1)], dim=1)

            # control completion bookkeeping (not used to stop the loop)
            newly_ctrl_finished = (next_ids == tok.eot) & active_control
            if newly_ctrl_finished.any():
                first_ctrl_eot_step[newly_ctrl_finished] = s
            active_control &= ~newly_ctrl_finished

            # ----- FORCE constrained EOT for rows where control just finished -----
            # This prevents hanging when control ends earlier; we still stop by constrained EOT only.
            if newly_ctrl_finished.any():
                allowed_next = allowed_next.clone()
                allowed_next[newly_ctrl_finished] = 0.0
                allowed_next[newly_ctrl_finished, eot_pos_in_allowed] = 1.0

            # One-hot EOT for rows already constrained-finished in prior steps
            if not active_constrained.all():
                finished_rows = ~active_constrained
                one_hot_finished = allowed_next.new_zeros((B, allowed_next.size(1)))
                one_hot_finished[:, eot_pos_in_allowed] = 1.0
                allowed_next = torch.where(
                    finished_rows.unsqueeze(1),
                    one_hot_finished,
                    allowed_next
                )

            # Log per-step allowed scores
            step_allowed.append(allowed_next)

            # ----- constrained greedy pick on CLOSED SET (after any forcing) -----
            best_allowed_idx = torch.argmax(allowed_next, dim=1)   # (B,)
            best_allowed_tok = allowed_ids_t[best_allowed_idx]      # (B,)

            # Append tokens only for rows still constrained-active
            for b in range(B):
                if active_constrained[b]:
                    constrained_tokens[b].append(int(best_allowed_tok[b].item()))

            # constrained completion & LOOP STOP: stop only when constrained hits EOT
            newly_constr_finished = (best_allowed_tok == tok.eot) & active_constrained
            if newly_constr_finished.any():
                first_constr_eot_step[newly_constr_finished] = s
            active_constrained &= ~newly_constr_finished

            if not active_constrained.any():
                break

        # -------------------- (5) stack outputs & mask --------------------
        allowed_logits = torch.stack(step_allowed, dim=0).permute(1, 0, 2).contiguous()  # [B, S, |A|]
        S = allowed_logits.shape[1]

        # Valid length = step index of constrained EOT (if none, S)
        constr_len = torch.where(
            first_constr_eot_step >= 0,
            first_constr_eot_step + 1,
            torch.as_tensor(S, device=whisper_dev)
        )
        ar = torch.arange(S, device=whisper_dev).unsqueeze(0)  # [1, S]
        mask = ar < constr_len.unsqueeze(1)                    # [B, S] boolean

        # # -------------------- (6) decode constrained transcripts --------------------
        # transcripts = []
        # for b in range(B):
        #     toks = constrained_tokens[b]
        #     if toks and toks[-1] == tok.eot:
        #         toks = toks[:-1]
        #     transcripts.append(tok.decode(toks))

        return mel, allowed_logits, mask, audio_embedding#, transcripts
    
    # def extract_logits(self, y, lengths):
    #     """
    #     Greedy, closed-set decode for a batch.
    #     Returns (mel[B,80,Tm], allowed_logits[B,S,|A|]).
    #     """
    #     whisper_dev = next(self.whisper.parameters()).device
    #     self.whisper.eval()
    #     # -------------------- decode helpers ------------------------
    #     tok = self.multilingual_tokenizer
    #     n_vocab = tok.encoding.n_vocab

    #     # self.allowed_toks may be a Python list; turn into a device tensor once
    #     allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=whisper_dev)

    #     # a mask we will add to logits each step: allowed=0, disallowed=-inf
    #     neg_inf = -1e9
    #     base_mask = torch.full((n_vocab,), neg_inf, device=whisper_dev)
    #     base_mask[allowed_ids_t] = 0.0

    #     # maximum tokens we’ll allow to emit (you can expose this as a hyperparam)
    #     max_decode_len = 224
    #     B = y.shape[0]

    #     # --------- (1) build batch of 16k audios as you already do ----------
    #     y_audio_list_padded = []
    #     true_audio_lengths_16k = []
    #     for i in range(B):
    #         spec_single = y[i].squeeze(0)  # (F, T)
    #         audio_single = self.to_audio(spec_single, lengths[i].item()).to(dtype=torch.float32)
    #         if self.sr != 16000:
    #             audio_single_16k = ta_resample(audio_single.unsqueeze(0), self.sr, 16000).squeeze(0).to(whisper_dev)
    #         else:
    #             audio_single_16k = audio_single.to(whisper_dev)
    #         true_audio_lengths_16k.append(int(audio_single_16k.shape[-1]))
    #         y_audio_list_padded.append(self._pad_or_trim(audio_single_16k))

    #     # (B, T_audio) on Whisper device
    #     y_audio_padded = torch.stack(y_audio_list_padded, dim=0).to(whisper_dev)

    #     # --------- (2) batched mel + encoder ----------
    #     mel = self._log_mel_spectrogram(y_audio_padded)           # (B, 80, Tm) on whisper_dev
    #     feats = self.whisper.encoder(mel)                          # (B, Tenc, D)

    #     hop = getattr(self, "mel_hop_length", 160)                 # Whisper default hop
    #     pad_target_len = y_audio_padded.shape[-1]                  # in samples @16k

    #     true_audio_lengths_16k = [min(L, pad_target_len) for L in true_audio_lengths_16k]   # cap if _pad_or_trim truncated
    #     true_mel_lengths  = [(L + hop - 1) // hop for L in true_audio_lengths_16k]         # ceil(L/hop)

    #     enc_down = 4  # Whisper conv front-end downsamples time by 2*2
    #     true_enc_lengths = [(m + enc_down - 1) // enc_down for m in true_mel_lengths]      # ceil(m/4)
    #     max_true_enc = max(true_enc_lengths)

    #     # --- NEW: slice encoder output to the longest real time step only ---
    #     audio_embedding = feats[:, :max_true_enc, :]   # (B, max_true_enc, C)

    #     # --------- (3) init tokens (batched) ----------
    #     prefix_1 = torch.tensor([tok.sot_sequence_including_notimestamps],
    #                             dtype=torch.long, device=whisper_dev)  # (1, Lp)
    #     out = prefix_1.repeat(B, 1)                                    # (B, Lp)

    #     # tracking finished sequences
    #     active = torch.ones(B, dtype=torch.bool, device=whisper_dev)
    #     step_logits = []  # each item: (B, |A|)
    #     first_eot_step = torch.full((B,), -1, dtype=torch.long, device=whisper_dev)
    #     match = (allowed_ids_t == tok.eot).nonzero(as_tuple=True)[0]
    #     if match.numel() == 0:
    #         raise ValueError("tok.eot is not in allowed_ids_t; cannot write 1-hot for finished rows.")
    #     eot_pos_in_allowed = int(match.item())
    #     # --------- (4) autoregressive loop (batched) ----------
    #     for s in range(max_decode_len):
    #         # logits over vocab for each position so far
    #         logits_full = self.whisper.decoder(out, feats)  # (B, cur_len, V)
    #         next_logits_full = logits_full[:, -1, :]        # (B, V)
    #         # allowed_next = torch.softmax(next_logits_full[:, allowed_ids_t], dim=-1)  # (B, |A|)
    #         allowed_next = normalize_logits(next_logits_full[:, allowed_ids_t], audio=y.to(next_logits_full.device))  # (B, |A|)

    #         # Overwrite finished rows with one-hot (1 at EOT, 0 elsewhere)
    #         if not active.all():
    #             # Create a zeros tensor matching dtype/device
    #             one_hot_finished = allowed_next.new_zeros((allowed_next.size(0), allowed_next.size(1)))
    #             # Set EOT position to 1 for finished rows
    #             one_hot_finished[~active, eot_pos_in_allowed] = 1.0
    #             # For active rows keep real values; for finished rows use 1-hot
    #             # Expand active to (B, 1) for broadcasting
    #             mask_active = active.unsqueeze(1)
    #             allowed_next = torch.where(mask_active, allowed_next, one_hot_finished)

    #         step_logits.append(allowed_next)

    #         # apply closed-set mask (broadcast base_mask[V] over batch)
    #         masked = next_logits_full + base_mask  # (B, V)

    #         # force already-finished rows to keep emitting EOT
    #         if not active.all():
    #             masked = masked.clone()
    #             masked[~active] = neg_inf
    #             masked[~active, tok.eot] = 1e9  # guarantee EOT wins

    #         # greedy pick per row
    #         next_ids = torch.argmax(masked, dim=1)  # (B,)

    #         # append to tokens
    #         out = torch.cat([out, next_ids.unsqueeze(1)], dim=1)  # (B, cur_len+1)

    #         # update finished mask
    #         newly_finished = (next_ids == tok.eot) & active
    #         if newly_finished.any():                    
    #             first_eot_step[newly_finished] = s
    #         active = active & (~newly_finished)

    #         if not active.any():
    #             break

    #     # stack allowed logits to [B, S, |A|]
    #     allowed_logits = torch.stack(step_logits, dim=0).permute(1, 0, 2).contiguous()
    #     S = allowed_logits.shape[1]
    #     valid_lengths = torch.where(
    #         first_eot_step >= 0,
    #         first_eot_step + 1,                     # include the EOT step itself
    #         torch.as_tensor(S, device=whisper_dev)  # if no EOT emitted
    #     )
    #     ar = torch.arange(S, device=whisper_dev).unsqueeze(0)  # [1,S]
    #     mask = ar < valid_lengths.unsqueeze(1)                 # [B,S] boolean
    #     return mel, allowed_logits, mask, audio_embedding
    
    
    # def pad_align_logits_with_eot(self,
    #     clean_logits: torch.Tensor,  # [B, S1, C]
    #     clean_mask:   torch.Tensor,  # [B, S1]  (bool: True=real step, False=pad)
    #     noisy_logits: torch.Tensor,  # [B, S2, C]
    #     noisy_mask:   torch.Tensor,  # [B, S2]  (bool)
    # ):
    #     """
    #     Pads the *shorter* sequence (per batch *uniformly* to the global max S)
    #     with a 1-hot vector at the EOT index. Adjusts masks accordingly.
    #     Returns:
    #     clean_logits_p, clean_mask_p, noisy_logits_p, noisy_mask_p, union_mask
    #     Shapes after: [B, S_max, C] for logits, [B, S_max] for masks.
    #     """
    #     device = clean_logits.device
    #     dtype  = clean_logits.dtype
    #     tok = self.multilingual_tokenizer
    #     allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=device)
    #     B, S1, C = clean_logits.shape
    #     _, S2, C2 = noisy_logits.shape
    #     assert C == C2, f"Channel (|A|) mismatch: clean C={C}, noisy C={C2}"

    #     # ----- find EOT index inside allowed ids -----
    #     match = (allowed_ids_t == tok.eot).nonzero(as_tuple=True)[0]
    #     if match.numel() == 0:
    #         raise ValueError("tok.eot is not in allowed_ids_t; cannot write 1-hot for padded rows.")
    #     eot_pos_in_allowed = int(match.item())

    #     # 1-hot vector at EOT for padding rows
    #     eot_onehot = torch.zeros(C, device=device, dtype=dtype)
    #     eot_onehot[eot_pos_in_allowed] = 1.0  # you can choose another pad value if you prefer

    #     S_max = max(S1, S2)
    #     if S_max == S1 == S2:
    #         # No padding needed; still provide union mask
    #         union_mask = clean_mask | noisy_mask
    #         return clean_logits, noisy_logits, union_mask

    #     # ----- pad CLEAN up to S_max -----
    #     if S1 < S_max:
    #         pad_steps = S_max - S1
    #         pad_logits_clean = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
    #         clean_logits = torch.cat([clean_logits, pad_logits_clean], dim=1)
    #         clean_mask   = torch.cat([clean_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)], dim=1)
    #     else:
    #         # already S_max; ensure mask shape is consistent
    #         assert clean_logits.shape[1] == S_max and clean_mask.shape[1] == S_max

    #     # ----- pad NOISY up to S_max -----
    #     if S2 < S_max:
    #         pad_steps = S_max - S2
    #         pad_logits_noisy = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
    #         noisy_logits = torch.cat([noisy_logits, pad_logits_noisy], dim=1)
    #         noisy_mask   = torch.cat([noisy_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)], dim=1)
    #     else:
    #         assert noisy_logits.shape[1] == S_max and noisy_mask.shape[1] == S_max

    #     # ----- combined mask (requested): clean_mask OR noisy_mask -----
    #     union_mask = clean_mask | noisy_mask

    #     return clean_logits, noisy_logits, union_mask
    def pad_align_logits_with_eot(self,
            clean_logits: torch.Tensor,  # [B, S1, C]
            clean_mask:   torch.Tensor,  # [B, S1]
            noisy_logits: torch.Tensor,  # [B, S2, C]
            noisy_mask:   torch.Tensor,  # [B, S2]
        ):
        """
        Pads the *shorter* sequence (per batch *uniformly* to the global max S)
        with a 1-hot vector at the EOT index. Adjusts masks accordingly.
        Returns:
            clean_logits_p, clean_mask_p, noisy_logits_p, noisy_mask_p, union_mask
        Shapes after: [B, S_max, C] for logits, [B, S_max] for masks.
        """
        device = clean_logits.device
        dtype  = clean_logits.dtype
        tok = self.multilingual_tokenizer
        allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=device)

        B, S1, C  = clean_logits.shape
        _, S2, C2 = noisy_logits.shape

        assert C == C2, f"Channel (|A|) mismatch: clean C={C}, noisy C={C2}"
        assert clean_mask.shape[0] == B and noisy_mask.shape[0] == B, "Batch mismatch logits/masks"
        assert clean_mask.dim() == 2 and noisy_mask.dim() == 2, "Masks must be [B, S]"

        # Allow masks to be shorter/longer than logits only if we are about to pad them
        # but we ALWAYS measure lengths from logits
        S_max = max(S1, S2)

        # ----- find EOT index inside allowed ids -----
        match = (allowed_ids_t == tok.eot).nonzero(as_tuple=True)[0]
        if match.numel() == 0:
            raise ValueError("tok.eot is not in allowed_ids_t; cannot write 1-hot for padded rows.")
        eot_pos_in_allowed = int(match.item())

        # 1-hot vector at EOT for padding rows
        eot_onehot = torch.zeros(C, device=device, dtype=dtype)
        eot_onehot[eot_pos_in_allowed] = 1.0

        # Early no-pad path, but now we also ensure masks match S1,S2
        if S_max == S1 == S2 and clean_mask.shape[1] == S1 and noisy_mask.shape[1] == S2:
            union_mask = clean_mask | noisy_mask
            return clean_logits, clean_mask, noisy_logits, noisy_mask, union_mask

        # ----- pad CLEAN up to S_max -----
        if S1 < S_max:
            pad_steps = S_max - S1
            pad_logits_clean = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
            clean_logits = torch.cat([clean_logits, pad_logits_clean], dim=1)
            clean_mask   = torch.cat(
                [clean_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)],
                dim=1
            )
        else:
            # if logits already S_max, we must expand mask if needed
            if clean_mask.shape[1] < S_max:
                pad_steps = S_max - clean_mask.shape[1]
                clean_mask = torch.cat(
                    [clean_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)],
                    dim=1
                )
            assert clean_logits.shape[1] == S_max and clean_mask.shape[1] == S_max

        # ----- pad NOISY up to S_max -----
        if S2 < S_max:
            pad_steps = S_max - S2
            pad_logits_noisy = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
            noisy_logits = torch.cat([noisy_logits, pad_logits_noisy], dim=1)
            noisy_mask   = torch.cat(
                [noisy_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)],
                dim=1
            )
        else:
            if noisy_mask.shape[1] < S_max:
                pad_steps = S_max - noisy_mask.shape[1]
                noisy_mask = torch.cat(
                    [noisy_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)],
                    dim=1
                )
            assert noisy_logits.shape[1] == S_max and noisy_mask.shape[1] == S_max

        # ----- combined mask -----
        union_mask = clean_mask | noisy_mask  # now both [B, S_max]

        return clean_logits, clean_mask, noisy_logits, noisy_mask, union_mask


    # def _load_pretrained_logits_if_available(self) -> bool:
    #     """Returns True if loaded successfully (and pretraining can be skipped)."""
    #     path = self.logits_pretrain_ckpt
    #     if not path:
    #         return False
    #     if not os.path.exists(path):
    #         return False
    #     try:
    #         ckpt = torch.load(path, map_location="cpu")
    #         state = ckpt.get("state_dict", ckpt)
            
    #         # Strip the "logits_model." prefix if it exists (PyTorch Lightning format)
    #         prefix = "logits_model."
    #         filtered_state = {}
    #         for key, value in state.items():
    #             if key.startswith(prefix):
    #                 new_key = key[len(prefix):]  # Remove prefix
    #                 filtered_state[new_key] = value
    #             else:
    #                 filtered_state[key] = value
            
    #         # Load with strict=False
    #         missing_keys, unexpected_keys = self.logits_model.load_state_dict(filtered_state, strict=False)
            
    #         print(f"[logits pretrain] Successfully loaded {len(filtered_state) - len(unexpected_keys)} parameters from {path}")
            
    #         if missing_keys:
    #             print(f"[logits pretrain] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")  # Show first 5
    #         if unexpected_keys:
    #             print(f"[logits pretrain] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")  # Show first 5
                
    #         return True
    #     except Exception as e:
    #         print(f"[logits pretrain] Failed to load {path}: {e}")
    #         return False
    
    # def load_logits_denoiser_from_pl_ckpt(model, ckpt_path, prefix="logits_model."):
    #     ckpt = torch.load(ckpt_path, map_location="cpu")

    #     # Lightning usually stores under "state_dict"; fall back to raw dict if needed
    #     state_dict = ckpt.get("state_dict", ckpt)

    #     # Filter only logits_model.* and strip the prefix
    #     filtered = {}
    #     for k, v in state_dict.items():
    #         if k.startswith(prefix):
    #             new_k = k[len(prefix):]  # drop "logits_model."
    #             filtered[new_k] = v

    #     # Try loading
    #     missing, unexpected = model.load_state_dict(filtered, strict=False)

    #     print(f"[logits pretrain] loaded {len(filtered)} tensors into LogitsDenoiser "
    #         f"(missing={len(missing)}, unexpected={len(unexpected)})")
    #     if missing:
    #         print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
    #     if unexpected:
    #         print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    #     return model
    
    def on_fit_start(self):
        super().on_fit_start()
        if not self._load_logits_on_fit_start or self._did_load_logits_ckpt:
            return
        path = (self.logits_pretrain_ckpt or "").strip()
        if not path:
            if self.global_rank == 0:
                print("[logits pretrain] no path provided; skipping.")
            return
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Drop any whisper/tokenizer keys (safety)
            bad = ("whisper", "whisper_model", "tokenizer", "allowed_ids_t")
            state = {k: v for k, v in state.items() if not any(k.startswith(b + ".") for b in bad)}
            # Prefer subtree logits_model.*
            only_logits = {k[len("logits_model.") :]: v
                           for k, v in state.items() if k.startswith("logits_model.")}
            if not only_logits:
                # maybe the ckpt already stores bare logits weights
                only_logits = state

            missing, unexpected = self.logits_model.load_state_dict(only_logits, strict=False)
            if self.global_rank == 0:
                total = len(only_logits)
                print(f"[logits pretrain] loaded logits weights from {path}")
                print(f"[logits pretrain] total keys considered: {total}")
                if missing:
                    print(f"[logits pretrain] missing ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
                if unexpected:
                    print(f"[logits pretrain] unexpected ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
            self._did_load_logits_ckpt = True
        except Exception as e:
            if self.global_rank == 0:
                print(f"[logits pretrain] FAILED to load {path}: {e}")


    # def on_fit_start(self):
    #     # If already trained & cached → just load & return
    #     if self._load_pretrained_logits_if_available():
    #         if self.trainer.is_global_zero:
    #             print("[logits pretrain] Loaded cached logits_model; skipping warmup.")
    #         if self.trainer.strategy is not None:
    #             self.trainer.strategy.barrier()
    #         return

    #     # If rank>0, let rank 0 try to pretrain. Others wait, then load.
    #     if not self.trainer.is_global_zero:
    #         if self.trainer.strategy is not None:
    #             self.trainer.strategy.barrier()
    #         if self._load_pretrained_logits_if_available():
    #             return
    #         return

    #     # ---------- Rank 0: run the one-time warmup ----------
    #     epochs = 40  # number of warmup epochs
    #     print(f"[logits pretrain] Starting warmup for {epochs} epochs…")
    #     logits_optimizer = torch.optim.Adam(self.logits_model.parameters(), lr=self.lr)
    #     model_dev = next(self.logits_model.parameters()).device
    #     best_loss = float("inf")   
    #     best_state = None
    #     best_epoch = -1

    #     for epoch in tqdm(range(epochs)):
    #         epoch_losses = []
    #         for batch in tqdm(self.data_module.train_dataloader()):
    #             specs, lengths = (batch["specs"], batch["lengths_samples"])
    #             y = specs[:, 0:1]  # (B, 1, F, T)
    #             x = specs[:, 1:2]  # (B, 1, F, T)

    #             # sample a single t per item in the (audio) batch, then broadcast across S
    #             t_b = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
    #             t_b = t_b.to(model_dev)  # shape [B]

    #             # ----- get clean/noisy logits: shapes [B, S, |A|] -----
    #             _, clean_logits, clean_mask, clean_audio_embedding = self.extract_logits(x, lengths)   # [B, S, C]
    #             _, noisy_logits, noisy_mask, noisy_audio_embedding = self.extract_logits(y, lengths)   # [B, S, C]
    #             noisy_logits = dtw_align_batch(clean_logits, noisy_logits)[0]
    #             clean_logits = clean_logits.to(model_dev)
    #             noisy_logits = noisy_logits.to(model_dev)
    #             mask = clean_mask.to(model_dev)                 # [B, S] boolean
    #             mask_flat = mask.reshape(B * S)                 # [B*S]
    #             B, S, C = clean_logits.shape

    #             # ---- diffuse per-step (but do it in parallel) ----
    #             # Your SDE.marginal_prob expects an extra trailing dim; keep that trick.
    #             noise = torch.randn_like(clean_logits)                           # [B, S, C]
    #             clean_logits_4d = clean_logits[..., None]                        # [B, S, C, 1]
    #             noisy_logits_4d = noisy_logits[..., None]                        # [B, S, C, 1]
    #             # Broadcast t over S: [B] -> [B, S]
    #             t_bs = t_b[:, None].expand(B, S).contiguous()                    # [B, S]
    #             # Some SDE impls accept vector t; compute mean/std stepwise by flattening t as needed
    #             mean_l, std_l = self.sde.marginal_prob(clean_logits_4d, noisy_logits_4d, t_bs)  # returns [B,S,C,1] mean, [B,S] std
    #             mean_l = mean_l.squeeze(-1)                                      # [B, S, C]
    #             x_t_l = mean_l + std_l[:, :, None] * noise                       # [B, S, C]

    #             # ---- PER-STEP MODEL CALL: flatten S into batch ----
    #             x_t_flat     = x_t_l.reshape(B * S, C)        # [B*S, C]
    #             noisy_flat   = noisy_logits.reshape(B * S, C) # [B*S, C]
    #             t_flat       = t_bs.reshape(B * S)            # [B*S]

    #             logits_optimizer.zero_grad()

    #             # Your model API is assumed to be (x_t_step, cond_step, t_step) -> score_step with shape [B*S, C]
    #             predicted_score = self.logits_model(x_t_flat, noisy_flat, t_flat)  # [B*S, C]

    #             # ---- loss: same score-matching form, but per-step ----
    #             sigma_flat = self.sde._std(t_flat)[:, None].to(model_dev)          # [B*S, 1]
    #             resid_flat = predicted_score * sigma_flat + noise.reshape(B * S, C)
    #             resid_flat = resid_flat[mask_flat]  # mask out padded frames
    #             # mean over all B*S items; sum over channels
    #             loss = 0.5 * resid_flat.pow(2).sum(-1).mean()

    #             loss.backward()
    #             logits_optimizer.step()
    #             epoch_losses.append(loss.detach().cpu())

    #         avg_loss = torch.stack(epoch_losses).mean()
    #         self.logger.experiment.log({"logits_pretrain_loss": avg_loss.item()})

    #         if avg_loss.item() < best_loss:
    #             best_loss = avg_loss.item()
    #             best_epoch = int(epoch + 1)
    #             best_state = {
    #                 "state_dict": self.logits_model.state_dict(),
    #                 "meta": {
    #                     "best_epoch": int(epoch + 1),
    #                     "best_loss": float(best_loss),
    #                     "allowed_toks_len": int(len(self.allowed_toks)),
    #                 },
    #             }
    #             torch.save(best_state, self.logits_pretrain_ckpt)
    #             print(f"[logits pretrain] New best model saved at epoch {epoch+1} (loss={best_loss:.6f})")

    #     # best checkpoint was already saved inside the epoch loop
    #     print(f"[logits pretrain] Best epoch: {best_epoch} (loss={best_loss:.6f}); "
    #         f"checkpoint saved to {self.logits_pretrain_ckpt}")

    #     # let other ranks proceed and then load
    #     if self.trainer.strategy is not None:
    #         self.trainer.strategy.barrier()

    # def on_fit_start(self):
    #     # If already trained & cached → just load & return
    #     if self._load_pretrained_logits_if_available():
    #         if self.trainer.is_global_zero:
    #             print("[logits pretrain] Loaded cached logits_model; skipping warmup.")
    #         if self.trainer.strategy is not None:
    #             self.trainer.strategy.barrier()
    #         return

    #     # If rank>0, let rank 0 try to pretrain. Others wait, then load.
    #     if not self.trainer.is_global_zero:
    #         if self.trainer.strategy is not None:
    #             self.trainer.strategy.barrier()
    #         if self._load_pretrained_logits_if_available():
    #             return
    #         return

    #     # ---------- Rank 0: run the one-time warmup ----------
    #     epochs = 40  # number of warmup epochs
    #     print(f"[logits pretrain] Starting warmup for {epochs} epochs…")
    #     logits_optimizer = torch.optim.Adam(self.logits_model.parameters(), lr=1e-4)
    #     model_dev = next(self.logits_model.parameters()).device
    #     best_loss = float("inf")   
    #     best_state = None
    #     best_epoch = -1
    #     tok = self.multilingual_tokenizer
    #     allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=model_dev)

    #     for epoch in tqdm(range(epochs)):
    #         epoch_losses = []
    #         for batch in tqdm(self.data_module.train_dataloader()):
    #             specs, lengths = (batch["specs"], batch["lengths_samples"])
    #             y = specs[:, 0:1]  # (B, 1, F, T)
    #             x = specs[:, 1:2]  # (B, 1, F, T)

    #             # sample a single t per item in the (audio) batch, then broadcast across S
    #             t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
    #             t = t.to(model_dev)  # shape [B]

    #             # ----- get clean/noisy logits: shapes [B, S, |A|] -----
    #             _, clean_logits, clean_mask, clean_audio_embedding = self.extract_logits(x, lengths)   # [B, S1, C]
    #             _, noisy_logits, noisy_mask, noisy_audio_embedding = self.extract_logits(y, lengths)   # [B, S2, C]
    #             clean_logits = clean_logits.to(model_dev)
    #             noisy_logits = noisy_logits.to(model_dev)
    #             clean_logits, noisy_logits, mask = self.pad_align_logits_with_eot(clean_logits, clean_mask, noisy_logits, noisy_mask)  # [B, S, C], [B, S] boolean
    #             B, S, C = clean_logits.shape

    #             # ---- diffuse per-step (but do it in parallel) ----
    #             noise = torch.randn_like(clean_logits)                           # [B, S, C]
    #             # clean_logits_4d = clean_logits[..., None]                        # [B, S, C, 1]
    #             # noisy_logits_4d = noisy_logits[..., None]                        # [B, S, C, 1]
    #             mean_l, std_l = self.sde.marginal_prob(clean_logits, noisy_logits, t, is_logits=True)  # returns [B,S,C,1] mean, [B] std
    #             # mean_l = mean_l.squeeze(-1)                                      # [B, S, C]
    #             x_t_l = mean_l + std_l[:, None, None] * noise                    # [B, S, C]

    #             match = (allowed_ids_t == tok.sot).nonzero(as_tuple=True)[0]
    #             if match.numel() == 0:
    #                 raise ValueError("tok.sot is not in allowed_ids_t; cannot write 1-hot for padded rows.")
    #             sot_pos_in_allowed = int(match.item())
    #             one_hot = F.one_hot(torch.full((B,), sot_pos_in_allowed, device=model_dev), num_classes=C).to(noisy_logits.dtype)
    #             out_logits = one_hot.unsqueeze(1)                                 # [B,1,C]
    #             alpha_t = torch.exp(-self.sde.theta_l * t)[:, None].to(model_dev)   # [B, 1]
    #             sigma_l = self.sde._std(t, is_logits=True)[:, None].to(model_dev)   # [B, 1]
    #             std2    = sigma_l ** 2                                            # [B, 1]
    #             predicted_scores = []
    #             for s in range(S):
    #                 x_t_step = x_t_l[:, s, ...]           # [B, C] 
    #                 noisy_step = noisy_logits[:, s, ...]  # same shape as above
    #                 predicted_score = self.logits_model(x_t_step, noisy_step, t, noisy_audio_embedding, out_logits)  # [B, C]
    #                 # predicted_score = x_t_step
    #                 predicted_scores.append(predicted_score.unsqueeze(1))
    #                 x_clean_pred = (x_t_step - (1.0 - alpha_t) * noisy_step + std2 * predicted_score) / alpha_t  # [B, C]
    #                 out_logits = torch.cat([out_logits, x_clean_pred.unsqueeze(1)], dim=1)  # [B, cur_len+1, C]
                    
    #             # concatenate all steps along dimension 1 → [B, S, C]
    #             predicted_score_all = torch.cat(predicted_scores, dim=1)
    #             logits_optimizer.zero_grad()

    #             sigma_l = self.sde._std(t, is_logits=True)[:, None, None].to(model_dev) 
    #             resid = predicted_score_all * sigma_l + noise    # [B, S, C]
    #             sq   = resid.pow(2).sum(-1)             # [B,S]
    #             m    = mask.to(sq.dtype)                # float for arithmetic
    #             loss = 0.5 * (sq * m).sum() / m.sum().clamp_min(1)

    #             loss.backward()
    #             logits_optimizer.step()
    #             epoch_losses.append(loss.detach().cpu())

    #         avg_loss = torch.stack(epoch_losses).mean()
    #         self.logger.experiment.log({"logits_pretrain_loss": avg_loss.item()})

    #         if avg_loss.item() < best_loss:
    #             best_loss = avg_loss.item()
    #             best_epoch = int(epoch + 1)
    #             best_state = {
    #                 "state_dict": self.logits_model.state_dict(),
    #                 "meta": {
    #                     "best_epoch": int(epoch + 1),
    #                     "best_loss": float(best_loss),
    #                     "allowed_toks_len": int(len(self.allowed_toks)),
    #                 },
    #             }
    #             torch.save(best_state, self.logits_pretrain_ckpt)
    #             print(f"[logits pretrain] New best model saved at epoch {epoch+1} (loss={best_loss:.6f})")

    #     # best checkpoint was already saved inside the epoch loop
    #     print(f"[logits pretrain] Best epoch: {best_epoch} (loss={best_loss:.6f}); "
    #         f"checkpoint saved to {self.logits_pretrain_ckpt}")

    #     # let other ranks proceed and then load
    #     if self.trainer.strategy is not None:
    #         self.trainer.strategy.barrier()

    def _loss(self, forward_out_audio, forward_out_logits, t, x_t_audio, z_audio, mean_audio, x_audio, x_t_logits, z_logits, mean_logits, x_logits, masks_audio, masks_logits):
        """
        Different loss functions for training the score model
        """
        sigma_a = self.sde._std(t, is_logits=False)[:, None, None, None]
        sigma_l = self.sde._std(t, is_logits=True)[:, None, None]

        if self.loss_type == "score_matching":
            score_a = forward_out_audio
            score_l = forward_out_logits
            if self.loss_weighting == "sigma^2":
                losses_a = torch.square(torch.abs(score_a * sigma_a + z_audio))  # Eq. (7)
                losses_l = torch.square(torch.abs(score_l * sigma_l + z_logits))  # Eq. (7)
            else:
                raise ValueError(f"Invalid loss weighting for loss_type=score_matching: {self.loss_weighting}")
            m_t = masks_audio.to(losses_a.dtype).squeeze(1).squeeze(1)   # [B, T]
            # per-frame mean over F (and channel if present)
            per_frame = losses_a.sum(dim=(1, 2))                  # [B, T]
            # num = 0.5 * (per_frame * m_t).sum(dim=-1)              # [B]
            # den = m_t.sum(dim=-1).clamp_min(1.0)                   # [B]
            # loss_audio = (num / den).mean()                        # scalar
            loss_audio = 0.5 * (per_frame * m_t).sum(dim=-1).mean() 
            # Sum over spatial dimensions and channels and mean over batch
            # loss_audio = torch.mean(0.5*torch.sum(losses_a.reshape(losses_a.shape[0], -1), dim=-1))
            mask_f = masks_logits.to(losses_l.dtype)  # cast to float for arithmetic
            loss_per_step = losses_l.sum(dim=-1)  # [B, S]
            loss_logits = 0.5 * (loss_per_step * mask_f).sum(dim=-1).mean()  # scalar
            # masked_loss = loss_per_step * mask_f  # [B, S]
            # loss_logits = 0.5 * masked_loss.sum() / mask_f.sum().clamp_min(1)
            # loss_logits = 0.5 * losses_l.mean(dim=(0, 1, 2))
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        
        loss = loss_audio + self.logits_weight*loss_logits
        return loss

    def _step(self, batch, batch_idx):
        specs, lengths, masks_audio = batch["specs"], batch["lengths_samples"], batch["time_mask"]
        y = specs[:, 0:1]        # (B, 1, F, T)
        x = specs[:, 1:2]        # (B, 1, F, T)
        model_dev = x.device

        audio_mean = y.mean(dim=(1, 2, 3), keepdim=True).abs()  # [B,1,1,1]
        audio_std  = y.std(dim=(1, 2, 3), keepdim=True)   # [B,1,1,1]
        target_mean = audio_mean.squeeze(-1).squeeze(-1)  # [B,1]
        target_std  = audio_std.squeeze(-1).squeeze(-1)   # [B,1]

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t, is_logits=False)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z

        _, clean_logits, clean_mask, clean_audio_embedding = self.extract_logits(x, lengths, target_mean, target_std)
        clean_logits = clean_logits.to(model_dev)

        # Getting x0_hat
        _, y_logits, y_mask, y_audio_embedding = self.extract_logits(y, lengths, target_mean, target_std)
        score_t = self(x_t, y, t, is_logits=False, logits_cond=y_logits)                 
        alpha_t = torch.exp(-self.sde.theta_x * t).view(-1, 1, 1, 1)   # broadcast to (B,1,1,1)
        std2    = (std.view(-1, 1, 1, 1)) ** 2
        x0_hat  = (x_t - (1.0 - alpha_t) * y + std2 * score_t) / alpha_t
        if self.sampling_mode == "parallel":
            _, noisy_logits, noisy_mask, noisy_audio_embedding = self.extract_logits(x0_hat, lengths, target_mean, target_std)
            noisy_logits = noisy_logits.to(model_dev)
        elif self.sampling_mode == "full" or self.sampling_mode == "nested":
            noisy_logits, noisy_mask, noisy_audio_embedding = y_logits, y_mask, y_audio_embedding
            noisy_logits = noisy_logits.to(model_dev)

        # if self.sampling_mode == "parallel":
        #     _, noisy_logits, noisy_mask, noisy_audio_embedding = self.extract_logits(x0_hat, lengths, target_mean, target_std)
        #     noisy_logits = noisy_logits.to(model_dev)
        #     # 1) align clean & noisy
        #     clean_logits, clean_mask, noisy_logits, noisy_mask, _ = \
        #         self.pad_align_logits_with_eot(clean_logits, clean_mask, noisy_logits, noisy_mask)

        #     # 2) align noisy & y
        #     noisy_logits, noisy_mask, y_logits, y_mask, _ = \
        #         self.pad_align_logits_with_eot(noisy_logits, noisy_mask, y_logits, y_mask)

        #     # 3) align clean & y, reusing updated masks
        #     clean_logits, clean_mask, y_logits, y_mask, masks_logits = \
        #         self.pad_align_logits_with_eot(clean_logits, clean_mask, y_logits, y_mask)

        #     half_t = 0.2 * self.sde.T
        #     mask = (t < half_t).unsqueeze(-1).unsqueeze(-1)
        #     logits_cond_parallel = torch.where(mask, noisy_logits, y_logits)

        # elif self.sampling_mode == "full":
        #     noisy_logits, noisy_mask, noisy_audio_embedding = y_logits, y_mask, y_audio_embedding
        #     noisy_logits = noisy_logits.to(model_dev)

        #     clean_logits, clean_mask, noisy_logits, noisy_mask, masks_logits = \
        #         self.pad_align_logits_with_eot(clean_logits, clean_mask, noisy_logits, noisy_mask)

        # elif self.sampling_mode == "nested":
        #     _, x0_logits, x0_mask, x0_audio_embedding = self.extract_logits(x0_hat, lengths, target_mean, target_std)
        #     x0_logits = x0_logits.to(model_dev)

        #     # align x0 & y
        #     x0_logits, x0_mask, y_logits, y_mask, _ = \
        #         self.pad_align_logits_with_eot(x0_logits, x0_mask, y_logits, y_mask)

        #     # align clean & y
        #     clean_logits, clean_mask, y_logits, y_mask, _ = \
        #         self.pad_align_logits_with_eot(clean_logits, clean_mask, y_logits, y_mask)

        #     # align clean & x0
        #     clean_logits, clean_mask, x0_logits, x0_mask, masks_logits = \
        #         self.pad_align_logits_with_eot(clean_logits, clean_mask, x0_logits, x0_mask)

        #     half_t = 0.2 * self.sde.T
        #     mask = (t < half_t).unsqueeze(-1).unsqueeze(-1)
        #     logits_cond_nested = torch.where(mask, x0_logits, y_logits)

        #     noisy_logits, noisy_mask, noisy_audio_embedding = y_logits, y_mask, y_audio_embedding
        #     noisy_logits = noisy_logits.to(model_dev)

        # noisy_logits , _ , _ = dtw_align_batch(clean_logits, noisy_logits, metric="cosine", band_ratio=0.2)
        clean_logits, clean_mask, noisy_logits, noisy_mask, masks_logits = \
                self.pad_align_logits_with_eot(clean_logits, clean_mask, noisy_logits, noisy_mask)
        
        sigma_logits = torch.randn_like(clean_logits)
        # clean_logits_4d  = clean_logits[:, :, :, None]
        # noisy_logits_4d = noisy_logits[:, :, :, None]
        
        mean_l, std_l = self.sde.marginal_prob(clean_logits, noisy_logits, t, is_logits=True)  
        # mean_l = mean_l.squeeze(-1)
        x_t_l = mean_l + std_l[:, None, None] * sigma_logits

        half_t = 0.5 * self.sde.T                    # scalar
        mask    = (t < half_t).unsqueeze(-1).unsqueeze(-1)         # (B, 1, 1)  True ↔ late steps
        logits_cond = torch.where(mask, x_t_l, noisy_logits)
        # if self.sampling_mode == "parallel":
        #     forward_out_audio = self(x_t, y, t, is_logits=False, logits_cond=logits_cond_parallel)
        #     forward_out_logits = self(x_t_l, logits_cond_parallel, t, is_logits=True, noisy_audio_embedding=noisy_audio_embedding)
        # elif self.sampling_mode == "full":
        #     forward_out_audio = self(x_t, y, t, is_logits=False, logits_cond=noisy_logits)
        #     forward_out_logits = self(x_t_l, noisy_logits, t, is_logits=True, noisy_audio_embedding=noisy_audio_embedding)
        # elif self.sampling_mode == "nested":
        #     forward_out_audio = self(x_t, y, t, is_logits=False, logits_cond=logits_cond_nested)
        #     forward_out_logits = self(x_t_l, noisy_logits, t, is_logits=True, noisy_audio_embedding=noisy_audio_embedding)
        if self.sampling_mode == "parallel":
            forward_out_audio = self(x_t, y, t, is_logits=False, logits_cond=logits_cond)
            forward_out_logits = self(x_t_l, noisy_logits, t, is_logits=True, noisy_audio_embedding=noisy_audio_embedding)
        elif self.sampling_mode == "full":
            forward_out_audio = self(x_t, y, t, is_logits=False, logits_cond=noisy_logits)
            forward_out_logits = self(x_t_l, noisy_logits, t, is_logits=True, noisy_audio_embedding=noisy_audio_embedding)
        elif self.sampling_mode == "nested":
            forward_out_audio = self(x_t, y, t, is_logits=False, logits_cond=logits_cond)
            forward_out_logits = self(x_t_l, noisy_logits, t, is_logits=True, noisy_audio_embedding=noisy_audio_embedding)

        loss = self._loss(
            forward_out_audio,
            forward_out_logits,
            t,
            x_t_audio=x_t,
            z_audio=z,
            mean_audio=mean,
            x_audio=x,
            x_t_logits=x_t_l,
            z_logits=sigma_logits,
            mean_logits=mean_l,
            x_logits=clean_logits,
            masks_audio=masks_audio,
            masks_logits=masks_logits
        )
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
            o_list, o_list_clean, o_list_corrupted, o_list_logits = [], [], [], [] # enhanced/clean/noisy/logits
            debug_seen_per_label = defaultdict(int)

            # -------------------- decode helpers ------------------------
            device = next(self.whisper.parameters()).device
            tok = self.multilingual_tokenizer
            n_vocab = tok.encoding.n_vocab

            # self.allowed_toks may be a Python list; turn into a device tensor once
            allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=device)

            # # a mask we will add to logits each step: allowed=0, disallowed=-inf
            # neg_inf = -1e9
            # base_mask = torch.full((n_vocab,), neg_inf, device=device)
            # base_mask[allowed_ids_t] = 0.0

            # maximum tokens we’ll allow to emit (you can expose this as a hyperparam)
            max_decode_len = 224

            def decode_constrained(signal_16k, target_mean, target_std):
                """
                Greedy, closed-set decode: at each step, mask logits to allowed set.
                Returns (pred_text, mel) where pred_text is the decoded string.
                """

                # to device
                wav = torch.tensor(signal_16k, dtype=torch.float32, device=device)
                true_audio_length = wav.shape[-1]
                wav = self._pad_or_trim(wav)  # (T,)
                mel = self._log_mel_spectrogram(wav)  # (1, 80, Tm)
                feats = self.whisper.encoder(mel)  # (1, T', C)

                hop_length = 160
                encoder_downsampling_factor = 2
                true_mel_length = (true_audio_length + hop_length - 1) // hop_length
                true_encoder_length = (true_mel_length + encoder_downsampling_factor - 1) // encoder_downsampling_factor
                audio_embedding = feats[:, :true_encoder_length, :]

                # start with SOT + <|notimestamps|>
                prefix = torch.tensor([tok.sot_sequence_including_notimestamps],
                                    dtype=torch.long, device=device)  # (1, L0)
                out = prefix.clone()
                per_step_allowed = []  # each item: (1, |A|)
                constrained_tokens = []

                for _ in range(max_decode_len):
                    # logits over full vocab for each position so far
                    logits = self.whisper.decoder(out, feats).squeeze(0)  # (L, V)
                    next_logits = logits[-1]  # (V,)
                    # record only the allowed positions' logits for this step -> (|A|,)
                    # allowed_step = torch.softmax(next_logits.index_select(0, allowed_ids_t).unsqueeze(0),dim=-1)  # (1, |A|)
                    best_allowed_idx = int(torch.argmax(next_logits.index_select(0, allowed_ids_t)).item())
                    constrained_token_id = allowed_ids_t[best_allowed_idx].item()
                    constrained_tokens.append(constrained_token_id)

                    allowed_step = next_logits.index_select(0, allowed_ids_t).unsqueeze(0)  # (1, |A|)
                    allowed_step = normalize_logits(allowed_step, target_mean, target_std)
                    per_step_allowed.append(allowed_step)

                    # apply closed-set mask
                    # next_logits = next_logits + base_mask

                    # greedy pick
                    next_id = int(torch.argmax(next_logits).item())
                    out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)

                    if constrained_token_id == tok.eot:
                        break
                
                step_logits_allowed = torch.cat(per_step_allowed, dim=0)  # (S, |A|)
                # strip the SOT sequence and optional trailing EOT for text
                # gen = out[0].tolist()
                # start = len(tok.sot_sequence_including_notimestamps)
                # if len(gen) > 0 and gen[-1] == tok.eot:
                #     gen = gen[:-1]
                # text = tok.decode(gen[start:])
                if len(constrained_tokens) > 0 and constrained_tokens[-1] == tok.eot:
                    constrained_tokens = constrained_tokens[:-1]
                text = tok.decode(constrained_tokens)
                return text, mel, step_logits_allowed, audio_embedding
            
            def decode_from_allowed_logits(
                step_logits_allowed: torch.Tensor,  # [1, S, |A|] or [S, |A|]
            ):
                """
                Greedy decode from per-step logits over a restricted token set.

                step_logits_allowed: (1, S, |A|) or (S, |A|)
                    - S = number of decoding steps
                    - |A| = size of allowed token set
                allowed_ids_t: (|A|,)
                    - mapping from restricted indices -> real Whisper token IDs
                tok: Whisper tokenizer (same one you used before)
                """
                # Drop batch dim if present
                if step_logits_allowed.dim() == 3:
                    # [1, S, |A|] -> [S, |A|]
                    step_logits_allowed = step_logits_allowed[0]
                best_idx_per_step = step_logits_allowed.argmax(dim=-1)        # [S]
                token_list = allowed_ids_t[best_idx_per_step].tolist()    # [S]

                if tok.eot in token_list:
                    token_list = token_list[:token_list.index(tok.eot)]

                text = tok.decode(token_list)
                return text
            
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
            
            for i, (clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)):
                # Load the clean and noisy speech
                x, sr_x = load(clean_file)
                x = x.squeeze().numpy()
                y, sr_y = load(noisy_file)
                y = y.squeeze().numpy()
                assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
                signal = y_tensor
                norm_factor = signal.abs().max().item()
                signal = signal / norm_factor
                Y = torch.unsqueeze(self._forward_transform(self._stft(signal.cuda())), 0).unsqueeze(0)  # (1, 1, F, T)
                Y = pad_spec(Y)

                audio_mean = Y.mean(dim=(1, 2, 3), keepdim=True).abs()  # [B,1,1,1]
                audio_std  = Y.std(dim=(1, 2, 3), keepdim=True)   # [B,1,1,1]
                target_mean = audio_mean.squeeze(-1).squeeze(-1)  # [B,1]
                target_std  = audio_std.squeeze(-1).squeeze(-1)   # [B,1]

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
                # ground_truth = self.transcripts[category]

                # closed-set ASR on noisy and clean
                corrupted_transcript, corrupted_mel, corrupted_logits, corrupted_audio_embedding = decode_constrained(y_16k, target_mean, target_std)
                clean_transcript, clean_mel, clean_logits, clean_audio_embedding = decode_constrained(x_16k, target_mean, target_std)
                # corrupted_transcript_unconstrained, corrupted_mel = decode_unconstrained(y_16k)
                # clean_transcript_unconstrained, clean_mel = decode_unconstrained(x_16k)

                # Enhance the noisy speech (keep native sr for enhancement)
                x_hat, logits_hat = self.enhance(y_tensor.unsqueeze(0), corrupted_logits.unsqueeze(0), corrupted_audio_embedding, N=self.sde.N, snr=0.5)
                logits_text = decode_from_allowed_logits(logits_hat.detach())

                # Resample enhanced to 16k for Whisper + metrics that expect 16k
                if sr_y != 16000:
                    x_hat_16k = resample(x_hat, orig_sr=sr_y, target_sr=16000).squeeze()
                else:
                    x_hat_16k = x_hat.detach().cpu().numpy() if torch.is_tensor(x_hat) else x_hat

                # closed-set ASR on enhanced
                enhanced_transcript, enhanced_mel, enhanced_logits, enhanced_audio_embedding = decode_constrained(x_hat_16k, target_mean, target_std)
                # enhanced_transcript_unconstrained, enhanced_mel = decode_unconstrained(x_hat_16k)
                enhanced_transcript_processed = expand_contractions(self.text_normalizer(enhanced_transcript))
                # ground_truth_processed = expand_contractions(self.text_normalizer(ground_truth))
                corrupted_transcript_processed = expand_contractions(self.text_normalizer(corrupted_transcript))
                clean_transcript_processed = expand_contractions(self.text_normalizer(clean_transcript))
                logits_transcript_processed = expand_contractions(self.text_normalizer(logits_text))

                # --- collect WER strings ---
                o_list.append(enhanced_transcript_processed)
                o_list_corrupted.append(corrupted_transcript_processed)
                o_list_clean.append(clean_transcript_processed)
                o_list_logits.append(logits_transcript_processed)
                # l_list.append(ground_truth_processed)

                # wer_clean_    = self.wer_metric(clean_transcript_processed, ground_truth_processed)
                # wer_enhanced_  = self.wer_metric(enhanced_transcript_processed, clean_transcript_processed)
                # wer_corrupted_ = self.wer_metric(corrupted_transcript_processed, clean_transcript_processed)

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
            # wer_clean     = self.wer_metric(o_list_clean, l_list)
            wer_enhanced  = self.wer_metric(o_list, o_list_clean)
            wer_corrupted = self.wer_metric(o_list_corrupted, o_list_clean)
            wer_logits    = self.wer_metric(o_list_logits, o_list_clean)

            self.log('pesq',  (pesq_sum / pesq_cnt) if pesq_cnt > 0 else float('nan'), on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi',  estoi_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_enhanced',  wer_enhanced,  on_step=False, on_epoch=True, sync_dist=True)
            # self.log('wer_clean',     wer_clean,     on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_corrupted', wer_corrupted, on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_logits', wer_logits, on_step=False, on_epoch=True, sync_dist=True)

        # keep your training loss path unchanged
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

        #     for i, (clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)):
        #         # Load the clean and noisy speech
        #         x, sr_x = load(clean_file)
        #         x = x.squeeze().numpy()
        #         y, sr_y = load(noisy_file) 
        #         assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

        #         # Resample if necessary
        #         if sr_x != 16000:
        #             x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
        #             y_16k = resample(y, orig_sr=sr_y, target_sr=16000).squeeze()
        #         else:
        #             x_16k = x
        #             y_16k = y
                
        #         label_str = os.path.splitext(os.path.basename(clean_file))[0].split('_')[0]
        #         tokens = [*self.multilingual_tokenizer.sot_sequence_including_notimestamps] \
        #                 + self.multilingual_tokenizer.encode(" " + label_str)
        #         labels = tokens[1:] + [self.multilingual_tokenizer.eot]
        #         multilingual_tokens = torch.tensor(tokens, dtype=torch.long)
        #         labels = torch.tensor(labels, dtype=torch.long)

        #         multilingual_tokens = multilingual_tokens.to(self.whisper.device)
        #         labels = labels.to(self.whisper.device)

        #         # ---- locate keyword position once per utterance ----------
        #         keyword_idx = next(j for j, t in enumerate(labels.tolist()) if t not in special_ids)

        #         def run_whisper(signal_16k, multilingual_tokens, labels, keyword_idx):
        #             # 1) move the waveform to the same device Whisper lives on
        #             device = next(self.whisper.parameters()).device
        #             signal  = torch.tensor(signal_16k, dtype=torch.float32, device=device)

        #             # 2) Whisper’s utility helpers
        #             # signal_padded_ref = whisper.pad_or_trim(signal).flatten()      # (T,)
        #             # mel_ref           = whisper.log_mel_spectrogram(signal_padded_ref).to(device)  # (80, T)
        #             signal_padded = self._pad_or_trim(signal)                  
        #             mel = self._log_mel_spectrogram(signal_padded)

        #             # 3) Encoder‑decoder forward
        #             # feats  = self.whisper.encoder(mel.unsqueeze(0))            # (1, …, T’)
        #             feats  = self.whisper.encoder(mel)
        #             tokens = multilingual_tokens.to(device).unsqueeze(0)       # (1, L)
        #             logits = self.whisper.decoder(tokens, feats).squeeze(0)    # (L, vocab)

        #             # 4) Cross‑entropy w.r.t. the ground‑truth label sequence
        #             ce = self.whisper_loss(logits.view(-1, logits.size(-1)),
        #                             labels.to(device).view(-1)).detach().cpu()

        #             # 5) Closed‑set keyword probabilities
        #             kw_logits = logits[keyword_idx]            # vector over entire vocab
        #             kw_probs  = kw_logits                      # turn into probabilities
        #             kw_probs  = kw_probs[self.allowed_toks.to(device)].softmax(-1)   # length‑10

        #             pred_id = kw_probs.argmax().item()
        #             return ce, pred_id, kw_probs, mel

                
        #         ce_loss_corrupted, pred_id_cor, kw_probs_cor, corrupted_mel = run_whisper(y_16k, multilingual_tokens, labels, keyword_idx)
        #         ce_loss_clean, pred_id_cln, kw_probs_cln, clean_mel = run_whisper(x_16k, multilingual_tokens, labels, keyword_idx)

        #         # Enhance the noisy speech
        #         device = next(self.whisper.parameters()).device
        #         y_tensor = torch.tensor(y_16k, dtype=torch.float32, device=device)
        #         x_hat = self.enhance(y_tensor, kw_probs_cor, N=self.sde.N, snr=0.5) # added the kw_probs_cor for conditioning

        #         if self.sr != 16000:
        #             x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
        #         else:
        #             x_hat_16k = x_hat    

        #         # x_hat_tensor = torch.tensor(x_hat_16k, dtype=torch.float32, device=device)
        #         # x_hat_whisper = self._level_normalize(x_hat_tensor, target_dbfs=-20.0)
        #         ce_loss_enhanced, pred_id_enh, kw_probs_enh, enhanced_mel = run_whisper(x_hat_16k, multilingual_tokens, labels, keyword_idx)
        #         true_id = (self.allowed_toks == labels[keyword_idx]).nonzero(as_tuple=True)[0].item()
        #         ground_truth = self.allowed_words[true_id]
        #         # --- closed-set accuracy ----------------------------------
        #         total_cnt   += 1
        #         correct_cnt_enhanced += int(pred_id_enh == true_id)

        #         # --- WER lists (as close to original as possible) ---------
        #         enhanced_transcript  = self.allowed_words[pred_id_enh]
        #         corrupted_transcript = self.allowed_words[pred_id_cor]
        #         clean_transcript     = self.allowed_words[pred_id_cln]

        #         o_list.append(enhanced_transcript)
        #         o_list_corrupted.append(corrupted_transcript)
        #         o_list_clean.append(clean_transcript)
        #         l_list.append(ground_truth)
        
        #         # pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb') 
        #         try:
        #             pesq_sum  += pesq(16000, x_16k, x_hat_16k, 'wb')
        #             pesq_cnt+=1
        #         except Exception:
        #             pass
        #         # si_sdr_sum += si_sdr(x, x_hat)
        #         si_sdr_sum += si_sdr(torch.tensor(x), torch.tensor(x_hat))
        #         estoi_sum += stoi(x, x_hat, self.sr, extended=True)

        #         stem = os.path.splitext(os.path.basename(clean_file))[0]
        #         pattern = rf'^{re.escape(label_str)}_0(?!\d)'
        #         if self.debug and re.match(pattern, stem):
        #             debug_seen_per_label[label_str] += 1
        #             self._plot_and_save_debug_info(
        #                 clean_mel=clean_mel,
        #                 corrupted_mel=corrupted_mel,
        #                 enhanced_mel=enhanced_mel,
        #                 clean_transcript=clean_transcript,
        #                 corrupted_transcript=corrupted_transcript,
        #                 enhanced_transcript=enhanced_transcript,
        #                 ground_truth=ground_truth,
        #                 index=debug_seen_per_label[label_str]
        #             )

        #     si_sdr_avg = si_sdr_sum / len(clean_files)
        #     estoi_avg = estoi_sum / len(clean_files)
        #     # --------------- metrics (names kept) -------------------------
        #     wer_clean     = self.wer_metric(o_list_clean, l_list)
        #     wer_enhanced  = self.wer_metric(o_list, l_list)
        #     wer_corrupted = self.wer_metric(o_list_corrupted, l_list)
        #     keyword_acc = torch.tensor(correct_cnt_enhanced / max(total_cnt, 1), device=self.device)

        #     self.log('pesq',pesq_sum / pesq_cnt if pesq_cnt>0 else float('nan'), on_step=False, on_epoch=True, sync_dist=True)
        #     self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log('estoi', estoi_avg, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log('wer_enhanced',  wer_enhanced,  on_step=False, on_epoch=True, sync_dist=True)
        #     self.log('wer_clean',     wer_clean,     on_step=False, on_epoch=True, sync_dist=True)
        #     self.log('wer_corrupted', wer_corrupted, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log('keyword_acc',  keyword_acc, on_step=False, on_epoch=True, sync_dist=True)

        # loss = self._step(batch, batch_idx)
        # self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # return loss
    
    def test_step(self, batch, batch_idx):
        # If your validation_step returns logs/metrics, just reuse it
        return self.validation_step(batch, batch_idx)


    def forward(self, x_t, y, t, is_logits=False, logits_cond=None, noisy_audio_embedding=None):
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
        if is_logits:
            model_dev = next(self.logits_model.parameters()).device
            tok = self.multilingual_tokenizer
            allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=model_dev)
            match = (allowed_ids_t == tok.sot).nonzero(as_tuple=True)[0]
            if match.numel() == 0:
                raise ValueError("tok.sot is not in allowed_ids_t; cannot write 1-hot for padded rows.")
            sot_pos_in_allowed = int(match.item())
            one_hot = torch.nn.functional.one_hot(torch.full((x_t.shape[0],), sot_pos_in_allowed, device=model_dev), num_classes=x_t.shape[-1]).to(x_t.dtype)
            out_logits = one_hot.unsqueeze(1)                                 # [B,1,C]
            alpha_t = torch.exp(-self.sde.theta_l * t).view(-1, *([1] * (x_t.ndim - 1)))   # [B,1,(...)]
            # std2    = (self.sde._std(t, is_logits=True).view_as(alpha_t)) ** 2                            # [B,1,(...)]
            std = self.sde._std(t, is_logits=True)
            predicted_scores = []
            for s in range(x_t.shape[1]):
                x_t_step = x_t[:, s, ...]             # [B, C] 
                noisy_step = y[:, s, ...]  # same shape as above
                # predicted_score = self.logits_model(x_t_step, noisy_step, t, noisy_audio_embedding, out_logits)  # [B, C]
                # predicted_score = self.logits_model(x_t_step, noisy_step, std, noisy_audio_embedding, out_logits)  # [B, C]
                predicted_score = self.logits_model(x_t_step, noisy_step, std, noisy_audio_embedding, out_logits)  # [B, C]
                # if we return epsilon:
                # predicted_score = -predicted_noise/std.unsqueeze(1)  # [B, C]
                # predicted_score = x_t_step
                predicted_scores.append(predicted_score.unsqueeze(1))
                # x_clean_pred = (x_t_step - (1.0 - alpha_t.squeeze(1)) * noisy_step + std2.squeeze(1) * predicted_score) / alpha_t.squeeze(1)
                # out_logits = torch.cat([out_logits, x_clean_pred.unsqueeze(1)], dim=1)  # [B, cur_len+1, C]
                out_logits = torch.cat([out_logits, noisy_step.unsqueeze(1)], dim=1)  # [B, cur_len+1, C]
                
            # concatenate all steps along dimension 1 → [B, S, C]
            predicted_score_all = torch.cat(predicted_scores, dim=1)
            return predicted_score_all
        else:
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
                if self.backbone == "ncsnpp_48k_logits_conditioned" or self.backbone == "ncsnpp_logits_conditioned":
                    dnn_input = torch.cat([x_t, y], dim=1)     
                    score = -self.dnn(dnn_input, t, class_logits=logits_cond)
                    return score
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

    def get_pc_sampler(self, predictor_name, corrector_name, y, logits_cond, audio_embedding, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, "to_audio_fn": self.to_audio, "pad_or_trim_fn": self._pad_or_trim,
                  "log_mel_spectogram_fn":self._log_mel_spectrogram, "sampling_mode": self.sampling_mode,
                  "num_iterations":self.num_iterations, "logits_diffusion_steps":self.logits_diffusion_steps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, logits_cond=logits_cond, audio_embedding_cond=audio_embedding, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns, logit_samples = [], [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n, logit_sample = sampler()
                    samples.append(sample)
                    logit_samples.append(logit_sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                logit_samples = torch.cat(logit_samples, dim=0)
                return samples, ns, logit_samples
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
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.size(0)
        
        # Process each item in the batch
        log_mels = []
        
        for i in range(batch_size):
            audio_single = audio[i]  # (T,)
            
            # dtype guard
            if audio_single.dtype != torch.float32:
                audio_single = audio_single.float()

            window = torch.hann_window(n_fft, device=audio_single.device, dtype=audio_single.dtype)

            # STFT → power
            stft = torch.stft(
                audio_single,
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
                sample_rate=sr,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                norm="slaney",
                mel_scale="slaney",
            ).to(power.device) 

            if fb.shape[0] != n_mels:      # old torchaudio → transpose
                fb = fb.t().contiguous()

            mel = fb @ power
            mel = mel[:, :-1]
            # log_mel = torch.log(torch.clamp(mel, min=1e-10))
            log_mel = torch.log10(torch.clamp(mel, min=1e-10))
            log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
            log_mel = (log_mel + 4.0) / 4.0

            ref_log_mel = whisper.log_mel_spectrogram(audio_single)
            
            log_mels.append(log_mel)
        
        # Stack all mel spectrograms
        result = torch.stack(log_mels, dim=0)  # (B, n_mels, time_frames)
        
        return result
    
    def _level_normalize(self, wav: torch.Tensor, target_dbfs: float = -20.0, eps: float = 1e-8):
        # wav: 1D float32 tensor in [-?, ?] at 16 kHz
        wav = wav - wav.mean()                                      # DC removal
        peak = wav.abs().max()
        if peak > 1:                                                # guard against >1.0 from enhancement
            wav = wav / (peak + eps)
        rms = wav.pow(2).mean().sqrt().clamp_min(eps)
        rms_db = 20.0 * torch.log10(rms)
        gain = 10.0 ** ((target_dbfs - rms_db) / 20.0)
        # cap gain so peaks stay below full scale
        safe_gain = torch.minimum(gain, 0.99 / (wav.abs().max() + eps))
        return wav * safe_gain

    def enhance(self, y, logits_cond, audio_embedding, sampler_type="pc", predictor="reverse_diffusion",
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
        allowed_ids_t = torch.as_tensor(self.allowed_toks, dtype=torch.long, device=logits_cond.device)

        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == 'OUVESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), logits_cond, audio_embedding, N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False, T_orig=T_orig,
                    allowed_ids_t=allowed_ids_t, sr=self.sr, **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y.cuda(), sampler_type=self.sde.sampler_type)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(self.sde.__class__.__name__))

        sample, nfe, logit_sample = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf, logit_sample
        else:
            return x_hat, logit_sample
    
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
        TIME_FRAMES = 200  # Number of time frames to display
        # if mels of dim 3 squeeze them
        if clean_mel.dim() == 3:
            clean_mel = clean_mel.squeeze(0)
            corrupted_mel = corrupted_mel.squeeze(0)
            enhanced_mel = enhanced_mel.squeeze(0)
        
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
