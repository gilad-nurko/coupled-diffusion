#!/usr/bin/env python3

import os
import json
import math
import argparse
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# lightning extras
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# project deps
from sgmse.util.other import set_torch_cuda_arch_list
from sgmse.data_module_ears_reverb import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.backbones.residual_logits_model import LogitsDenoiser

# whisper + audio
import whisper
import torchaudio
from torchaudio.functional import resample as ta_resample

# ---------------------------------------------------------------
# CUDA + matmul mode
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision("high")
# ---------------------------------------------------------------

# --------------------- helpers (unchanged) ---------------------

def pad_or_trim(x: torch.Tensor, length: int = 30 * 16_000) -> torch.Tensor:
    if x.size(-1) < length:
        x = F.pad(x, (0, length - x.size(-1)))
    else:
        x = x[..., :length]
    return x


@torch.no_grad()
def log_mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 400,
    hop: int = 160,
    n_mels: int = 80,
    sr: int = 16_000,
    f_min: float = 0.0,
    f_max: float = 8_000.0,
) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    batch_size = audio.size(0)
    out = []

    hann_cache = {}
    def _hann(n_fft, device, dtype):
        key = (n_fft, device.type, device.index, str(dtype))
        if key not in hann_cache:
            hann_cache[key] = torch.hann_window(n_fft, device=device, dtype=dtype)
        return hann_cache[key]

    fbanks_cache = {}
    def _mel_fb(n_fft, n_mels, sr, f_min, f_max, device, dtype):
        key = (n_fft, n_mels, sr, f_min, f_max, device.type, device.index)
        if key not in fbanks_cache:
            fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1, sample_rate=sr, n_mels=n_mels,
                f_min=f_min, f_max=f_max, norm="slaney", mel_scale="slaney"
            ).to(device=device, dtype=dtype)
            if fb.shape[0] != n_mels:
                fb = fb.t().contiguous()
            fbanks_cache[key] = fb
        return fbanks_cache[key]

    for i in range(batch_size):
        a = audio[i]
        if a.dtype != torch.float32:
            a = a.float()

        window = _hann(n_fft, a.device, a.dtype)
        stft = torch.stft(
            a, n_fft, hop, window=window, win_length=n_fft,
            center=True, pad_mode="reflect", return_complex=True
        )
        power = stft.abs().pow(2.0)
        fb = _mel_fb(n_fft, n_mels, sr, f_min, f_max, a.device, a.dtype)
        mel = fb @ power
        mel = mel[:, :-1]

        log_mel = torch.log10(torch.clamp(mel, min=1e-10))
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        out.append(log_mel)

    return torch.stack(out, dim=0)


def build_allowed_token_id_set_with_tok(transcripts_path, tok):
    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    allowed = set()
    allowed.update(tok.sot_sequence_including_notimestamps)
    allowed.add(tok.eot)

    for s in [" ", "  ", "\n", "\n\n", "\t"]:
        allowed.update(tok.encode(s))

    for punct in [".", ",", "!", "?", "'", '"', ":", ";", "-", "—", "…", "(", ")", "[", "]"]:
        for p in (punct, " " + punct):
            allowed.update(tok.encode(p))

    for _, text in transcripts.items():
        allowed.update(tok.encode(text))
        allowed.update(tok.encode(" " + text))

    corpus = " " + " ".join(transcripts.values())
    allowed.update(tok.encode(corpus))
    corpus_nospace = "".join(transcripts.values())
    allowed.update(tok.encode(corpus_nospace))

    return sorted(allowed)

def visualize_logits_distribution(logits: torch.Tensor, audio: torch.Tensor, save_dir: str = "logits_visualizations_from_pretrain"):
        """
        Compute and visualize the mean logits distribution and histogram.

        Args:
            logits (torch.Tensor): Tensor of shape [B, S, |A|]
            index (str): Index or identifier to include in the filenames
            save_dir (str): Directory to save the output plots
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- compute mean logits over batch and sequence ---
        logits_one_step = logits.unsqueeze(1)  # shape [B, 1, |A|]
        mean_logits = logits_one_step.mean(dim=(0, 1))           # shape [|A|]
        mean_audio = audio.mean(dim=(0, 1, 2))
        # --- plot probability distribution across tokens ---
        plt.figure(figsize=(12, 5))
        plt.plot(mean_logits.cpu().numpy(), linewidth=0.8)
        plt.title(f"Mean Logits → Probability Distribution")
        plt.xlabel("Token index")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.tight_layout()
        dist_path = os.path.join(save_dir, f"mean_logits_distribution.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()

        # --- plot histogram of mean logit values ---
        plt.figure(figsize=(8, 5))
        plt.hist(mean_logits.cpu().numpy(), bins=100, color='steelblue', alpha=0.75)
        plt.title(f"Histogram of Mean Logit Values")
        plt.xlabel("Logit value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        hist_path = os.path.join(save_dir, f"mean_logits_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()

        # (a) Mean audio vs time
        x = torch.arange(mean_audio.shape[-1], dtype=torch.float32)
        x = x / float(48000)
        x_label = "Time (s)"

        plt.figure(figsize=(12, 4))
        plt.plot(x.cpu().numpy(), mean_audio.detach().cpu().numpy(), linewidth=0.8)
        plt.title(f"Mean Audio Time Series")
        plt.xlabel(x_label)
        plt.ylabel("Amplitude" if audio.dim() <= 3 else "Avg value across F")
        plt.grid(True)
        plt.tight_layout()
        audio_ts_path = os.path.join(save_dir, f"mean_audio_timeseries.png")
        plt.savefig(audio_ts_path, dpi=300)
        plt.close()

        # (b) Histogram of audio sample values
        plt.figure(figsize=(8, 5))
        plt.hist(mean_audio.detach().cpu().numpy(), bins=100, alpha=0.75)
        plt.title(f"Histogram of Mean Audio Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        audio_hist_path = os.path.join(save_dir, f"mean_audio_histogram.png")
        plt.savefig(audio_hist_path, dpi=300)
        plt.close()

@torch.no_grad()
def extract_logits_batch(
    whisper_model,
    tokenizer,
    specs: torch.Tensor,
    lengths_samples: torch.Tensor,
    to_audio_fn,
    sr_in: int,
    device_whisper,
    allowed_token_ids: torch.LongTensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    max_decode_len: int = 224,
):
    tok = tokenizer
    n_vocab = tok.encoding.n_vocab

    neg_inf = -1e9
    base_mask = torch.full((n_vocab,), neg_inf, device=device_whisper)
    base_mask[allowed_token_ids] = 0.0

    B = specs.shape[0]
    wav16_list, true_L_16k = [], []
    for i in range(B):
        spec_single = specs[i].squeeze(0)
        audio_single = to_audio_fn(spec_single, int(lengths_samples[i].item())).to(dtype=torch.float32)
        if sr_in != 16000:
            audio_16 = ta_resample(audio_single.unsqueeze(0), sr_in, 16000).squeeze(0)
        else:
            audio_16 = audio_single
        audio_16 = audio_16.to(device_whisper)
        true_L_16k.append(int(audio_16.shape[-1]))
        wav16_list.append(pad_or_trim(audio_16))
    wav16 = torch.stack(wav16_list, dim=0)

    mel = log_mel_spectrogram(wav16)                      # [B,80,Tm]
    feats = whisper_model.encoder(mel)                    # [B,Tenc,D]

    hop = 160
    pad_target_len = wav16.shape[-1]
    true_L_16k = [min(L, pad_target_len) for L in true_L_16k]
    true_mel_L = [(L + hop - 1) // hop for L in true_L_16k]
    enc_down = 4
    true_enc_L = [(m + enc_down - 1) // enc_down for m in true_mel_L]
    max_true_enc = max(true_enc_L)
    audio_embedding = feats[:, :max_true_enc, :]

    prefix = torch.tensor([tok.sot_sequence_including_notimestamps], dtype=torch.long, device=device_whisper)
    out = prefix.repeat(B, 1)

    active = torch.ones(B, dtype=torch.bool, device=device_whisper)
    step_logits = []
    first_eot_step = torch.full((B,), -1, dtype=torch.long, device=device_whisper)

    match = (allowed_token_ids == tok.eot).nonzero(as_tuple=True)[0]
    if match.numel() == 0:
        raise ValueError("tok.eot not in allowed_token_ids.")
    eot_pos_in_allowed = int(match.item())

    for s in range(max_decode_len):
        logits_full = whisper_model.decoder(out, feats)       # (B, cur_len, V)
        next_logits_full = logits_full[:, -1, :]              # (B, V)
        allowed_next = next_logits_full[:, allowed_token_ids]  # (B, |A|)
        allowed_next = normalize_logits(allowed_next, target_mean, target_std)

        if not active.all():
            one_hot_finished = allowed_next.new_zeros((allowed_next.size(0), allowed_next.size(1)))
            one_hot_finished[~active, eot_pos_in_allowed] = 1.0
            allowed_next = torch.where(active[:, None], allowed_next, one_hot_finished)

        step_logits.append(allowed_next)

        masked = next_logits_full + base_mask
        if not active.all():
            masked = masked.clone()
            masked[~active] = neg_inf
            masked[~active, tok.eot] = 1e9

        next_ids = torch.argmax(masked, dim=1)
        out = torch.cat([out, next_ids.unsqueeze(1)], dim=1)

        newly_finished = (next_ids == tok.eot) & active
        if newly_finished.any():
            first_eot_step[newly_finished] = s
        active = active & (~newly_finished)

        if not active.any():
            break

    allowed_logits = torch.stack(step_logits, dim=0).permute(1, 0, 2).contiguous()
    S = allowed_logits.shape[1]
    valid_lengths = torch.where(
        first_eot_step >= 0,
        first_eot_step + 1,
        torch.as_tensor(S, device=device_whisper)
    )
    ar = torch.arange(S, device=device_whisper).unsqueeze(0)
    mask = ar < valid_lengths.unsqueeze(1)
    return mel, allowed_logits, mask, audio_embedding


def pad_align_logits_with_eot(
    clean_logits: torch.Tensor,
    clean_mask:   torch.Tensor,
    noisy_logits: torch.Tensor,
    noisy_mask:   torch.Tensor,
    allowed_token_ids: torch.LongTensor,
    eot_id: int
):
    device = clean_logits.device
    dtype  = clean_logits.dtype
    B, S1, C = clean_logits.shape
    _, S2, C2 = noisy_logits.shape
    assert C == C2, f"Channel mismatch: {C} != {C2}"

    match = (allowed_token_ids == eot_id).nonzero(as_tuple=True)[0]
    if match.numel() == 0:
        raise ValueError("EOT not in allowed_token_ids.")
    eot_pos_in_allowed = int(match.item())

    eot_onehot = torch.zeros(C, device=device, dtype=dtype)
    eot_onehot[eot_pos_in_allowed] = 1.0

    S_max = max(S1, S2)
    if S_max == S1 == S2:
        return clean_logits, noisy_logits, (clean_mask | noisy_mask)

    if S1 < S_max:
        pad_steps = S_max - S1
        pad_logits_clean = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
        clean_logits = torch.cat([clean_logits, pad_logits_clean], dim=1)
        clean_mask   = torch.cat([clean_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)], dim=1)
    if S2 < S_max:
        pad_steps = S_max - S2
        pad_logits_noisy = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
        noisy_logits = torch.cat([noisy_logits, pad_logits_noisy], dim=1)
        noisy_mask   = torch.cat([noisy_mask, torch.zeros(B, pad_steps, dtype=torch.bool, device=device)], dim=1)

    union_mask = clean_mask | noisy_mask
    return clean_logits, noisy_logits, union_mask

def normalize_logits(logits: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    logits_mean = logits.mean(dim=1, keepdim=True)  # [B,1]
    logits_std  = logits.std(dim=1, keepdim=True)   # [B,1]

    # avoid divide-by-zero
    eps = 1e-8
    normalized_logits = (logits - logits_mean) / (logits_std + eps)

    logits_rescaled = normalized_logits * target_std + target_mean  # [B,|A|]
    return logits_rescaled

# ------------------- LightningModule -------------------

class LogitsPretrainModule(pl.LightningModule):
    def __init__(self, *,
                 sde_name: str,
                 sde_kwargs: dict,
                 logits_size: int,
                 sr: int,
                 t_eps: float,
                 whisper_name: str,
                 whisper_lang: str,
                 transcripts_path: str,
                 initial_lr: float = 1e-3,
                 final_lr: float = 1e-4,
                 total_steps_for_sched: int = 1000):
        super().__init__()
        self.save_hyperparameters()

        # core model
        self.logits_model = LogitsDenoiser(logits_size=logits_size)
        self.sde = SDERegistry.get_by_name(sde_name)(**sde_kwargs)

        # placeholders populated in setup()
        self.whisper_model = None
        self.tokenizer = None
        self.allowed_ids_t = None

    def setup(self, stage: str):
        # load whisper/tokenizer on correct device per-rank
        if self.whisper_model is None:
            options = whisper.DecodingOptions(language=self.hparams.whisper_lang, without_timestamps=True)
            self.whisper_model = whisper.load_model(self.hparams.whisper_name).to(self.device)
            for p in self.whisper_model.parameters():
                p.requires_grad = False
            self.whisper_model.eval()
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                True, language=self.hparams.whisper_lang, task=options.task
            )
            tp_effective = self.hparams.transcripts_path
            allowed_toks = build_allowed_token_id_set_with_tok(tp_effective, self.tokenizer)
            self.allowed_ids_t = torch.as_tensor(allowed_toks, dtype=torch.long, device=self.device)
    
    def on_train_start(self):
        if self.whisper_model is not None:
            self.whisper_model.eval()
    
    def _loss_from_batch(self, batch):
        specs, lengths = batch["specs"], batch["lengths_samples"]
        y = specs[:, 0:1]  # noisy [B,1,F,T]
        x = specs[:, 1:2]  # clean [B,1,F,T]

        # match your train math exactly
        audio_mean = y.mean(dim=(1, 2, 3), keepdim=True).abs()
        audio_std  = y.std(dim=(1, 2, 3), keepdim=True)
        target_mean = audio_mean.squeeze(-1).squeeze(-1)  # [B,1]
        target_std  = audio_std.squeeze(-1).squeeze(-1)   # [B,1]

        B = x.shape[0]
        t = torch.rand(B, device=self.device) * (self.sde.T - self.hparams.t_eps) + self.hparams.t_eps

        with torch.no_grad():
            _, clean_logits, clean_mask, clean_audio_embedding = extract_logits_batch(
                self.whisper_model, self.tokenizer, x, lengths, self.trainer.datamodule.istft,
                self.hparams.sr, self.device, self.allowed_ids_t, target_mean, target_std
            )
            _, noisy_logits,  noisy_mask,  noisy_audio_embedding  = extract_logits_batch(
                self.whisper_model, self.tokenizer, y, lengths, self.trainer.datamodule.istft,
                self.hparams.sr, self.device, self.allowed_ids_t, target_mean, target_std
            )

        clean_logits = clean_logits.to(self.device)
        noisy_logits = noisy_logits.to(self.device)
        clean_mask   = clean_mask.to(self.device)
        noisy_mask   = noisy_mask.to(self.device)

        clean_logits, noisy_logits, mask = pad_align_logits_with_eot(
            clean_logits, clean_mask, noisy_logits, noisy_mask,
            self.allowed_ids_t, self.tokenizer.eot
        )
        B, S, C = clean_logits.shape

        noise = torch.randn_like(clean_logits)
        mean_l, std_l = self.sde.marginal_prob(clean_logits, noisy_logits, t, is_logits=True)
        x_t_l = mean_l + std_l[:, None, None] * noise

        # AR bootstrap with SOT onehot
        match = (self.allowed_ids_t == self.tokenizer.sot).nonzero(as_tuple=True)[0]
        if match.numel() == 0:
            raise ValueError("sot not in allowed token set.")
        sot_pos = int(match.item())
        one_hot = F.one_hot(torch.full((B,), sot_pos, device=self.device), num_classes=C).to(noisy_logits.dtype)
        out_logits = one_hot.unsqueeze(1)

        # keep the same schedule/params
        sigma_l = self.sde._std(t, is_logits=True)[:, None]  # [B,1]

        predicted_scores = []
        for s_idx in range(S):
            x_t_step   = x_t_l[:, s_idx, :]
            noisy_step = noisy_logits[:, s_idx, :]
            pred = self.logits_model(x_t_step, noisy_step, std_l, noisy_audio_embedding, out_logits)
            predicted_scores.append(pred.unsqueeze(1))
            out_logits = torch.cat([out_logits, noisy_step.unsqueeze(1)], dim=1)

        predicted_all = torch.cat(predicted_scores, dim=1)
        resid = predicted_all * sigma_l[:, None] + noise
        sq = resid.pow(2).sum(-1)
        m  = mask.to(sq.dtype)
        loss = 0.5 * (sq * m).sum() / m.sum().clamp_min(1.0)
        return loss, B
    
    def training_step(self, batch, batch_idx):
        loss, B = self._loss_from_batch(batch)

        # logging (sync across DDP ranks)
        self.log("train/loss_step",  loss, on_step=True,  on_epoch=False, prog_bar=True,  batch_size=B, sync_dist=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True,  prog_bar=True,  batch_size=B, sync_dist=True)

        cur_lr = self.optimizers().param_groups[0]["lr"] if self.optimizers() else self.hparams.initial_lr
        self.log("train/lr", cur_lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("trainer/global_step", float(self.global_step), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Lightning runs val with no grads by default
        loss, B = self._loss_from_batch(batch)
        self.log("val/loss_step",  loss, on_step=True,  on_epoch=False, prog_bar=True,  batch_size=B, sync_dist=True)
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True,  prog_bar=True,  batch_size=B, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.logits_model.parameters(), lr=self.hparams.initial_lr)

        # exact per-step linear decay 1e-3 -> 1e-4 across total_steps_for_sched
        total = max(int(self.hparams.total_steps_for_sched), 1)
        # LinearLR decays from start_factor to end_factor over `total_iters` steps.
        start_factor = 1.0
        end_factor = float(self.hparams.final_lr / self.hparams.initial_lr)
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=start_factor, end_factor=end_factor, total_iters=total)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",  # step-wise decay
                "frequency": 1,
                "name": "linear_decay",
            },
        }

# ----------------------- CLI / main -----------------------

def add_argparse_args(parser: ArgumentParser):
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--t_eps", type=float, default=0.03)
    parser.add_argument("--logits_pretrain_ckpt", type=str, required=True, help="Path to save the pretrained logits model checkpoint")
    parser.add_argument("--wandb_project", type=str, default="logits_pretrain")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=0, help="If >0, save snapshot every N steps.")
    parser.add_argument("--sr", type=int, default=48000)

    # lightning trainer knobs
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="auto",
                        help="'auto' or an int like 1,2,8")
    parser.add_argument("--strategy", type=str, default="ddp",
                        help="'auto', 'ddp', etc.")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--resume_ckpt", type=str, default=None)

    # Whisper
    parser.add_argument("--whisper_name", type=str, default="base")
    parser.add_argument("--whisper_lang", type=str, default="en")
    parser.add_argument("--transcripts_path", type=str, required=True, help="Path to the transcripts JSON file")

    # SDE + DataModule
    parser.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")


def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


def main():
    base = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for p in (base, parser):
        add_argparse_args(p)

    # Pull SDE + DataModule arg schemas like train.py
    sde_tmp = SDERegistry.get_by_name("ouve")
    dm_cls = SpecsDataModule

    sde_group = parser.add_argument_group("SDE", description="SDE class args")
    sde_tmp.add_argparse_args(sde_group)

    dm_group = parser.add_argument_group("DataModule", description=dm_cls.__name__)
    dm_cls.add_argparse_args(dm_group)

    args = parser.parse_args()

    # ---- Build DM to compute total steps for LR schedule ----
    dm_args = vars(get_argparse_groups(parser, args)["DataModule"]).copy()
    dm = dm_cls(**dm_args)
    dm.setup(stage="fit")

    # steps per epoch = number of optimizer steps (considering grad accumulation)
    train_batches = len(dm.train_dataloader())
    accum = max(int(args.accumulate_grad_batches), 1)
    steps_per_epoch = math.ceil(train_batches / accum)
    total_steps = steps_per_epoch * int(args.epochs)

    # ---- Instantiate SDE with chosen name ----
    sde_name = args.sde
    sde_kwargs = vars(get_argparse_groups(parser, args)["SDE"])  # pass-through

    # ---- Whisper tokenizer to get allowed size (build temporarily on CPU) ----
    tok = whisper.tokenizer.get_tokenizer(True, language=args.whisper_lang, task="transcribe")
    allowed_toks = build_allowed_token_id_set_with_tok(args.transcripts_path, tok)
    logits_size = len(allowed_toks)

    # ---- LightningModule ----
    module = LogitsPretrainModule(
        sde_name=sde_name,
        sde_kwargs=sde_kwargs,
        logits_size=logits_size,
        sr=args.sr,
        t_eps=args.t_eps,
        whisper_name=args.whisper_name,
        whisper_lang=args.whisper_lang,
        transcripts_path=args.transcripts_path,
        initial_lr=1e-4,
        final_lr=1e-5,
        total_steps_for_sched=total_steps,
    )

    # ---- Logging ----
    run_name = args.wandb_name or f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name, log_model=False)

    # ---- Checkpointing ----
    ckpt_dir = os.path.dirname(args.logits_pretrain_ckpt) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=os.path.basename(args.logits_pretrain_ckpt).replace(".pt", "") + "-best",
        monitor="val/loss_epoch",      
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        every_n_train_steps=None,
    )

    callbacks = [best_ckpt, LearningRateMonitor(logging_interval="step")]

    if args.save_every and int(args.save_every) > 0:
        every_n = int(args.save_every)
        step_ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=os.path.basename(args.logits_pretrain_ckpt).replace(".pt", "") + "-step{step}",
            every_n_train_steps=every_n,
            save_top_k=-1,
            save_last=False,
            save_weights_only=True,
        )
        callbacks.append(step_ckpt)

    # ---- Strategy ----
    strategy = args.strategy
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False, broadcast_buffers=False)

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=accum,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # ---- Fit ----
    trainer.fit(model=module, datamodule=dm, ckpt_path=args.resume_ckpt)

    # ---- Save best weights to target path for backward-compatibility (.pt) ----
    if best_ckpt.best_model_path:
        # load state dict from best checkpoint and save to user-specified .pt
        ckpt = torch.load(best_ckpt.best_model_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Lightning prefixes with 'logits_model.' in state_dict; keep full state_dict for clarity
        torch.save({
            "state_dict": state_dict,
            "meta": {
                "best_path": best_ckpt.best_model_path,
                "total_steps": total_steps,
                "allowed_toks_len": logits_size,
            }
        }, args.logits_pretrain_ckpt)
        print(f"[logits pretrain] Best saved -> {args.logits_pretrain_ckpt}")


if __name__ == "__main__":
    main()
