#!/usr/bin/env python3
"""
Compute WER between clean and enhanced audios using Whisper ASR.

Matching rule: a WAV from enhanced corresponds to a WAV from clean
if the filename (without extension) shares the same numeric ID
before the first underscore, e.g., 00000_*.

Outputs:
  - <out_dir>/transcripts/{fid}_clean.txt
  - <out_dir>/transcripts/{fid}_enhanced.txt
  - <out_dir>/metrics.json  (overall WER)
"""
import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchaudio
import torchmetrics
# avoid shadowing the module
import whisper as whisper_module
from whisper.normalizers import EnglishTextNormalizer

ID_REGEX = re.compile(r"^(\d+)_")

def extract_id(stem: str) -> str:
    m = ID_REGEX.match(stem)
    return m.group(1) if m else ""

def find_wavs(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in root.rglob("*.wav"):
        fid = extract_id(p.stem)
        if not fid:
            continue
        if fid in mapping and mapping[fid] != p:
            print(f"[warn] duplicate id {fid}: keeping {p}, previously {mapping[fid]}")
        mapping[fid] = p
    return mapping

def build_allowed_token_id_set_with_tok(transcripts_path, tok):
    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    allowed = set()
    # (0) Special/control tokens
    allowed.update(tok.sot_sequence_including_notimestamps)
    allowed.add(tok.eot)
    # (1) whitespace / separators
    for s in [" ", "  ", "\n", "\n\n", "\t"]:
        allowed.update(tok.encode(s))
    # (2) punctuation (with/without leading space)
    for punct in [".", ",", "!", "?", "'", '"', ":", ";", "-", "—", "…", "(", ")", "[", "]"]:
        for s in (punct, " " + punct):
            allowed.update(tok.encode(s))
    # (3) sentences as-is and with leading space
    for _, text in transcripts.items():
        allowed.update(tok.encode(text))
        allowed.update(tok.encode(" " + text))
    # (4) safety net over concatenated corpus
    corpus = " " + " ".join(transcripts.values())
    allowed.update(tok.encode(corpus))
    corpus_nospace = "".join(transcripts.values())
    allowed.update(tok.encode(corpus_nospace))
    return sorted(allowed)

def pad_or_trim(x: torch.Tensor, length: int = 30 * 16_000):
    if x.size(-1) < length:
        x = F.pad(x, (0, length - x.size(-1)))
    else:
        x = x[..., :length]
    return x

def log_mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 400,
    hop: int = 160,
    n_mels: int = 80,
    sr: int = 16_000,
    f_min: float = 0.0,
    f_max: float = 8_000.0,
):
    # keep your custom implementation
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    batch_size = audio.size(0)
    log_mels = []

    for i in range(batch_size):
        audio_single = audio[i]
        if audio_single.dtype != torch.float32:
            audio_single = audio_single.float()

        window = torch.hann_window(n_fft, device=audio_single.device, dtype=audio_single.dtype)
        stft = torch.stft(
            audio_single, n_fft, hop,
            window=window, win_length=n_fft,
            center=True, pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().pow(2.0)

        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            sample_rate=sr,
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

        # keep parity reference (not used)
        _ = whisper_module.log_mel_spectrogram(audio_single)

        log_mels.append(log_mel)

    return torch.stack(log_mels, dim=0)  # (B, n_mels, time_frames)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--enhanced_root", type=str, required=False,
    #                     default="/mlspeech/data/gilad/paper_ears_reverbed/EARS-Reverb_v2_4sec_chunks_test_enhanced_with_logits_nested")
    # parser.add_argument("--clean_root", type=str, required=False,
    #                     default="/mlspeech/data/gilad/paper_ears_reverbed/EARS-Reverb_v2_4sec_chunks/test/clean")
    parser.add_argument("--enhanced_root", type=str, required=False,
                        default="/mlspeech/data/gilad/paper_ears_wham/EARS-WHAM_v2_4sec_chunks_test_enhanced_with_logits_nested_epoch20")
    parser.add_argument("--clean_root", type=str, required=False,
                        default="/mlspeech/data/gilad/paper_ears_wham/EARS-WHAM_v2_4sec_chunks/test/clean")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="ears_wer_output")
    parser.add_argument("--transcripts_json", type=str,
                        default="/mlspeech/data/gilad/paper_ears_reverbed/transcripts.json")
    args = parser.parse_args()

    enhanced_root = Path(args.enhanced_root).expanduser().resolve()
    clean_root = Path(args.clean_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if not enhanced_root.exists():
        print(f"[error] Enhanced root not found: {enhanced_root}")
        sys.exit(1)
    if not clean_root.exists():
        print(f"[error] Clean root not found: {clean_root}")
        sys.exit(1)

    print("[info] Scanning WAVs…")
    enhanced_map = find_wavs(enhanced_root)
    clean_map = find_wavs(clean_root)
    ids = sorted(set(enhanced_map.keys()) & set(clean_map.keys()))
    if not ids:
        print("[error] No matching IDs were found between the two roots.")
        sys.exit(1)
    print(f"[info] Matched {len(ids)} pairs.")

    print(f"[info] Loading Whisper model: {args.model} on {args.device}")
    wer_metric = torchmetrics.WordErrorRate()
    wmodel = whisper_module.load_model(args.model).to(args.device)
    options = whisper_module.DecodingOptions(language=args.language, without_timestamps=True)
    tok = whisper_module.tokenizer.get_tokenizer(True, language=args.language, task=options.task)

    allowed_toks = build_allowed_token_id_set_with_tok(args.transcripts_json, tok)
    normalizer = EnglishTextNormalizer()

    # -------- decode helpers (unchanged logic) --------
    device = args.device
    n_vocab = tok.encoding.n_vocab
    allowed_ids_t = torch.as_tensor(allowed_toks, dtype=torch.long, device=device)
    neg_inf = -1e9
    base_mask = torch.full((n_vocab,), neg_inf, device=device)
    base_mask[allowed_ids_t] = 0.0
    max_decode_len = 224

    # @torch.no_grad()
    # def decode_constrained(signal_16k):
    #     """
    #     Unconstrained greedy decode for Whisper's internal state progression.
    #     But we retrospectively extract logits for only the allowed set at each step,
    #     AND construct the output text from argmax of the allowed set.
    #     Returns (pred_text, mel, step_logits_allowed[S,|A|], audio_embedding[1,T',C]).
    #     """
    #     wav = torch.tensor(signal_16k, dtype=torch.float32, device=device)
    #     true_audio_length = wav.shape[-1]
    #     wav = pad_or_trim(wav)  # (T,)
    #     mel = log_mel_spectrogram(wav)  # (1, 80, Tm)
    #     feats = wmodel.encoder(mel)     # (1, T', C)

    #     hop_length = 160
    #     encoder_downsampling_factor = 4
    #     true_mel_length = (true_audio_length + hop_length - 1) // hop_length
    #     true_encoder_length = (true_mel_length + encoder_downsampling_factor - 1) // encoder_downsampling_factor
    #     audio_embedding = feats[:, :true_encoder_length, :]

    #     prefix = torch.tensor([tok.sot_sequence_including_notimestamps],
    #                         dtype=torch.long, device=device)  # (1, L0)
    #     out = prefix.clone()
    #     per_step_allowed = []  # store constrained logits
    #     constrained_tokens = []  # store tokens from closed set

    #     for _ in range(max_decode_len):
    #         logits = wmodel.decoder(out, feats).squeeze(0)  # (L, V)
    #         next_logits = logits[-1]                        # (V,)

    #         # Extract and normalize logits for ONLY the allowed set
    #         allowed_logits = next_logits.index_select(0, allowed_ids_t)  # (|A|,)
    #         allowed_probs = torch.softmax(allowed_logits.unsqueeze(0), dim=-1)  # (1, |A|)
    #         per_step_allowed.append(allowed_probs)

    #         # Pick the best token from the ALLOWED SET for output text
    #         best_allowed_idx = int(torch.argmax(allowed_logits).item())
    #         constrained_token_id = allowed_ids_t[best_allowed_idx].item()
    #         constrained_tokens.append(constrained_token_id)

    #         # But continue decoding with UNCONSTRAINED token for next step
    #         next_id = int(torch.argmax(next_logits).item())
    #         out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)
            
    #         # Check EOT from the constrained token (for output text)
    #         if constrained_token_id == tok.eot:
    #             break

    #     step_logits_allowed = torch.cat(per_step_allowed, dim=0)  # (S, |A|)
        
    #     # Decode text from the CONSTRAINED tokens
    #     if len(constrained_tokens) > 0 and constrained_tokens[-1] == tok.eot:
    #         constrained_tokens = constrained_tokens[:-1]
    #     text = tok.decode(constrained_tokens)
        
    #     return text, mel, step_logits_allowed, audio_embedding

    @torch.no_grad()
    def decode_constrained(signal_16k):
        """
        Greedy, closed-set decode: at each step, mask logits to allowed set.
        Returns (pred_text, mel) where pred_text is the decoded string.
        """

        # to device
        wav = torch.tensor(signal_16k, dtype=torch.float32, device=device)
        wav = pad_or_trim(wav)  # (T,)
        mel = log_mel_spectrogram(wav)  # (1, 80, Tm)
        feats = wmodel.encoder(mel)  # (1, T', C)

        # start with SOT + <|notimestamps|>
        prefix = torch.tensor([tok.sot_sequence_including_notimestamps],
                            dtype=torch.long, device=device)  # (1, L0)
        out = prefix.clone()
        constrained_tokens = []

        for _ in range(max_decode_len):
            # logits over full vocab for each position so far
            logits = wmodel.decoder(out, feats).squeeze(0)  # (L, V)
            next_logits = logits[-1]  # (V,)
            # record only the allowed positions' logits for this step -> (|A|,)
            # allowed_step = torch.softmax(next_logits.index_select(0, allowed_ids_t).unsqueeze(0),dim=-1)  # (1, |A|)
            best_allowed_idx = int(torch.argmax(next_logits.index_select(0, allowed_ids_t)).item())
            constrained_token_id = allowed_ids_t[best_allowed_idx].item()
            constrained_tokens.append(constrained_token_id)

            # allowed_step = next_logits.index_select(0, allowed_ids_t).unsqueeze(0)  # (1, |A|)
            # allowed_step = normalize_logits(allowed_step, target_mean, target_std)
            # per_step_allowed.append(allowed_step)

            # apply closed-set mask
            # next_logits = next_logits + base_mask

            # greedy pick
            next_id = int(torch.argmax(next_logits).item())
            out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)

            if constrained_token_id == tok.eot:
                break
        
        if len(constrained_tokens) > 0 and constrained_tokens[-1] == tok.eot:
            constrained_tokens = constrained_tokens[:-1]
        text = tok.decode(constrained_tokens)
        return text
    
    # def decode_only_constrained(signal_16k):
    #     """
    #     Greedy, closed-set decode: at each step, mask logits to allowed set.
    #     Returns (pred_text, mel, step_logits_allowed[S,|A|], audio_embedding[1,T',C]).
    #     (We won't save the logits/embeddings, just keep them local.)
    #     """
    #     wav = torch.tensor(signal_16k, dtype=torch.float32, device=device)
    #     true_audio_length = wav.shape[-1]
    #     wav = pad_or_trim(wav)  # (T,)
    #     mel = log_mel_spectrogram(wav)  # (1, 80, Tm)
    #     feats = wmodel.encoder(mel)     # (1, T', C)

    #     hop_length = 160
    #     encoder_downsampling_factor = 4
    #     true_mel_length = (true_audio_length + hop_length - 1) // hop_length
    #     true_encoder_length = (true_mel_length + encoder_downsampling_factor - 1) // encoder_downsampling_factor
    #     audio_embedding = feats[:, :true_encoder_length, :]

    #     prefix = torch.tensor([tok.sot_sequence_including_notimestamps],
    #                           dtype=torch.long, device=device)  # (1, L0)
    #     out = prefix.clone()
    #     per_step_allowed = []  # store for completeness (not saved)

    #     for _ in range(max_decode_len):
    #         logits = wmodel.decoder(out, feats).squeeze(0)  # (L, V)
    #         next_logits = logits[-1]                        # (V,)

    #         allowed_step = torch.softmax(
    #             next_logits.index_select(0, allowed_ids_t).unsqueeze(0), dim=-1
    #         )
    #         per_step_allowed.append(allowed_step)

    #         next_logits = next_logits + base_mask
    #         next_id = int(torch.argmax(next_logits).item())
    #         out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)
    #         if next_id == tok.eot:
    #             break

    #     step_logits_allowed = torch.cat(per_step_allowed, dim=0)  # (S, |A|)
    #     gen = out[0].tolist()
    #     start = len(tok.sot_sequence_including_notimestamps)
    #     if len(gen) > 0 and gen[-1] == tok.eot:
    #         gen = gen[:-1]
    #     text = tok.decode(gen[start:])
    #     return text, mel, step_logits_allowed, audio_embedding

    # -------- main loop: ONLY save transcripts + final WER --------
    print("[info] Transcribing and computing WER…")
    hyps_c, refs_c, refs_only_c, hyps_only_c, hyps_r, refs_r = [], [], [], [], [], []

    for fid in tqdm(ids):
        clean_path = clean_map[fid]
        enh_path = enhanced_map[fid]

        clean_audio, sr = torchaudio.load(clean_path)
        enh_audio, _ = torchaudio.load(enh_path)

        if sr != 16000:
            clean_audio_16k = torchaudio.functional.resample(clean_audio, sr, 16000).squeeze()
            enh_audio_16k = torchaudio.functional.resample(enh_audio, sr, 16000).squeeze()
        else:
            clean_audio_16k = clean_audio.squeeze()
            enh_audio_16k = enh_audio.squeeze()

        # ---------- constrained decoding ----------
        enh_text_c= decode_constrained(enh_audio_16k)
        clean_text_c = decode_constrained(clean_audio_16k)
        # enh_text_c, _, _, _ = decode_constrained(enh_audio_16k)
        # clean_text_c, _, _, _ = decode_constrained(clean_audio_16k)
        # enh_text_only_c, _, _, _ = decode_only_constrained(enh_audio_16k)
        # clean_text_only_c, _, _, _ = decode_only_constrained(clean_audio_16k)

        # # ---------- regular Whisper decoding ----------
        # with torch.no_grad():
        #     clean_wav_30s = whisper_module.pad_or_trim(clean_audio_16k.to(device, dtype=torch.float32))
        #     enh_wav_30s   = whisper_module.pad_or_trim(enh_audio_16k.to(device, dtype=torch.float32))

        #     clean_mel = whisper_module.log_mel_spectrogram(clean_wav_30s).to(device)  # [80, T]
        #     enh_mel   = whisper_module.log_mel_spectrogram(enh_wav_30s).to(device)    # [80, T]

        #     clean_reg = wmodel.decode(clean_mel, options)
        #     enh_reg   = wmodel.decode(enh_mel,   options)


        # clean_text_r = clean_reg.text
        # enh_text_r   = enh_reg.text

        # ---------- normalization ----------
        clean_text_c_n = normalizer(clean_text_c)
        enh_text_c_n   = normalizer(enh_text_c)
        # clean_text_only_c_n = normalizer(clean_text_only_c)
        # enh_text_only_c_n   = normalizer(enh_text_only_c)
        # clean_text_r_n = normalizer(clean_text_r)
        # enh_text_r_n   = normalizer(enh_text_r)

        # ---------- save all into one file ----------
        # combined_text = (
        #     f"[CLEAN - CONSTRAINED]\n{clean_text_c_n}\n\n"
        #     f"[ENHANCED - CONSTRAINED]\n{enh_text_c_n}\n\n"
        #     f"[CLEAN - REGULAR]\n{clean_text_r_n}\n\n"
        #     f"[ENHANCED - REGULAR]\n{enh_text_r_n}\n"
        # )
        # (transcripts_dir / f"{fid}.txt").write_text(combined_text, encoding="utf-8")

        refs_c.append(clean_text_c_n)
        hyps_c.append(enh_text_c_n)
        # refs_only_c.append(clean_text_only_c_n)
        # hyps_only_c.append(enh_text_only_c_n)
        # refs_r.append(clean_text_r_n)
        # hyps_r.append(enh_text_r_n)

    wer_c = wer_metric(hyps_c, refs_c).item()
    # wer_only_c = wer_metric(hyps_only_c, refs_only_c).item()
    # wer_r = wer_metric(hyps_r, refs_r).item()
    # (out_dir / "metrics_c.json").write_text(json.dumps({"num_pairs": len(ids), "wer": wer_c}, indent=2), encoding="utf-8")
    # (out_dir / "metrics_r.json").write_text(json.dumps({"num_pairs": len(ids), "wer": wer_r}, indent=2), encoding="utf-8")

    print(f"[done] Wrote transcripts to: {transcripts_dir}")
    print(f"[done] Wrote overall WER to: {out_dir / 'metrics.json'}")
    print(f"[result] Overall WER constrained (enhanced vs clean): {wer_c:.4f}")
    # print(f"[result] Overall WER constrained only (enhanced vs clean): {wer_only_c:.4f}")
    # print(f"[result] Overall WER regular (enhanced vs clean): {wer_r:.4f}")

if __name__ == "__main__":
    main()
