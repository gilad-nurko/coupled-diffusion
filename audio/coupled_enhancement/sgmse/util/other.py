import os
import torch
import numpy as np
import scipy.stats
from scipy.signal import butter, sosfilt
import json
import re
from typing import List, Tuple, Optional

from pesq import pesq
from pystoi import stoi


def si_sdr_components(s_hat, s, n):
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))

def hp_filter(signal, cut_off=80, order=10, sr=16000):
    factor = cut_off /sr * 2
    sos = butter(order, factor, 'hp', output='sos')
    filtered = sosfilt(sos, signal)
    return filtered

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def snr_dB(s,n):
    s_power = 1/len(s)*np.sum(s**2)
    n_power = 1/len(n)*np.sum(n**2)
    snr_dB = 10*np.log10(s_power/n_power)
    return snr_dB

def pad_spec(Y, mode="zero_pad"):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    if mode == "zero_pad":
        pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    elif mode == "reflection":
        pad2d = torch.nn.ReflectionPad2d((0, num_pad, 0,0))
    elif mode == "replication":
        pad2d = torch.nn.ReplicationPad2d((0, num_pad, 0,0))
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return pad2d(Y)

def ensure_dir(file_path):
    directory = file_path
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_metrics(x, y, x_hat_list, labels, sr=16000):
    _si_sdr_mix = si_sdr(x, y)
    _pesq_mix = pesq(sr, x, y, 'wb')
    _estoi_mix = stoi(x, y, sr, extended=True)
    print(f'Mixture:  PESQ: {_pesq_mix:.2f}, ESTOI: {_estoi_mix:.2f}, SI-SDR: {_si_sdr_mix:.2f}')
    for i, x_hat in enumerate(x_hat_list):
        _si_sdr = si_sdr(x, x_hat)
        _pesq = pesq(sr, x, x_hat, 'wb')
        _estoi = stoi(x, x_hat, sr, extended=True)
        print(f'{labels[i]}: {_pesq:.2f}, ESTOI: {_estoi:.2f}, SI-SDR: {_si_sdr:.2f}')

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def print_mean_std(data, decimal=2):
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    if decimal == 2:
        string = f'{mean:.2f} ± {std:.2f}'
    elif decimal == 1:
        string = f'{mean:.1f} ± {std:.1f}'
    return string

def set_torch_cuda_arch_list():
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs found.")
        return
    
    num_gpus = torch.cuda.device_count()
    compute_capabilities = []

    for i in range(num_gpus):
        cc_major, cc_minor = torch.cuda.get_device_capability(i)
        cc = f"{cc_major}.{cc_minor}"
        compute_capabilities.append(cc)
    
    cc_string = ";".join(compute_capabilities)
    os.environ['TORCH_CUDA_ARCH_LIST'] = cc_string
    print(f"Set TORCH_CUDA_ARCH_LIST to: {cc_string}")

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

def _pairwise_cost(
    clean: torch.Tensor,   # [S_c, A]
    noisy: torch.Tensor,   # [S_n, A]
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Returns C in R^{S_c x S_n} with C[i,j] = distance(clean[i], noisy[j]).
    Implemented using matmul (no A-dimension broadcasting).
    """
    if metric == "cosine":
        # normalize along A
        eps = 1e-12
        c = clean / (clean.norm(dim=-1, keepdim=True) + eps)
        n = noisy / (noisy.norm(dim=-1, keepdim=True) + eps)
        sim = c @ n.T                                  # [S_c, S_n]
        C = 1.0 - sim                                  # cosine distance
    elif metric == "l2":
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        clean_n2 = (clean * clean).sum(dim=-1)         # [S_c]
        noisy_n2 = (noisy * noisy).sum(dim=-1)         # [S_n]
        dot = clean @ noisy.T                          # [S_c, S_n]
        C = clean_n2[:, None] + noisy_n2[None, :] - 2.0 * dot
        C = C.clamp_min_(0.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return C

def _dtw_cost_and_prev(
    C: torch.Tensor,                   # [S_c, S_n]
    band: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classic DTW DP with optional Sakoe–Chiba band.
    Returns (D, prev) where:
      D[i,j] = min cumulative cost to (i,j)
      prev[i,j] in {0,1,2} encodes argmin predecessor:
        0 = (i-1, j)   (up)
        1 = (i,   j-1) (left)
        2 = (i-1, j-1) (diag)
    """
    device = C.device
    S_c, S_n = C.shape
    inf = torch.tensor(float("inf"), device=device)
    D = torch.full((S_c, S_n), inf, device=device)
    prev = torch.full((S_c, S_n), -1, dtype=torch.int8, device=device)

    # Optional band mask
    if band is not None:
        I = torch.arange(S_c, device=device)[:, None]
        J = torch.arange(S_n, device=device)[None, :]
        mask = (J - I).abs() <= band
        C = C.masked_fill(~mask, float("inf"))

    D[0, 0]   = C[0, 0]
    prev[0,0] = 2

    # first row/col
    for i in range(1, S_c):
        if torch.isinf(C[i,0]): break
        D[i,0] = C[i,0] + D[i-1,0]
        prev[i,0] = 0
    for j in range(1, S_n):
        if torch.isinf(C[0,j]): break
        D[0,j] = C[0,j] + D[0,j-1]
        prev[0,j] = 1

    # main DP
    for i in range(1, S_c):
        # band skip: find feasible Js
        j_start = 0 if band is None else max(1, i - band)
        j_end   = S_n if band is None else min(S_n, i + band + 1)
        for j in range(j_start, j_end):
            if torch.isinf(C[i,j]): 
                continue
            up   = D[i-1, j]
            left = D[i,   j-1]
            diag = D[i-1, j-1]
            # choose predecessor
            m = torch.min(torch.stack([up, left, diag], dim=0), dim=0)
            D[i, j] = C[i, j] + m.values
            prev[i, j] = m.indices.to(torch.int8)

    return D, prev

def _backtrack(prev: torch.Tensor) -> List[Tuple[int,int]]:
    """
    Backtrack DTW path from (S_c-1,S_n-1) to (0,0).
    Returns list of (i,j) from start to end (ascending order).
    """
    i, j = prev.shape[0]-1, prev.shape[1]-1
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        p = int(prev[i, j].item())
        if p == 2:      # diag
            i, j = i-1, j-1
        elif p == 0:    # up
            i = i-1
        elif p == 1:    # left
            j = j-1
        else:
            # unreachable / masked zone safeguard (shouldn't happen if path exists)
            break
    path.reverse()
    return path

def _aggregate_to_clean_len(
    noisy: torch.Tensor,                  # [S_n, A]
    path: List[Tuple[int,int]],
    S_c: int
) -> torch.Tensor:
    """
    Make length exactly S_c by averaging all noisy steps aligned to each clean i.
    Ensures at least one j per i (true for standard DTW).
    """
    A = noisy.shape[-1]
    out = torch.zeros((S_c, A), device=noisy.device, dtype=noisy.dtype)
    counts = torch.zeros((S_c,), device=noisy.device, dtype=torch.int32)
    for (i, j) in path:
        out[i] += noisy[j]
        counts[i] += 1
    counts = torch.clamp(counts, min=1).to(noisy.dtype)
    out = out / counts[:, None]
    return out

def dtw_align_single(
    clean: torch.Tensor,    # [S_c, A]
    noisy: torch.Tensor,    # [S_n, A]
    metric: str = "cosine",
    band: Optional[int] = None
) -> Tuple[torch.Tensor, float, List[Tuple[int,int]]]:
    """
    Returns:
      aligned_noisy: [S_c, A]     (noisy warped to clean length)
      total_cost: float
      path: list of (i,j)
    """
    C = _pairwise_cost(clean, noisy, metric=metric)        # [S_c, S_n]
    D, prev = _dtw_cost_and_prev(C, band=band)
    path = _backtrack(prev)
    aligned_noisy = _aggregate_to_clean_len(noisy, path, clean.shape[0])
    total_cost = float(D[-1, -1].item())
    return aligned_noisy, total_cost, path

def dtw_align_batch(
    clean_logits: torch.Tensor,     # [B, S_c_max, A]
    noisy_logits: torch.Tensor,     # [B, S_n_max, A]
    clean_lengths: Optional[torch.Tensor] = None,  # [B] int
    noisy_lengths: Optional[torch.Tensor] = None,  # [B] int
    metric: str = "cosine",
    band_ratio: Optional[float] = 0.2,            # e.g., 20% band; set None for full
) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int,int]]]]:
    """
    For each b, align noisy[b] to clean[b]'s length.
    Returns:
      aligned_noisy_padded: [B, S_c_max, A]
      out_lengths: [B] (== clean_lengths if provided, else S_c_max)
      paths: list of paths for each b
    """
    B, S_c_max, A = clean_logits.shape
    _, S_n_max, _ = noisy_logits.shape
    device = clean_logits.device
    if clean_lengths is None:
        clean_lengths = torch.full((B,), S_c_max, device=device, dtype=torch.int64)
    if noisy_lengths is None:
        noisy_lengths = torch.full((B,), S_n_max, device=device, dtype=torch.int64)

    aligned = []
    paths = []
    for b in range(B):
        S_c = int(clean_lengths[b].item())
        S_n = int(noisy_lengths[b].item())
        clean_b = clean_logits[b, :S_c, :]       # [S_c, A]
        noisy_b = noisy_logits[b, :S_n, :]       # [S_n, A]
        band = None
        if band_ratio is not None:
            band = max(0, int(band_ratio * max(S_c, S_n)))
        aligned_b, _, path_b = dtw_align_single(
            clean_b, noisy_b, metric=metric, band=band
        )
        # pad back to S_c_max
        pad_len = S_c_max - aligned_b.shape[0]
        if pad_len > 0:
            aligned_b = torch.cat(
                [aligned_b, aligned_b.new_zeros(pad_len, A)], dim=0
            )
        aligned.append(aligned_b)
        paths.append(path_b)
    aligned_noisy_padded = torch.stack(aligned, dim=0)      # [B, S_c_max, A]
    return aligned_noisy_padded, clean_lengths, paths

def normalize_logits(logits: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    target_mean = target_mean.to(logits.device)
    target_std = target_std.to(logits.device)
    logits_mean = logits.mean(dim=1, keepdim=True)  # [B,1]
    logits_std  = logits.std(dim=1, keepdim=True)   # [B,1]

    # avoid divide-by-zero
    eps = 1e-8
    normalized_logits = (logits - logits_mean) / (logits_std + eps)

    logits_rescaled = normalized_logits * target_std + target_mean  # [B,|A|]
    return logits_rescaled