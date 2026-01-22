#!/usr/bin/env python
# ears_make_reverb_like_wsj0.py
"""
EARS reverberation like the WSJ0 script, with flat outputs and custom naming.

- Input root: /mlspeech/data/gilad/ears_dataset
  Expected layout: p001/**/*.wav, p002/**/*.wav, ...
- Allowed files are filtered by KEYS of transcripts.json (basenames).
- Output (flat, no nested person/session dirs):
    <target>/
      train/
        clean/  # anechoic (very dry, padded)
        noisy/  # reverberant
      valid/
        clean/
        noisy/
      test/
        clean/
        noisy/

Filenames:
  <json_key>_person_<pXXX>.wav
  e.g., emo_adoration_sentences_person_p001.wav

Example:
  python ears_make_reverb_like_wsj0.py \
      --data /mlspeech/data/gilad/ears_dataset \
      --json /mlspeech/data/gilad/ears_dataset/transcripts.json \
      --target /mlspeech/data/gilad/ears_dataset_reverb_likewsj0
"""

import os
import json
import argparse
import shutil
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from glob import glob
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# ----------------------- Parameters (mirroring your snippet) ------------------ #
SEED = 100
np.random.seed(SEED)

T60_RANGE = [0.4, 1.0]                 # seconds
SNR_RANGE = [0, 20]                    # (kept for parity; not used here)
DIM_RANGE = [5, 15, 5, 15, 2, 6]       # [xmin,xmax, ymin,ymax, zmin,zmax] (m)
MIN_DISTANCE_TO_WALL = 1.0             # meters
MIC_ARRAY_RADIUS = 0.16
TARGET_T60_SHAPE = {"CI": 0.10, "HA": 0.2}   # kept for parity; not used
TARGETS_CROP = {"CI": 16e-3, "HA": 40e-3}    # kept for parity; not used
NB_SAMPLES_PER_ROOM = 1
CHANNELS = 1
SAMPLE_RATE = 48000

# ------------------------------- Helpers ------------------------------------- #
def normalize_key(name: str):
    base = os.path.basename(name)
    base_lower = base.lower()
    root, ext = os.path.splitext(base_lower)
    return base_lower, root  # with ext (lower), without ext (lower)

def collect_allowed_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    keys = list(j.keys())
    if not keys:
        raise RuntimeError("No keys found in transcripts.json")
    allowed_with_ext, allowed_wo_ext = set(), set()
    for k in keys:
        wext, wout = normalize_key(k)
        allowed_with_ext.add(wext)
        allowed_wo_ext.add(wout)
    return allowed_with_ext, allowed_wo_ext

def is_allowed_path(wav_path: str, allowed_with_ext, allowed_wo_ext):
    wext, wout = normalize_key(wav_path)
    return (wext in allowed_with_ext) or (wout in allowed_wo_ext)

def obtain_clean_file(path: str, sample_rate: int = SAMPLE_RATE):
    speech, sr = sf.read(path)
    if sr != sample_rate:
        raise AssertionError(f"Expected {sample_rate} Hz, got {sr} for {path}")
    return speech.squeeze(), sr

def unique_path(base_dir: Path, filename: str) -> Path:
    """
    Ensure we don't overwrite: if filename exists, append _dupN before extension.
    """
    p = base_dir / filename
    if not p.exists():
        return p
    stem, ext = os.path.splitext(filename)
    n = 1
    while True:
        p_try = base_dir / f"{stem}_dup{n}{ext}"
        if not p_try.exists():
            return p_try
        n += 1

# ------------------------------- Main ---------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/mlspeech/data/gilad/ears_dataset",
                        help="Root containing p001, p002, ...")
    parser.add_argument("--json", type=str, default="/mlspeech/data/gilad/ears_dataset/transcripts.json",
                        help="Path to transcripts.json whose KEYS are allowed basenames")
    parser.add_argument("--target", type=str, default="/mlspeech/data/gilad/ears_dataset_reverb",
                        help="Target directory (will be overwritten)")
    parser.add_argument("--test_size", type=float, default=0.20,
                        help="Fraction for test+valid combined")
    parser.add_argument("--valid_size_within_tv", type=float, default=0.50,
                        help="Fraction of test+valid that becomes valid")
    args = parser.parse_args()

    # Prepare target (clean)
    if os.path.exists(args.target):
        shutil.rmtree(args.target)

    # Allowed basenames from JSON keys
    print("Reading allowed basenames from JSON keys …")
    allowed_with_ext, allowed_wo_ext = collect_allowed_from_json(args.json)

    # Collect and filter wavs
    print("Collecting WAVs from EARS …")
    all_wavs_all = glob(os.path.join(args.data, "p*", "**", "*.wav"), recursive=True)
    all_wavs = [p for p in all_wavs_all if is_allowed_path(p, allowed_with_ext, allowed_wo_ext)]
    if not all_wavs:
        raise RuntimeError("No WAVs matched the keys in transcripts.json")

    print(f"Selected {len(all_wavs)} / {len(all_wavs_all)} files by JSON-key filter.")

    # Split
    train, tmp  = train_test_split(all_wavs, test_size=args.test_size, random_state=SEED, shuffle=True)
    valid, test = train_test_split(tmp,      test_size=args.valid_size_within_tv, random_state=SEED, shuffle=True)
    split_dict = {"train": train, "valid": valid, "test": test}

    # Output dirs (flat)
    def mk_split_dirs(split):
        clean_dir = Path(args.target) / split / "clean"
        noisy_dir = Path(args.target) / split / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
        return clean_dir, noisy_dir

    out_dirs = {s: mk_split_dirs(s) for s in split_dict}

    # Process
    print("Processing and simulating …")
    for split, paths in split_dict.items():
        clean_dir, noisy_dir = out_dirs[split]
        print(f"  {split.upper():5}  ({len(paths)} files)")

        reverberant_room = None

        for i_sample, wav in enumerate(tqdm(paths)):
            # Sample a new room each NB_SAMPLES_PER_ROOM
            if i_sample % NB_SAMPLES_PER_ROOM == 0:
                t60 = np.random.uniform(T60_RANGE[0], T60_RANGE[1])
                room_dim = np.array([
                    np.random.uniform(DIM_RANGE[0], DIM_RANGE[1]),
                    np.random.uniform(DIM_RANGE[2], DIM_RANGE[3]),
                    np.random.uniform(DIM_RANGE[4], DIM_RANGE[5]),
                ])
                center_mic_position = np.array([
                    np.random.uniform(MIN_DISTANCE_TO_WALL, room_dim[0] - MIN_DISTANCE_TO_WALL),
                    np.random.uniform(MIN_DISTANCE_TO_WALL, room_dim[1] - MIN_DISTANCE_TO_WALL),
                    np.random.uniform(MIN_DISTANCE_TO_WALL, room_dim[2] - MIN_DISTANCE_TO_WALL),
                ])
                source_position = np.array([
                    np.random.uniform(MIN_DISTANCE_TO_WALL, room_dim[0] - MIN_DISTANCE_TO_WALL),
                    np.random.uniform(MIN_DISTANCE_TO_WALL, room_dim[1] - MIN_DISTANCE_TO_WALL),
                    np.random.uniform(MIN_DISTANCE_TO_WALL, room_dim[2] - MIN_DISTANCE_TO_WALL),
                ])

                # microphone array (CHANNELS=1 → effectively a single mic)
                mic_array_2d = pra.beamforming.circular_2D_array(
                    center_mic_position[:-1], CHANNELS, phi0=0, radius=MIC_ARRAY_RADIUS
                )
                mic_array = np.pad(
                    mic_array_2d, ((0, 1), (0, 0)), mode="constant", constant_values=center_mic_position[-1]
                )

                # Reverberant room
                e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
                reverberant_room = pra.ShoeBox(
                    room_dim, fs=SAMPLE_RATE, materials=pra.Material(e_absorption), max_order=min(3, max_order)
                )
                reverberant_room.set_ray_tracing()
                reverberant_room.add_microphone_array(mic_array)

            # Load speech
            speech, sr = obtain_clean_file(wav, sample_rate=SAMPLE_RATE)

            # Reverberant
            reverberant_room.add_source(source_position, signal=speech)
            reverberant_room.compute_rir()
            reverberant_room.simulate()
            t60_real = float(np.mean(reverberant_room.measure_rt60()).squeeze())
            reverberant = np.stack(reverberant_room.mic_array.signals).swapaxes(0, 1)  # (T, C)

            # Dry (very low reverberation, max_order=0)
            e_absorption_dry = 0.99
            dry_room = pra.ShoeBox(
                room_dim, fs=SAMPLE_RATE, materials=pra.Material(e_absorption_dry), max_order=0
            )
            dry_room.add_microphone_array(mic_array)
            dry_room.add_source(source_position, signal=speech)
            dry_room.compute_rir()
            dry_room.simulate()
            _t60_real_dry = float(np.mean(dry_room.measure_rt60()).squeeze())
            dry = np.stack(dry_room.mic_array.signals).swapaxes(0, 1)  # (T, C)

            # Pad dry by 0.5 s to avoid cutting reverb tails alignment differences
            dry = np.pad(dry, ((0, int(0.5 * SAMPLE_RATE)), (0, 0)), mode="constant", constant_values=0)

            # Align lengths
            min_len = min(reverberant.shape[0], dry.shape[0])
            dry = dry[:min_len]
            reverberant = reverberant[:min_len]

            # Scale to avoid clipping; match your snippet behavior (max to 0.9)
            output_scaling = max(np.max(np.abs(reverberant)), 1e-9) / 0.9
            dry_out = (1.0 / output_scaling) * dry
            rev_out = (1.0 / output_scaling) * reverberant

            # File naming: <json_key>_person_<pXXX>.wav
            base_key = os.path.splitext(os.path.basename(wav))[0]  # from actual file path
            person = Path(wav).parts[-2]                           # p001, p0xx…
            new_name = f"{base_key}_person_{person}.wav"

            # Single-channel write (CHANNELS=1). If >1 channels, collapse to first.
            if rev_out.ndim == 2 and rev_out.shape[1] > 1:
                rev_write = rev_out[:, 0]
                dry_write = dry_out[:, 0]
            else:
                rev_write = rev_out.squeeze()
                dry_write = dry_out.squeeze()

            # Ensure unique filenames to avoid overwrites in flat layout
            clean_path = unique_path(clean_dir, new_name)
            noisy_path = unique_path(noisy_dir, new_name)

            sf.write(clean_path, dry_write, samplerate=SAMPLE_RATE)
            sf.write(noisy_path, rev_write, samplerate=SAMPLE_RATE)

    print("Done – reverberant dataset saved to:", args.target)
