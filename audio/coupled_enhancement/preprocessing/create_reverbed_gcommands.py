#!/usr/bin/env python
# gsc_make_reverb.py
"""
Create reverberant (“noisy”) versions of the Google Speech-Commands dataset.

Example:
    python gsc_make_reverb.py \
        --gcommands /path/to/speech_commands_v0.02 \
        --target    /path/to/speech_commands_reverb
"""
import os
from glob import glob
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from librosa import load
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------------------------------------------------------- #
#  Tunable parameters – feel free to change
# ----------------------------------------------------------------------------- #
TARGET_COMMANDS = {
    "down", "go", "left", "no", "off",
    "on", "right", "stop", "up", "yes"
}

SR             = 16_000
RT60_RANGE     = (0.25, 0.8)             # seconds
ROOM_RANGE_XY  = (3, 8)                  # metres
ROOM_RANGE_Z   = (2, 4)
MIN_DIST_WALL  = 0.5                     # speaker/mic ↔ wall
NB_SAMPLES_PER_ROOM = 1                  # draw new room every sample
SEED = 100
rng  = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------- #
#  Helper – create one random room
# ----------------------------------------------------------------------------- #
def make_random_room():
    rt60  = rng.uniform(*RT60_RANGE)
    room_dim = np.array([
        rng.uniform(*ROOM_RANGE_XY),
        rng.uniform(*ROOM_RANGE_XY),
        rng.uniform(*ROOM_RANGE_Z),
    ])

    absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=SR,
        materials=pra.Material(absorption),
        max_order=min(max_order, 3),
        ray_tracing=False
    )

    # Source ↔ Mic positions
    src = np.array([
        rng.uniform(MIN_DIST_WALL, room_dim[0] - MIN_DIST_WALL),
        rng.uniform(MIN_DIST_WALL, room_dim[1] - MIN_DIST_WALL),
        rng.uniform(MIN_DIST_WALL, room_dim[2] - MIN_DIST_WALL)
    ])
    mic = np.array([
        rng.uniform(MIN_DIST_WALL, room_dim[0] - MIN_DIST_WALL),
        rng.uniform(MIN_DIST_WALL, room_dim[1] - MIN_DIST_WALL),
        rng.uniform(MIN_DIST_WALL, room_dim[2] - MIN_DIST_WALL)
    ])

    room.add_source(src)
    room.add_microphone_array(mic[:, None])  # (3,1)
    room.image_source_model()                # build RIRs once
    return room


# ----------------------------------------------------------------------------- #
#  CLI + data preparation
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gcommands", type=str,
                        default="/mlspeech/datasets/public/SpeechCommand/base/speech_commands_v0.02")
    parser.add_argument("--target", type=str,
                        default="/mlspeech/data/gilad/google_commands_reverb")
    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    #  Collect wav files
    # --------------------------------------------------------------------- #
    print("Collecting Speech-Commands files …")
    folders = [
        os.path.join(args.gcommands, d)
        for d in os.listdir(args.gcommands)
        if d in TARGET_COMMANDS and os.path.isdir(os.path.join(args.gcommands, d))
    ]

    all_wavs = []
    for f in folders:
        all_wavs.extend(glob(os.path.join(f, "*.wav")))

    train, tmp  = train_test_split(all_wavs, test_size=0.20, random_state=0)
    valid, test = train_test_split(tmp,      test_size=0.50, random_state=0)
    split_dict  = {"train": train, "valid": valid, "test": test}

    # --------------------------------------------------------------------- #
    #  Make output dirs
    # --------------------------------------------------------------------- #
    def mk_dirs(base):
        clean = Path(args.target) / base / "clean"
        reverb = Path(args.target) / base / "noisy"
        clean.mkdir(parents=True, exist_ok=True)
        reverb.mkdir(parents=True, exist_ok=True)
        return clean, reverb

    out_paths = {s: mk_dirs(s) for s in split_dict.keys()}

    # --------------------------------------------------------------------- #
    #  Main loop
    # --------------------------------------------------------------------- #
    print("Generating reverberant data …")
    room = None  # initialise outside the loop

    for split, file_list in split_dict.items():
        clean_dir, rev_dir = out_paths[split]
        print(f"   {split.upper():5}  ({len(file_list)} files)")
        for i, wav in enumerate(tqdm(file_list)):
            # ---------------------------------------------------------------- #
            #  Draw a new room once every NB_SAMPLES_PER_ROOM clips
            # ---------------------------------------------------------------- #
            if i % NB_SAMPLES_PER_ROOM == 0:
                room = make_random_room()

            # ---------------------------------------------------------------- #
            #  Load speech, convolve with room impulse response
            # ---------------------------------------------------------------- #
            s, _ = load(wav, sr=SR)
            room.sources[0].signal = s
            room.simulate()               # fast © pyroomacoustics
            y = room.mic_array.signals[0]  # (samples,)

            # Trim/pad so that y has the same length as s
            if len(y) < len(s):
                y = np.pad(y, (0, len(s) - len(y)))
            else:
                y = y[:len(s)]

            # ---------------------------------------------------------------- #
            #  Write to disk
            # ---------------------------------------------------------------- #
            label = os.path.basename(os.path.dirname(wav))
            fname = f"{label}_{os.path.basename(wav)}"

            sf_kwargs = dict(samplerate=SR, subtype="PCM_16")
            sf.write(clean_dir / fname,  s, **sf_kwargs)
            sf.write(rev_dir   / fname,  y, **sf_kwargs)

    print("Done – reverberant dataset saved to:", args.target)
