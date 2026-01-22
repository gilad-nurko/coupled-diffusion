import sys
import csv
import json
import torch
import numpy as np
import pyloudnorm as pyln

from glob import glob
from os import listdir, makedirs
from os.path import join, isdir, exists
from argparse import ArgumentParser
from soundfile import read, write
from tqdm import tqdm
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample

# =====================================================================
# Saving helpers
# =====================================================================
# Train/valid: full loudness + SNR metadata
def save_files_train_valid(
    target_dir,
    subset,
    speaker,
    id,
    speech_file,
    speech_start,
    speech_end,
    noise_file,
    noise_start,
    noise_end,
    mixture,
    speech,
    loudness_speech,
    loudness_noise,
    loudness_mixture,
    snr_dB,
    sr,
):
    base = speech_file.split("/")[-1][:-4]
    with open(join(target_dir, f"{subset}.csv"), "a") as text_file:
        text_file.write(
            f"{id:05},{speaker},{base},{speech_start},{speech_end},"
            f"{noise_file.split('/')[-1][:-4]},{noise_start},{noise_end},"
            f"{loudness_speech:.1f},{loudness_noise:.1f},{loudness_mixture:.1f},{snr_dB:.1f}\n"
        )
    write(
        join(target_dir, subset, "noisy", speaker, f"{id:05}_{snr_dB:.1f}dB_category_{base}.wav"),
        mixture,
        sr,
        subtype="FLOAT",
    )
    write(
        join(target_dir, subset, "clean", speaker, f"{id:05}_category_{base}.wav"),
        speech,
        sr,
        subtype="FLOAT",
    )
    id += 1
    return id


# Test: only SNR + noise positions (as in original EARS-WHAM)
def save_files_test(
    target_dir,
    speaker,
    id,
    speech_file,
    speech_start,
    speech_end,
    noise_file,
    noise_start,
    noise_end,
    mixture,
    speech,
    snr_dB,
    sr,
):
    base = speech_file.split("/")[-1][:-4]
    with open(join(target_dir, "test.csv"), "a") as text_file:
        text_file.write(
            f"{id:05},{speaker},{base},{speech_start},{speech_end},"
            f"{noise_file.split('/')[-1][:-4]},{noise_start},{noise_end},{snr_dB:.1f}\n"
        )
    write(
        join(target_dir, "test", "noisy", speaker, f"{id:05}_{snr_dB:.1f}dB_category_{base}.wav"),
        mixture,
        sr,
        subtype="FLOAT",
    )
    write(
        join(target_dir, "test", "clean", speaker, f"{id:05}_category_{base}.wav"),
        speech,
        sr,
        subtype="FLOAT",
    )
    id += 1
    return id


def find_emotion_style(speech_file, emotions_styles=[]):
    for emo_style in emotions_styles:
        if emo_style.lower() in speech_file.lower():
            return emo_style
    return None


# =====================================================================
# VAD-based non-speech gap finder (unchanged from your code)
# =====================================================================
def find_non_speech_boundaries(
    wav: np.ndarray,
    sr: int,
    min_silence_ms: int = 300,
    frame_ms: int = 30,
    hop_ms: int = 10,
    energy_rel_thresh: float = 0.5,
):
    """
    Returns: list of (start_sample, end_sample) non-speech intervals (inclusive start, exclusive end)
    Heuristic: tries webrtcvad if available; else energy-based VAD.
    """
    wav = wav.astype(np.float32)
    # mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    vad_mask = None
    try:
        import webrtcvad, struct

        vad = webrtcvad.Vad(2)  # 0-3, 2 is a good balance
        # WebRTC expects 16-bit PCM @16k. Resample if needed.
        if sr != 16000:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav16 = resampler(wav_t).squeeze(0).numpy()
            sr16 = 16000
        else:
            wav16 = wav
            sr16 = sr

        fm = 30  # ms
        frame_len = int(sr16 * fm / 1000)
        hop_len = frame_len  # non-overlap for VAD bytes
        pcm16 = np.clip(wav16, -1, 1)
        pcm16 = (pcm16 * 32767).astype(np.int16)

        def frame_bytes(x, start, length):
            return struct.pack("<%dh" % length, *x[start : start + length])

        voiced = []
        for s in range(0, len(pcm16) - frame_len + 1, hop_len):
            fbytes = frame_bytes(pcm16, s, frame_len)
            voiced.append(vad.is_speech(fbytes, sr16))
        voiced = np.array(voiced, dtype=bool)

        # Map back to original sr by labeling samples per frame
        mask = np.zeros(len(wav), dtype=bool)
        frame_len_orig = int(sr * fm / 1000)
        for i, v in enumerate(voiced):
            start = i * frame_len_orig
            end = min(start + frame_len_orig, len(mask))
            mask[start:end] = v
        vad_mask = mask

    except Exception:
        # Energy-based fallback
        fm = frame_ms
        hm = hop_ms
        frame_len = int(sr * fm / 1000)
        hop_len = int(sr * hm / 1000)
        if frame_len <= 0:
            frame_len = 1
        if hop_len <= 0:
            hop_len = 1

        n_frames = 1 + max(0, (len(wav) - frame_len) // hop_len)
        E = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_len
            seg = wav[s : s + frame_len]
            E[i] = np.mean(seg**2) if len(seg) else 0.0
        thr = energy_rel_thresh * (np.median(E) + 1e-8)
        speech_frames = E > thr

        # Smooth
        if len(speech_frames) >= 3:
            sm = speech_frames.astype(np.int32)
            speech_frames = (sm[:-2] + sm[1:-1] + sm[2:]) >= 2
            speech_frames = np.r_[speech_frames[0], speech_frames, speech_frames[-1]]

        vad_mask = np.zeros(len(wav), dtype=bool)
        for i, sp in enumerate(speech_frames):
            s = i * hop_len
            e = min(s + frame_len, len(vad_mask))
            vad_mask[s:e] = sp

    min_silence_samples = int(sr * (min_silence_ms / 1000.0))
    non = ~vad_mask
    intervals = []
    in_ns = False
    start = 0
    for i, v in enumerate(non):
        if v and not in_ns:
            in_ns = True
            start = i
        elif not v and in_ns:
            if i - start >= min_silence_samples:
                intervals.append((start, i))
            in_ns = False
    if in_ns:
        if len(non) - start >= min_silence_samples:
            intervals.append((start, len(non)))
    return intervals


# =====================================================================
# Main
# =====================================================================
def main(args):
    # Reproducibility
    np.random.seed(42)

    # Sampling rate
    if getattr(args, "16k") is True:
        target_sr = 16000
    else:
        target_sr = args.sr

    # Directories
    # speech_dir = join(args.data_dir, "EARS")
    speech_dir = join("/mlspeech/data/gilad/paper_ears_reverbed", "EARS")
    noise_dir = join(args.data_dir, "WHAM48kHz")
    if target_sr == 48000:
        target_dir = join(args.data_dir, "EARS-WHAM_v2_4sec_chunks")
    else:
        target_dir = join(args.data_dir, f"EARS-WHAM_v2_{target_sr//1000}k")

    assert isdir(speech_dir), f"The directory {speech_dir} does not exist"
    assert isdir(noise_dir), f"The directory {noise_dir} does not exist"

    if exists(target_dir):
        print(
            f"[Warning] Abort EARS-WHAM generation script. "
            f"The directory {target_dir} already exists."
        )
        sys.exit()
    else:
        makedirs(target_dir)

    # Speaker splits
    all_speakers = sorted(listdir(speech_dir))
    valid_speakers = ["p100", "p101"]
    test_speakers = ["p102", "p103", "p104", "p105", "p106", "p107"]

    speakers = {
        "train": [s for s in all_speakers if s not in valid_speakers + test_speakers],
        "valid": valid_speakers,
        "test": test_speakers,
    }

    # Hold-out styles + freeform exclusion
    hold_out_styles = ["interjection", "melodic", "nonverbal", "vegetative"]

    # WHAM noise metadata
    noise_splits = {"train": [], "valid": [], "test": []}
    with open(join(noise_dir, "high_res_wham", "high_res_metadata.csv"), mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            subset = row["WHAM! Split"].lower()  # "train", "valid", "test"
            filename = row["Filename"]
            if subset in noise_splits:
                noise_splits[subset].append(filename)

    # Emotions / styles for test SNR distribution
    emotions_styles = [
        "adoration",
        "amazement",
        "amusement",
        "anger",
        "confusion",
        "contentment",
        "cuteness",
        "desire",
        "disappointment",
        "disgust",
        "distress",
        "embarassment",
        "extasy",
        "fast",
        "fear",
        "guilt",
        "highpitch",
        "interest",
        "loud",
        "lowpitch",
        "neutral",
        "pain",
        "pride",
        "realization",
        "relief",
        "regular",
        "sadness",
        "serenity",
        "slow",
        "whisper",
    ]

    meter = pyln.Meter(48000)
    resample = Resample(48000, target_sr, dtype=torch.float64)

    # =================================================================
    # Train / valid
    # =================================================================
    for subset in ["train", "valid"]:
        print(f"Generate {subset} split")
        with open(join(target_dir, f"{subset}.csv"), "w") as text_file:
            text_file.write(
                "id,speaker,speech_file,speech_start,speech_end,noise_file,noise_start,"
                "noise_end,speech_dB,noise_dB,mixture_dB,snr_dB\n"
            )

        speech_files = []
        for speaker in speakers[subset]:
            speech_files += sorted(glob(join(speech_dir, speaker, "*.wav")))
            makedirs(join(target_dir, subset, "clean", speaker), exist_ok=True)
            makedirs(join(target_dir, subset, "noisy", speaker), exist_ok=True)

        # Remove hold-out styles + freeform
        speech_files = [
            sf
            for sf in speech_files
            if sf.split("/")[-1].split("_")[0] not in hold_out_styles
            and "freeform" not in sf[:-4].split("/")[-1].split("_")
        ]

        # WHAM noise files for this subset
        noise_files = [
            join(noise_dir, "high_res_wham", "audio", filename)
            for filename in noise_splits.get(subset, [])
        ]

        id = 0
        for speech_file in tqdm(speech_files):
            speech, sr = read(speech_file)
            speaker = speech_file.split("/")[-2]
            assert sr == 48000, "Script only works for speech files of 48kHz."

            # Min length check
            if len(speech) < args.min_length * args.sr:
                continue

            # High-pass speech
            speech = highpass_biquad(
                torch.from_numpy(speech), sample_rate=sr, cutoff_freq=args.cutoff_freq
            ).numpy()

            # -----------------------------
            # Sample noise + set SNR
            # -----------------------------
            noise = np.zeros((0, 0))
            while noise.shape[0] < speech.shape[0]:
                noise_file = np.random.choice(noise_files)
                noise, sr_noise = read(noise_file, always_2d=True)
            assert sr == sr_noise, "Sampling rates of speech and noise should match."

            channel = np.random.randint(0, noise.shape[1])
            noise = noise[:, channel]

            noise_start_global = np.random.randint(len(noise) - len(speech) + 1)
            noise_segment = noise[noise_start_global : noise_start_global + len(speech)]

            snr_dB = np.round(
                np.random.uniform(args.min_snr, args.max_snr), decimals=1
            )
            loudness_speech_full = meter.integrated_loudness(speech)
            loudness_noise_full = meter.integrated_loudness(noise_segment)
            target_loudness = loudness_speech_full - snr_dB
            delta_loudness = target_loudness - loudness_noise_full
            gain = np.power(10.0, delta_loudness / 20.0)
            noise_scaled_full = gain * noise_segment
            mixture_full = speech + noise_scaled_full

            # Fix clipping by increasing SNR (less noise)
            while np.max(np.abs(mixture_full)) >= 1.0:
                snr_dB = snr_dB + 1
                target_loudness = loudness_speech_full - snr_dB
                delta_loudness = target_loudness - loudness_noise_full
                gain = np.power(10.0, delta_loudness / 20.0)
                noise_scaled_full = gain * noise_segment
                mixture_full = speech + noise_scaled_full

            # -----------------------------
            # Chunking with VAD-based gaps
            # -----------------------------
            if len(mixture_full) >= int(args.cut_length * args.sr):
                long_mixture = mixture_full
                long_speech = speech

                min_silence_ms = getattr(args, "min_silence_ms", 150)
                non_speech_gaps = find_non_speech_boundaries(
                    long_speech, sr=args.sr, min_silence_ms=min_silence_ms
                )
                pause_edges = sorted({p for a, b in non_speech_gaps for p in (a, b)})
                N = len(long_speech)
                cut_len = int(args.cut_length * args.sr)
                min_len = int(args.min_cut_length * args.sr)
                max_len = int(args.max_cut_length * args.sr)
                if not pause_edges or pause_edges[-1] != N:
                    pause_edges.append(N)

                import bisect

                def pick_best_end(start_idx_samples: int):
                    i = bisect.bisect_right(pause_edges, start_idx_samples)
                    if i >= len(pause_edges):
                        return None
                    lo = start_idx_samples + min_len
                    hi = start_idx_samples + max_len
                    j_lo = bisect.bisect_left(pause_edges, lo, i)
                    j_hi = bisect.bisect_right(pause_edges, hi, i)
                    if j_lo >= j_hi:
                        return None
                    best_end = None
                    best_err = None
                    target_end = start_idx_samples + cut_len
                    for k in range(j_lo, j_hi):
                        e = pause_edges[k]
                        err = abs(e - target_end)
                        if (
                            best_err is None
                            or err < best_err
                            or (err == best_err and e <= target_end)
                        ):
                            best_err = err
                            best_end = e
                    return best_end

                last_end = 0
                pad_sec = 0.15
                pad = int(pad_sec * args.sr)

                while True:
                    remaining = N - last_end
                    if remaining < min_len:
                        break

                    end = pick_best_end(last_end)
                    if end is None:
                        after_max = last_end + max_len
                        idx = bisect.bisect_right(pause_edges, after_max)
                        if idx >= len(pause_edges):
                            break
                        last_end = pause_edges[idx]
                        continue

                    seg_len = end - last_end
                    if seg_len < min_len:
                        last_end = end
                        continue
                    if seg_len > max_len:
                        last_end = end
                        continue

                    start = last_end
                    stop = end

                    mix_chunk = long_mixture[start:stop]
                    speech_chunk = long_speech[start:stop]
                    noise_chunk = mix_chunk - speech_chunk

                    loudness_speech = meter.integrated_loudness(speech_chunk)
                    loudness_noise = meter.integrated_loudness(noise_chunk)
                    loudness_mixture = meter.integrated_loudness(mix_chunk)

                    # Resample to target_sr if needed
                    if sr != target_sr:
                        mix_chunk = resample(torch.from_numpy(mix_chunk)).numpy()
                        speech_chunk = resample(torch.from_numpy(speech_chunk)).numpy()

                    if ("whisper" in speech_file) or (loudness_speech > args.min_dB):
                        ns = noise_start_global + start
                        ne = noise_start_global + stop
                        id = save_files_train_valid(
                            target_dir,
                            subset,
                            speaker,
                            id,
                            speech_file,
                            start,
                            stop,
                            noise_file,
                            ns,
                            ne,
                            mix_chunk,
                            speech_chunk,
                            loudness_speech,
                            loudness_noise,
                            loudness_mixture,
                            snr_dB,
                            target_sr,
                        )

                    last_end = end + pad

            else:
                speech_start = 0
                speech_end = -1
                noise_chunk = mixture_full - speech
                loudness_speech = meter.integrated_loudness(speech)
                loudness_noise = meter.integrated_loudness(noise_chunk)
                loudness_mixture = meter.integrated_loudness(mixture_full)

                if sr != target_sr:
                    mixture_full_rs = resample(torch.from_numpy(mixture_full)).numpy()
                    speech_rs = resample(torch.from_numpy(speech)).numpy()
                else:
                    mixture_full_rs = mixture_full
                    speech_rs = speech

                if ("whisper" in speech_file) or (loudness_speech > args.min_dB):
                    ns = noise_start_global
                    ne = noise_start_global + len(mixture_full)
                    id = save_files_train_valid(
                        target_dir,
                        subset,
                        speaker,
                        id,
                        speech_file,
                        speech_start,
                        speech_end,
                        noise_file,
                        ns,
                        ne,
                        mixture_full_rs,
                        speech_rs,
                        loudness_speech,
                        loudness_noise,
                        loudness_mixture,
                        snr_dB,
                        target_sr,
                    )

    # =================================================================
    # Test split
    # =================================================================
    # Ramps
    ramp_duration = args.ramp_time_in_ms / 1000
    ramp_samples = int(ramp_duration * 48000)
    ramp = np.linspace(0, 1, ramp_samples)

    print("Generate test split")
    with open("test_files.json", "r") as json_file:
        data = json.load(json_file)

    with open(join(target_dir, "test.csv"), "w") as text_file:
        text_file.write(
            "id,speaker,speech_file,speech_start,speech_end,"
            "noise_file,noise_start,noise_end,snr_dB\n"
        )

    test_speakers = list(data.keys())

    test_files = []
    for speaker in test_speakers:
        makedirs(join(target_dir, "test", "clean", speaker), exist_ok=True)
        makedirs(join(target_dir, "test", "noisy", speaker), exist_ok=True)
        # Filter out freeform
        speech_files = [k for k in data[speaker].keys() if "freeform" not in k]
        for sf in speech_files:
            test_files.append(join(speech_dir, speaker, sf + ".wav"))

    # WHAM test noise files
    noise_files_test = [
        join(noise_dir, "high_res_wham", "audio", filename)
        for filename in noise_splits.get("test", [])
    ]

    # Ensure uniform SNR coverage per emotion/style
    number_of_files_per_emotion = 12
    snr_bins = np.linspace(args.min_snr, args.max_snr, number_of_files_per_emotion + 1)
    counter_emotion_style = {x: 0 for x in emotions_styles}

    np.random.seed(42)
    np.random.shuffle(test_files)

    id = 0
    for test_file in tqdm(test_files):
        speaker = test_file.split("/")[-2]
        speech_file = test_file.split("/")[-1][:-4]

        speech, sr = read(join(speech_dir, speaker, speech_file + ".wav"))
        assert sr == 48000, "Script only works for speech files of 48kHz."

        # High-pass
        speech = highpass_biquad(
            torch.from_numpy(speech), sample_rate=sr, cutoff_freq=args.cutoff_freq
        ).numpy()

        cutting_times = data[speaker][speech_file]

        for cutting_time in cutting_times:
            start = cutting_time[0]
            end = cutting_time[1]
            speech_cut = speech[start:end]

            if len(speech_cut) > args.max_time_test_set_in_s * args.sr:
                continue

            # Ensure noise long enough
            noise, sr_noise = read(
                np.random.choice(noise_files_test), always_2d=True
            )
            while noise.shape[0] < speech_cut.shape[0]:
                noise, sr_noise = read(
                    np.random.choice(noise_files_test), always_2d=True
                )
            assert sr == sr_noise, "Sampling rates of speech and noise should match."

            channel = np.random.randint(0, noise.shape[1])
            noise = noise[:, channel]

            noise_start = np.random.randint(len(noise) - len(speech_cut) + 1)
            noise_cut = noise[noise_start : noise_start + len(speech_cut)]

            # SNR sampling with emotion-wise bins
            emo_style = find_emotion_style(speech_file, emotions_styles)
            if emo_style is not None:
                idx = counter_emotion_style[emo_style] % number_of_files_per_emotion
                min_snr_bin = snr_bins[idx]
                max_snr_bin = snr_bins[idx + 1]
                counter_emotion_style[emo_style] += 1
                snr_dB = np.round(
                    np.random.uniform(min_snr_bin, max_snr_bin), decimals=1
                )
            else:
                snr_dB = np.round(
                    np.random.uniform(args.min_snr, args.max_snr), decimals=1
                )

            loudness_speech_cut = meter.integrated_loudness(speech_cut)
            loudness_noise_cut = meter.integrated_loudness(noise_cut)
            target_loudness = loudness_speech_cut - snr_dB
            delta_loudness = target_loudness - loudness_noise_cut
            gain = np.power(10.0, delta_loudness / 20.0)
            noise_scaled = gain * noise_cut
            mixture = speech_cut + noise_scaled

            # Fix clipping
            while np.max(np.abs(mixture)) >= 1.0:
                snr_dB = snr_dB + 1
                target_loudness = loudness_speech_cut - snr_dB
                delta_loudness = target_loudness - loudness_noise_cut
                gain = np.power(10.0, delta_loudness / 20.0)
                noise_scaled = gain * noise_cut
                mixture = speech_cut + noise_scaled

            # Apply ramps
            mixture[:ramp_samples] = mixture[:ramp_samples] * ramp
            mixture[-ramp_samples:] = mixture[-ramp_samples:] * ramp[::-1]
            speech_cut[:ramp_samples] = speech_cut[:ramp_samples] * ramp
            speech_cut[-ramp_samples:] = speech_cut[-ramp_samples:] * ramp[::-1]

            # Resample to target_sr if needed
            if sr != target_sr:
                mixture_rs = resample(torch.from_numpy(mixture)).numpy()
                speech_cut_rs = resample(torch.from_numpy(speech_cut)).numpy()
            else:
                mixture_rs = mixture
                speech_cut_rs = speech_cut

            noise_end = noise_start + len(mixture)  # before resampling
            id = save_files_test(
                target_dir,
                speaker,
                id,
                test_file,
                start,
                end,
                noise_file,
                noise_start,
                noise_end,
                mixture_rs,
                speech_cut_rs,
                snr_dB,
                target_sr,
            )


if __name__ == "__main__":
    """
    Usage:
        python generate_ears_wham_chunks.py --data_dir /data3/databases
    """
parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory which should contain subdirectories EARS and WHAM48kHz")
parser.add_argument("--min_length", type=float, default=3.0, help="Minimum length of speech files in seconds")
parser.add_argument("--cut_length", type=float, default=4.0, help="Target cut length (around this) in seconds")
parser.add_argument("--min_cut_length", type=float, default=2.0, help="Minimum cut length in seconds")
parser.add_argument("--max_cut_length", type=float, default=6.0, help="Maximum cut length in seconds")
parser.add_argument("--min_silence_ms", type=float, default=50, help="Minimum silence length in ms to be considered a gap")
parser.add_argument("--cutoff_freq", type=float, default=75.0, help="Cutoff frequency for Hi-Pass filter")
parser.add_argument("--min_dB", type=float, default=-55.0, help="Minimum loudness threshold for clean speech")
parser.add_argument("--sr", type=int, default=48000, help="Sampling rate of input EARS speech")
parser.add_argument("--16k", action="store_true", help="Set output sampling rate to 16kHz")
parser.add_argument("--ramp_time_in_ms", type=int, default=10, help="Ramp time in ms (apply to test mixtures)")
parser.add_argument("--min_snr", type=float, default=-2.5, help="Minimum SNR in dB for mixtures")
parser.add_argument("--max_snr", type=float, default=17.5, help="Maximum SNR in dB for mixtures")
parser.add_argument("--max_time_test_set_in_s", type=int, default=29, help="Maximum time in seconds for the test segments")
args = parser.parse_args()

main(args)
