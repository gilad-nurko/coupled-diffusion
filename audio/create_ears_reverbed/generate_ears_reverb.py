import sys
import sofa
import mat73
import torch
import numpy as np
import pyloudnorm as pyln

import json
from glob import glob
from os import listdir, makedirs
from os.path import join, isdir, exists
from argparse import ArgumentParser
from soundfile import read, write
from tqdm import tqdm
from scipy.signal import convolve
from scipy import stats
from librosa import resample as resample_librosa
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample

# TODO:check how speech_file looks like and add the speech type to the name of the audios
def save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, rir_file, channel, 
               gain, rt60, mixture, speech, data_dir, sr):
    with open(join(target_dir, f"{subset}.csv"), "a") as text_file:
        text_file.write(f"{id:05},{speaker},{speech_file.split('/')[-1][:-4]},{speech_start},{speech_end},"
            + f"{rir_file.replace(data_dir, '')},{channel},{gain},{rt60:.2f}\n")
    write(join(target_dir, subset, "reverberant", speaker, f"{id:05}_{rt60:.2f}_category_{speech_file.split('/')[-1][:-4]}.wav"), mixture, sr, subtype="FLOAT")
    write(join(target_dir, subset, "clean", speaker, f"{id:05}_category_{speech_file.split('/')[-1][:-4]}.wav"), speech, sr, subtype="FLOAT")
    id += 1
    return id

def calc_rt60(h, sr=48000, rt='t30'): 
    """
    RT60 measurement routine acording to Schroeder's method [1].

    [1] M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.

    Adapted from https://github.com/python-acoustics/python-acoustics/blob/99d79206159b822ea2f4e9d27c8b2fbfeb704d38/acoustics/room.py#L156
    """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    h_abs = np.abs(h) / np.max(np.abs(h))

    # Schroeder integration
    sch = np.cumsum(h_abs[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch)+1e-20)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / sr
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60

def find_non_speech_boundaries(
    wav: np.ndarray, sr: int, 
    min_silence_ms: int = 300, frame_ms: int = 30, hop_ms: int = 10,
    energy_rel_thresh: float = 0.5
):
    """
    Returns: list of (start_sample, end_sample) non-speech intervals (inclusive start, exclusive end)
    Heuristic: tries webrtcvad if available; else energy-based VAD.
    """
    wav = wav.astype(np.float32)
    # mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    # --- Try WebRTC VAD first ---
    vad_mask = None
    try:
        import webrtcvad, struct
        vad = webrtcvad.Vad(2)  # 0-3, 2 is a good balance
        # WebRTC expects 16-bit PCM @16k. Resample if needed.
        import math
        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav16 = resampler(wav_t).squeeze(0).numpy()
            sr16 = 16000
        else:
            wav16 = wav
            sr16 = sr

        # Frame to 10/20/30 ms as bytes
        fm = 30  # ms
        frame_len = int(sr16 * fm / 1000)
        hop_len = frame_len  # non-overlap for VAD bytes
        pcm16 = np.clip(wav16, -1, 1)
        pcm16 = (pcm16 * 32767).astype(np.int16)

        def frame_bytes(x, start, length):
            return struct.pack("<%dh" % length, *x[start:start+length])

        voiced = []
        for s in range(0, len(pcm16) - frame_len + 1, hop_len):
            fbytes = frame_bytes(pcm16, s, frame_len)
            voiced.append(vad.is_speech(fbytes, sr16))
        voiced = np.array(voiced, dtype=bool)

        # Map back to original sr by labeling samples per frame
        mask = np.zeros(len(wav), dtype=bool)
        # scale factor from 16k frames back to original sr
        # approximate: assign per frame a span at original sr
        frame_len_orig = int(sr * fm / 1000)
        idx = 0
        for i, v in enumerate(voiced):
            start = i * frame_len_orig
            end = min(start + frame_len_orig, len(mask))
            mask[start:end] = v
        vad_mask = mask

    except Exception:
        # --- Energy-based fallback ---
        fm = frame_ms
        hm = hop_ms
        frame_len = int(sr * fm / 1000)
        hop_len  = int(sr * hm / 1000)
        if frame_len <= 0: frame_len = 1
        if hop_len <= 0: hop_len = 1

        n_frames = 1 + max(0, (len(wav) - frame_len) // hop_len)
        E = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_len
            seg = wav[s:s+frame_len]
            E[i] = np.mean(seg**2) if len(seg) else 0.0
        thr = energy_rel_thresh * (np.median(E) + 1e-8)
        speech_frames = E > thr

        # Smooth with small hysteresis: open-close
        # (simple 3-frame median)
        if len(speech_frames) >= 3:
            sm = speech_frames.astype(np.int32)
            speech_frames = (sm[:-2] + sm[1:-1] + sm[2:]) >= 2
            # pad back
            speech_frames = np.r_[speech_frames[0], speech_frames, speech_frames[-1]]
        # Expand frame decisions back to samples
        vad_mask = np.zeros(len(wav), dtype=bool)
        for i, sp in enumerate(speech_frames):
            s = i * hop_len
            e = min(s + frame_len, len(vad_mask))
            vad_mask[s:e] = sp

    # Build non-speech intervals with minimum duration
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

def main(args):
    # Reproducibility
    np.random.seed(42)
    counter = 0
    # Set sampling rate
    if getattr(args, '16k') == True:
        target_sr = 16000
    else:
        target_sr = args.sr

    # Organize directories
    speech_dir = join(args.data_dir, "EARS")
    if target_sr == 48000:
        target_dir = join(args.data_dir, "EARS-Reverb_v2_4sec_chunks")
    else:
        target_dir = join(args.data_dir, f"EARS-Reverb_v2_{target_sr//1000}k")

    assert isdir(speech_dir), f"The directory {speech_dir} does not exist"

    if exists(target_dir):
        print(f"[Warning] Abort EARS-Reverb generation script. The directory {join(args.data_dir, target_dir)} already exists.")
        sys.exit()
    else:
        makedirs(target_dir)

    all_speakers = sorted(listdir(speech_dir))
    # Define training split
    valid_speakers = ["p100", "p101"] 
    test_speakers = ["p102", "p103", "p104", "p105", "p106", "p107"]
    
    speakers = {
        "train": [s for s in all_speakers if s not in valid_speakers + test_speakers],
        "valid": valid_speakers, 
        "test": test_speakers
        }
    
    # Hold out speaking styles 
    hold_out_styles = ["interjection", "melodic", "nonverbal", "vegetative"]

    # Splits for RIRs
    rir_files = {
        "train": [],
        "valid": [],
        "test": [],
    }

    # ACE-Challenge dataset
    dir = join(args.data_dir, "ACE-Challenge")
    names = ["Chromebook", "Crucif", "EM32", "Lin8Ch", "Mobile", "Single"]
    for name in names:
        rir_files["test"] += sorted(glob(join(dir, f"ACE_Corpus_RIRN_{name}", "**", "*RIR.wav"), recursive=True))
      
    # AIR dataset
    dir = join(args.data_dir, "AIR", "AIR_1_4", "AIR_wav_files")
    rir_files["valid"] += sorted(glob(join(dir, "*.wav")))

    # ARNI dataset
    dir = join(args.data_dir, "ARNI")
    all_arni_files = sorted(glob(join(dir, "**", "*.wav"), recursive=True))
    # remove file numClosed_26-35/IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav because it is corrupted
    all_arni_files = [file for file in all_arni_files if "numClosed_26-35/IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav" not in file]
    rir_files["train"] += sorted(list(np.random.choice(all_arni_files, size=1000, replace=False))) # take 1000 of 132037 RIRs

    # BRUDEX dataset
    dir = join(args.data_dir, "BRUDEX")
    rir_files["train"] += sorted(glob(join(dir, "rir", "**", "*.mat"), recursive=True))

    # dEchorate dataset
    dir = join(args.data_dir, "dEchorate", "sofa")
    rir_files["train"] += sorted(glob(join(dir, "**", "*.sofa"), recursive=True))

    # DetmoldSRIR dataset
    dir = join(args.data_dir, "DetmoldSRIR")
    rir_files["train"] += sorted(glob(join(dir, "SetA_SingleSources", "Data", "**", "*.wav"), recursive=True))

    # Palimpsest dataset
    dir = join(args.data_dir, "Palimpsest")
    rir_files["train"] += sorted(glob(join(dir, "**", "*.wav"), recursive=True))

    meter = pyln.Meter(48000)
    resample = Resample(48000, target_sr, dtype=torch.float64)

    # Select speech files for split
    for subset in ["train", "valid"]:
        print(f"Generate {subset} split")
        with open(join(target_dir, f"{subset}.csv"), "w") as text_file:
            text_file.write(f"id,speaker,speech_file,speech_start,speech_end,rir_file,channel,gain,rt60\n")
        speech_files = []
        for speaker in speakers[subset]:  
            speech_files += sorted(glob(join(speech_dir, speaker, "*.wav")))
            makedirs(join(target_dir, subset, "clean", speaker))  
            makedirs(join(target_dir, subset, "reverberant", speaker))  
        # TODO: check how speech_file.split("/")[-1].split("_") looks like and write this line right "freeform" not in speech_file.split("/")[-1].split("_")
        # Remove files of hold out styles, and of freeform speaking style
        speech_files = [speech_file for speech_file in speech_files if speech_file.split("/")[-1].split("_")[0] not in hold_out_styles and "freeform" not in speech_file[:-4].split("/")[-1].split("_")]
        id = 0
        for speech_file in tqdm(speech_files):
            speech, sr = read(speech_file)
            speaker = speech_file.split("/")[-2]
            assert sr == 48000, "Script only works for speech files of 48kHz."

            # Only take speech files that are longer than min_length
            if len(speech) < args.min_length*args.sr:
                continue

            # Fitler speech signal with Hi-Pass filter 
            speech = highpass_biquad(torch.from_numpy(speech), sample_rate=sr, cutoff_freq=args.cutoff_freq).numpy()

            # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples
            rt60 = np.inf
            while rt60 > args.max_rt60:
                rir_file = np.random.choice(rir_files[subset])

                if "ARNI" in rir_file:
                    rir, sr_rir = read(rir_file, always_2d=True)
                    # Take random channel if file is multi-channel
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                    assert sr_rir == 44100, f"Sampling rate of {rir_file} is {sr}"
                    rir = resample_librosa(rir, orig_sr=sr_rir, target_sr=48000)
                    sr_rir = 48000
                elif rir_file.endswith(".wav"):
                    rir, sr_rir = read(rir_file, always_2d=True)
                    # Take random channel if file is multi-channel
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                elif rir_file.endswith(".sofa"):
                    hrtf = sofa.Database.open(rir_file)
                    rir = hrtf.Data.IR.get_values()
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[0,channel,:]
                    sr_rir = hrtf.Data.SamplingRate.get_values().item()
                elif rir_file.endswith(".mat"):
                    rir = mat73.loadmat(rir_file)
                    sr_rir = rir["fs"].item()
                    rir = rir["data"]
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                else:
                    raise ValueError(f"Unknown file format: {rir_file}")

                assert sr_rir == sr, f"Sampling rate of {rir_file} is {sr_rir} Hz and not 48000 Hz"

                # Cut RIR to get direct path at the beginning
                max_index = np.argmax(np.abs(rir))
                rir = rir[max_index:]

                # Normalize RIRs in range [0.1, 0.7] for numerical stability
                if np.max(np.abs(rir)) < 0.1:
                    rir = 0.1 * rir / np.max(np.abs(rir))
                elif np.max(np.abs(rir)) > 0.7:
                    rir = 0.7 * rir / np.max(np.abs(rir))

                rt60 = calc_rt60(rir, sr=sr)

                mixture = convolve(speech, rir)[:len(speech)]

                # normalize mixture
                loudness_speech = meter.integrated_loudness(speech)
                loudness_mixture = meter.integrated_loudness(mixture)
                delta_loudness = loudness_speech - loudness_mixture
                gain = np.power(10.0, delta_loudness/20.0)
                # if gain is inf sample again
                if np.isinf(gain):
                    rt60 = np.inf
                mixture = gain * mixture

            if np.max(np.abs(mixture)) > 1.0:
                mixture = mixture / np.max(np.abs(mixture))

            # Cut long files into pieces
            if len(mixture) >= int((args.cut_length)*args.sr):
                long_mixture = mixture
                long_speech = speech
                min_silence_ms = getattr(args, "min_silence_ms", 150)
                non_speech_gaps = find_non_speech_boundaries(long_speech, sr=args.sr, min_silence_ms=min_silence_ms)
                pause_edges = sorted({p for a,b in non_speech_gaps for p in (a,b)})
                N = len(long_speech)
                cut_len = int(args.cut_length * args.sr)
                min_len = int(args.min_cut_length * args.sr)
                max_len = int(args.max_cut_length * args.sr)
                if not pause_edges or pause_edges[-1] != N:
                    pause_edges.append(N)
                
                import bisect
                def pick_best_end(start_idx_samples: int) -> int | None:
                    # Find edges strictly after the start
                    i = bisect.bisect_right(pause_edges, start_idx_samples)
                    if i >= len(pause_edges):
                        return None

                    # Gather all edges within the allowed window [min_len, max_len]
                    lo = start_idx_samples + min_len
                    hi = start_idx_samples + max_len
                    j_lo = bisect.bisect_left(pause_edges, lo, i)     # first >= lo
                    j_hi = bisect.bisect_right(pause_edges, hi, i)    # first >  hi

                    if j_lo >= j_hi:
                        # No end within [min,max] → nothing acceptable (would be >6s or <min)
                        return None

                    # Among candidates in [j_lo, j_hi), choose end minimizing |(end-start) - cut_len|
                    best_end = None
                    best_err = None
                    target_end = start_idx_samples + cut_len

                    # Prefer the “closest to 4s” with a tie-break favoring the *below* option
                    for k in range(j_lo, j_hi):
                        e = pause_edges[k]
                        err = abs(e - target_end)
                        if (best_err is None) or (err < best_err) or (err == best_err and e <= target_end):
                            best_err = err
                            best_end = e

                    return best_end
                
                last_end = 0
                pad_sec = 0.15  # seconds of silence padding
                pad = int(pad_sec * args.sr)

                while True:
                    remaining = N - last_end
                    if remaining < min_len:
                        break

                    end = pick_best_end(last_end)

                    if end is None:
                        # There is no legal VAD edge within [min, 6s] → skip the long region:
                        # Advance to the first edge after last_end+max_len (non-overlapping)
                        after_max = last_end + max_len
                        idx = bisect.bisect_right(pause_edges, after_max)
                        if idx >= len(pause_edges):
                            break  # nothing more to do
                        last_end = pause_edges[idx]  # drop the too-long chunk
                        continue

                    seg_len = end - last_end
                    # Guard (should already be true by construction)
                    if seg_len < min_len:
                        # Too short: push start forward to this edge to avoid overlap and continue
                        last_end = end
                        continue
                    if seg_len > max_len:
                        # Too long: explicitly skip
                        last_end = end
                        continue

                    # ====== (Optional) padding — simplest is to skip or zero-pad; here we skip for purity ======
                    start = last_end
                    stop  = end

                    mix_chunk   = long_mixture[start:stop]
                    speech_chunk= long_speech[start:stop]

                    # --- Resample if needed ---
                    if args.sr != target_sr:
                        mix_chunk   = resample(torch.from_numpy(mix_chunk)).numpy()
                        speech_chunk= resample(torch.from_numpy(speech_chunk)).numpy()

                    # Loudness gating (keep your existing rule)
                    loudness_speech = meter.integrated_loudness(speech_chunk)
                    if ("whisper" in speech_file) or (loudness_speech > args.min_dB):
                        id = save_files(
                            target_dir, subset, speaker, id, speech_file,
                            start, stop, rir_file,
                            channel, gain, rt60, mix_chunk, speech_chunk, args.data_dir, target_sr
                        )

                    # Non-overlapping advance
                    last_end = end + pad  # skip the padding region

            else:
                speech_start = 0
                speech_end = -1

                # Measure loudness of the speech
                loudness_speech = meter.integrated_loudness(speech)

                # Resample to target sampling rate
                if sr != target_sr:
                    mixture = resample(torch.from_numpy(mixture)).numpy()
                    speech = resample(torch.from_numpy(speech)).numpy()

                # Save file if it contains whisper or min_dB speech loundness
                if "whisper" in speech_file or loudness_speech > args.min_dB:
                    id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, rir_file, 
                        channel, gain, rt60, mixture, speech, args.data_dir, target_sr)
    
    # ramps at beginning and end
    ramp_duration = args.ramp_time_in_ms / 1000
    ramp_samples = int(ramp_duration * 48000)
    ramp = np.linspace(0, 1, ramp_samples)
                
    print("Generate test split")
    with open("test_files.json", "r") as json_file:
        data = json.load(json_file)

    with open(join(target_dir, f"test.csv"), "w") as text_file:
        text_file.write(f"id,speaker,speech_file,speech_start,speech_end,rir_file,channel,gain,rt60\n")

    test_speakers = list(data.keys())

    test_files = []
    for speaker in test_speakers:
        makedirs(join(target_dir, "test", "clean", speaker))  
        makedirs(join(target_dir, "test", "reverberant", speaker))  
        # speech_files = list(data[speaker].keys())
        speech_files = [k for k in data[speaker].keys() if "freeform" not in k]
        for speech_file in speech_files:
            test_files.append(join(speech_dir, speaker, speech_file + ".wav"))

    # Reproducibility
    np.random.seed(42)
    np.random.shuffle(test_files)

    id = 0
    for test_file in tqdm(test_files):
        speaker = test_file.split("/")[-2]
        speech_file = test_file.split("/")[-1][:-4]

        speech, sr = read(join(speech_dir, speaker, speech_file + ".wav"))
        assert sr == 48000, "Script only works for speech files of 48kHz."

        # Fitler speech signal with Hi-Pass filter 
        speech = highpass_biquad(torch.from_numpy(speech), sample_rate=sr, cutoff_freq=args.cutoff_freq).numpy()

        cutting_times = data[speaker][speech_file]

        for cutting_time in cutting_times:
            start = cutting_time[0]
            end = cutting_time[1]
            speech_cut = speech[start:end]

            # Only take speech files that not longer than max_time_test_set_in_s
            if len(speech_cut) > args.max_time_test_set_in_s*args.sr:
                continue
            
            # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples
            rt60 = np.inf
            while rt60 > args.max_rt60:
                rir_file = np.random.choice(rir_files["test"])

                if "ARNI" in rir_file:
                    rir, sr_rir = read(rir_file, always_2d=True)
                    # Take random channel if file is multi-channel
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                    assert sr_rir == 44100, f"Sampling rate of {rir_file} is {sr}"
                    rir = resample(rir, orig_sr=sr_rir, target_sr=48000)
                    sr_rir = 48000
                elif rir_file.endswith(".wav"):
                    rir, sr_rir = read(rir_file, always_2d=True)
                    # Take random channel if file is multi-channel
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                elif rir_file.endswith(".sofa"):
                    hrtf = sofa.Database.open(rir_file)
                    rir = hrtf.Data.IR.get_values()
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[0,channel,:]
                    sr_rir = hrtf.Data.SamplingRate.get_values().item()
                elif rir_file.endswith(".mat"):
                    rir = mat73.loadmat(rir_file)
                    sr_rir = rir["fs"].item()
                    rir = rir["data"]
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                else:
                    raise ValueError(f"Unknown file format: {rir_file}")

                assert sr_rir == sr, f"Sampling rate of {rir_file} is {sr_rir} Hz and not 48000 Hz"

                # Cut RIR to get direct path at the beginning
                max_index = np.argmax(np.abs(rir))
                rir = rir[max_index:]

                # Normalize RIRs in range [0.1, 0.7] for numerical stability
                if np.max(np.abs(rir)) < 0.1:
                    rir = 0.1 * rir / np.max(np.abs(rir))
                elif np.max(np.abs(rir)) > 0.7:
                    rir = 0.7 * rir / np.max(np.abs(rir))

                rt60 = calc_rt60(rir, sr=sr)

                mixture = convolve(speech_cut, rir)[:len(speech_cut)]
            
                # normalize mixture
                loudness_speech = meter.integrated_loudness(speech_cut)
                loudness_mixture = meter.integrated_loudness(mixture)
                delta_loudness = loudness_speech - loudness_mixture
                gain = np.power(10.0, delta_loudness/20.0)
                mixture = gain * mixture
                # if gain is inf sample again 
                if np.isinf(gain):
                    rt60 = np.inf

            if np.max(np.abs(mixture)) > 1.0:
                mixture = mixture / np.max(np.abs(mixture))

            # Apply ramps
            mixture[:ramp_samples] = mixture[:ramp_samples] * ramp
            mixture[-ramp_samples:] = mixture[-ramp_samples:] * ramp[::-1]
            speech_cut[:ramp_samples] = speech_cut[:ramp_samples] * ramp
            speech_cut[-ramp_samples:] = speech_cut[-ramp_samples:] * ramp[::-1]

            # Resample to target sampling rate
            if sr != target_sr:
                mixture = resample(torch.from_numpy(mixture)).numpy()
                speech_cut = resample(torch.from_numpy(speech_cut)).numpy()

            id = save_files(target_dir, "test", speaker, id, test_file, start, end, rir_file, channel, gain, rt60, 
                mixture, speech_cut, args.data_dir, target_sr)


if __name__ == '__main__':
    '''
    Usage:

    python generate_ears_reverb.py --data_dir /data3/databases

    '''

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='Path to data directory which should contain subdirectories EARS and WHAM!48kHz')
    parser.add_argument("--min_length", type=float, default=3.0, help='Minimum length of speech files in seconds')
    parser.add_argument("--cut_length", type=float, default=4.0, help='Cut long files to this length in seconds')
    parser.add_argument("--min_cut_length", type=float, default=2.0, help='min cut length in seconds')
    parser.add_argument("--max_cut_length", type=float, default=6.0, help='min cut length in seconds')
    parser.add_argument("--min_silence_ms", type=float, default=50, help='min silence length in seconds')
    parser.add_argument("--cutoff_freq", type=float, default=75.0, help="Cutoff frequency for Hi-Pass filter")
    parser.add_argument("--min_dB", type=float, default=-55.0, help="Minimum loundness threshold for clean speech")
    parser.add_argument("--sr", type=int, default=48000, help='Sampling rate')
    parser.add_argument("--16k", action="store_true", help="Set sampling rate to 16kHz")
    parser.add_argument("--ramp_time_in_ms", type=int, default=10, help="Ramp time in ms")
    parser.add_argument("--max_rt60", type=float, default=2.0, help="Maximum RT60 in seconds")
    parser.add_argument("--max_time_test_set_in_s", type=int, default=29, help="Maximum time in seconds for the test set")
    args = parser.parse_args()

    main(args)
