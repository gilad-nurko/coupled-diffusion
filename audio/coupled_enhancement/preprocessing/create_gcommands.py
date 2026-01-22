import os
from glob import glob
from librosa import load
from pathlib import Path
import numpy as np
from soundfile import write
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

TARGET_COMMANDS = {"down", "go", "left", "no", "off",
                   "on", "right", "stop", "up", "yes"}
min_snr = 20
max_snr = 30
sr = 16000

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        "--gcommands",
        type=str,
        default="/mlspeech/datasets/public/SpeechCommand/base/speech_commands_v0.02",
        help="Path to Google Speech Commands v0.02 root directory"
    )

    parser.add_argument(
        "--white_noise",
        type=str,
        default="/mlspeech/datasets/public/SpeechCommand/base/speech_commands_v0.02/_background_noise_/white_noise.wav",
        help="Path to white noise .wav file"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="/mlspeech/data/gilad/google_commands_easiest_version",
        help="Target path to save clean and noisy files"
    )

    args = parser.parse_args()

    # Collect .wav files from the selected sub-folders
    print("Collecting speech-command files...")
    command_folders = [
        os.path.join(args.gcommands, d)
        for d in os.listdir(args.gcommands)
        if (
            d in TARGET_COMMANDS                    # keep only wanted folders
            and os.path.isdir(os.path.join(args.gcommands, d))
        )
    ]

    all_speech_files = []
    for folder in command_folders:
        all_speech_files.extend(glob(os.path.join(folder, '*.wav')))

    # Train/Valid/Test split
    train_files, test_valid_files = train_test_split(all_speech_files, test_size=0.2, random_state=0)
    valid_files, test_files = train_test_split(test_valid_files, test_size=0.5, random_state=0)

    # Load white noise
    print(f"Loading white noise from: {args.white_noise}")
    white_noise, _ = load(args.white_noise, sr=None)

    # Create output directories
    def make_dirs(base):
        clean_path = Path(os.path.join(args.target, base, 'clean'))
        noisy_path = Path(os.path.join(args.target, base, 'noisy'))
        clean_path.mkdir(parents=True, exist_ok=True)
        noisy_path.mkdir(parents=True, exist_ok=True)
        return clean_path, noisy_path

    train_clean_path, train_noisy_path = make_dirs('train')
    valid_clean_path, valid_noisy_path = make_dirs('valid')
    test_clean_path, test_noisy_path = make_dirs('test')

    # Mix and write files
    def mix_and_save(file_list, clean_path, noisy_path, split_name):
        print(f"Creating {split_name} files...")
        for speech_file in tqdm(file_list):
            s, _ = load(speech_file, sr=sr)

            snr_dB = np.random.uniform(min_snr, max_snr)
            speech_power = 1 / len(s) * np.sum(s ** 2)

            start = np.random.randint(0, len(white_noise) - len(s))
            n = white_noise[start:start + len(s)]

            noise_power = 1 / len(n) * np.sum(n ** 2)
            noise_power_target = speech_power * np.power(10, -snr_dB / 10)
            k = noise_power_target / noise_power
            n = n * np.sqrt(k)

            x = s + n

            label = os.path.basename(os.path.dirname(speech_file))
            file_name = f"{label}_{os.path.basename(speech_file)}"
            write(os.path.join(clean_path, file_name), s, sr)
            write(os.path.join(noisy_path, file_name), x, sr)

    # Apply to all splits
    mix_and_save(train_files, train_clean_path, train_noisy_path, 'training')
    mix_and_save(valid_files, valid_clean_path, valid_noisy_path, 'validation')
    mix_and_save(test_files, test_clean_path, test_noisy_path, 'test')
