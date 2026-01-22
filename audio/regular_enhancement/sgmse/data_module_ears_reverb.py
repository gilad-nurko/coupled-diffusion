
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
# from torchaudio import load
import torchaudio
import whisper
import os
import numpy as np
import torch.nn.functional as F
from sgmse.util.other import pad_spec
import json
import re


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

def stft_num_frames(n_samples: int, n_fft: int, hop_length: int, center: bool = True) -> int:
    # Matches torch.stft framing with center=True (pads by n_fft//2 on both sides)
    if center:
        n_samples = n_samples + 2 * (n_fft // 2)
    return 1 + max(0, (n_samples - n_fft) // hop_length)

def pad_time_to_multiple(x: torch.Tensor, mult_time: int):
    """
    x: [B, 2, F, T] complex tensor (or [B, C, F, T])
    Returns: x_padded, t_pad (how many frames were added on the right)
    """
    if mult_time <= 1:
        return x, 0
    T = x.shape[-1]
    # First make T divisible by mult_time
    t_pad = (mult_time - (T % mult_time)) % mult_time
    T_new = T + t_pad
    # If quotient is odd, add one more block of mult_time
    if (T_new // mult_time) % 2 == 1:
        t_pad += mult_time
    if t_pad > 0:
        x = F.pad(x, (0, t_pad, 0, 0))
    return x, t_pad

def parse_category_from_stem(stem: str) -> str:
    """
    Extracts the text after 'category_' in the filename stem.

    Example:
      '00000_0.51_category_emo_adoration_sentences' -> 'emo_adoration_sentences'
      '00001_0.44_category_emo_anger_sentences'     -> 'emo_anger_sentences'
    """
    m = re.search(r'category_(.*)$', stem)
    return m.group(1) if m else stem

class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, transcripts_path="/mlspeech/data/gilad/paper_ears_reverbed/transcripts.json", **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "**", "*.wav")))
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverberant", "*.wav")))
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverberant", "**", "*.wav")))
        # elif format == "reverb":
        #     self.clean_files = []
        #     self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "*.wav")))
        #     self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "**", "*.wav")))
        #     self.noisy_files = []
        #     self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
        #     self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "**", "*.wav")))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.sample_rate = 48000
        # Whisper tokenizer setup
        woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.multilingual_tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, language="en", task=woptions.task
        )
        self.eot_id = self.multilingual_tokenizer.eot

        # Load transcripts once
        with open(transcripts_path, "r", encoding="utf-8") as f:
            self.transcripts = json.load(f)
        if not isinstance(self.transcripts, dict):
            raise ValueError(f"Transcripts at {transcripts_path} must be a dict of {{category: transcript_str}}")

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        clean_path = self.clean_files[i]
        noisy_path = self.noisy_files[i]
        # # Parse category from filename stem
        # stem = os.path.splitext(os.path.basename(clean_path))[0]
        # category = parse_category_from_stem(stem)
        x, sr = torchaudio.load(clean_path)
        y, sr_y = torchaudio.load(noisy_path)

        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.sample_rate)
        if sr_y != self.sample_rate:
            y = torchaudio.functional.resample(y, sr_y, self.sample_rate)
        # convert x,y from [1, T] to [T]
        if x.dim() == 2:
            x = x.squeeze(0)
        if y.dim() == 2:
            y = y.squeeze(0)
        
        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        # x = x[..., :self.hop_length*127]
        # y = y[..., :self.hop_length*127]
        current_len = x.size(-1)
        # # formula applies for center=True
        # target_len = (self.num_frames - 1) * self.hop_length
        # pad = max(target_len - current_len, 0)
        # if pad == 0:
        #     # extract random part of the audio file
        #     if self.shuffle_spec:
        #         start = int(np.random.uniform(0, current_len-target_len))
        #     else:
        #         start = int((current_len-target_len)/2)
        #     x = x[..., start:start+target_len]
        #     y = y[..., start:start+target_len]
        # else:
        #     # pad audio if the length T is smaller than num_frames
        #     # x = F.pad(x, (0, pad), mode='constant')
        #     # y = F.pad(y, (0, pad), mode='constant')
        #     x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
        #     y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        # X = pad_spec(X, mode="zero_pad")
        # Y = pad_spec(Y, mode="zero_pad")
        specs = torch.stack([Y, X], dim=0)  # (2, F, T), complex

        # # --------- NEW: tokens & labels from transcript ---------
        # # Get transcript text by category; raise if missing (helps catch dataset issues)
        # try:
        #     transcript = self.transcripts[category]
        # except KeyError as e:
        #     raise KeyError(f"Transcript missing for category '{category}'. Check transcripts.json keys.") from e

        # # Whisper best practice: prefix a space so tokenization behaves like continuation
        # # SOT + encoded transcript, labels = next-token shift + EOT
        # tok = self.multilingual_tokenizer
        # tokens = [*tok.sot_sequence_including_notimestamps] + tok.encode(" " + transcript)
        # labels = tokens[1:] + [tok.eot]

        # multilingual_tokens = torch.tensor(tokens, dtype=torch.long)
        # labels = torch.tensor(labels, dtype=torch.long)
        # # --------------------------------------------------------

        T_frames = specs.shape[-1]

        # return: stacked STFTs, tokens, labels
        return specs, current_len, T_frames

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)
    

class STFTDataCollatorWithPadding:
    def __init__(self, time_downsample_multiple: int = 32, eot_id=50257, ignore_index=-100):
        self.time_downsample_multiple = time_downsample_multiple
        self.eot_id = eot_id
        self.ignore_index = ignore_index

    def __call__(self, features):
        # Unpack items
        # Each item: (specs [2,F,T_i], toks, labs, length_samples, T_frames_i)
        specs_list, sample_lengths, T_list = zip(*features)

        # # 1) Pad multilingual tokens / labels (unchanged behavior)
        # max_tok = max(max(len(t) for t in toks_list), max(len(l) for l in labs_list))
        # toks = torch.stack([F.pad(t, (0, max_tok - len(t)), value=self.eot_id) for t in toks_list])
        # labs = torch.stack([F.pad(l, (0, max_tok - len(l)), value=self.ignore_index) for l in labs_list])

        # 2) Find max T in batch and pad specs to batch max (per-batch padding)
        # Stack after padding because T_i differ
        F_dim = specs_list[0].shape[-2]  # frequency bins
        max_T = max(int(Ti) for Ti in T_list)

        padded_specs = []
        time_masks = []  # [B,1,1,T_max_after_multiple]

        for specs, Ti in zip(specs_list, T_list):
            # specs: [2,F,Ti], complex
            t_pad = max_T - int(Ti)
            if t_pad > 0:
                specs = F.pad(specs, (0, t_pad, 0, 0))  # pad time to right

            padded_specs.append(specs)

        specs = torch.stack(padded_specs, dim=0)  # [B,2,F,max_T], complex

        # 3) Now ensure T is a multiple of the UNet time stride
        specs, t_mult_pad = pad_time_to_multiple(specs, self.time_downsample_multiple)
        T_after = specs.shape[-1]

        # 4) Build time mask (1 for real, 0 for padded)
        # First mask up to max_T, then extend with zeros for the multiple padding
        mask = torch.zeros(len(T_list), 1, 1, T_after, dtype=specs.real.dtype)
        for b, Ti in enumerate(T_list):
            mask[b, :, :, :int(Ti)] = 1.0
        # After step (2) we padded to max_T; areas [Ti:max_T) are already zeros.
        # After step (3) we may have added t_mult_pad zeros at the end — kept as zeros.

        # 5) Also pad tokens/labels already done above
        sample_lengths = torch.tensor(sample_lengths, dtype=torch.long)
        T_frames = torch.tensor(T_list, dtype=torch.long)

        return {
            "specs": specs,                      # [B,2,F,T*], complex
            "time_mask": mask,                   # [B,1,1,T*], float {0,1}
            # "multilingual_tokens": toks,         # [B,L_tokens]
            # "labels": labs,                      # [B,L_tokens]
            "lengths_samples": sample_lengths,   # [B] original waveform lengths (samples)
            "lengths_frames": T_frames           # [B] original frame counts before padding
        }


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, default="/mlspeech/data/gilad/paper_ears_reverbed/EARS-Reverb_v2_4sec_chunks", help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default", "reverb"), default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=1, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=1534, help="Number of FFT bins. 510 by default.")  
        parser.add_argument("--hop_length", type=int, default=384, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=128, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.065, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.667, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--time_downsample_multiple", type=int, default=32, help="Pad time frames to this multiple so UNet down/upsampling divides cleanly.")
        parser.add_argument("--transcripts_path", type=str, default="/mlspeech/data/gilad/paper_ears_reverbed/transcripts.json", help="Path to transcripts JSON mapping {category: transcript_str}.")
        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent",
        time_downsample_multiple: int = 32, **kwargs
        ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.time_downsample_multiple = time_downsample_multiple
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs
        # Build a whisper tokenizer here only to grab a consistent EOT id for padding
        woptions_dm = whisper.DecodingOptions(language="en", without_timestamps=True)
        tok_dm = whisper.tokenizer.get_tokenizer(multilingual=True, language="en", task=woptions_dm.task)
        self.collator = STFTDataCollatorWithPadding(
            time_downsample_multiple=self.time_downsample_multiple,
            eot_id=tok_dm.eot
        )

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, transcripts_path=self.kwargs.get("transcripts_path",
                                                                           "/mlspeech/data/gilad/paper_ears_reverbed/transcripts.json"),
            **self.kwargs
        )
        # if stage == 'fit' or stage is None:
        #     self.train_set = Specs(data_dir=self.base_dir, subset='train',
        #         dummy=self.dummy, shuffle_spec=True, format=self.format,
        #         normalize=self.normalize, **specs_kwargs)
        #     self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
        #         dummy=self.dummy, shuffle_spec=False, format=self.format,
        #         normalize=self.normalize, **specs_kwargs)
        # if stage == 'test' or stage is None:
        #     self.test_set = Specs(data_dir=self.base_dir, subset='test',
        #         dummy=self.dummy, shuffle_spec=False, format=self.format,
        #         normalize=self.normalize, **specs_kwargs)
        if stage in ('fit', None):
            self.train_set = Specs(
                data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize, **specs_kwargs
            )

        # Validation split (needed for fit and validate)
        if stage in ('fit', 'validate', None):
            self.valid_set = Specs(
                data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs
            )

        # Test split (needed for test)
        if stage in ('test', None):
            self.test_set = Specs(
                data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs
            )

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True, collate_fn=self.collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False, collate_fn=self.collator
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False, collate_fn=self.collator
        )
