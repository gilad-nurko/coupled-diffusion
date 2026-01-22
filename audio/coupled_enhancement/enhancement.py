import os
import glob
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list, pad_spec
set_torch_cuda_arch_list()

from sgmse.model_ears_with_logits_1D_model import WhisperGuidedScoreModel


def normalize_logits(step_logits_allowed: torch.Tensor,
                     target_mean: torch.Tensor,
                     target_std: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Simple per-step normalization of logits to match the audio statistics.
    step_logits_allowed: (1, |A|)
    target_mean, target_std: (1, 1)
    """
    mean = step_logits_allowed.mean(dim=-1, keepdim=True)
    std = step_logits_allowed.std(dim=-1, keepdim=True) + eps
    # Broadcast target_mean/std over the vocab dimension
    return (step_logits_allowed - mean) / std * target_std + target_mean


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--test_dir", type=str, default='/mlspeech/data/gilad/paper_ears_reverbed/EARS-Reverb_v2_4sec_chunks/test/reverberant',
    #                     help='Directory containing the test data')
    # parser.add_argument("--enhanced_dir", type=str, default='/mlspeech/data/gilad/paper_ears_reverbed/EARS-Reverb_v2_4sec_chunks_test_enhanced_with_logits_nested_try2',
    #                     help='Directory to save the enhanced data')
    parser.add_argument("--test_dir", type=str, default='/mlspeech/data/gilad/paper_ears_wham/EARS-WHAM_v2_4sec_chunks/test/noisy',
                        help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, default='/mlspeech/data/gilad/paper_ears_wham/EARS-WHAM_v2_4sec_chunks_test_enhanced_with_logits_nested_epoch44',
                        help='Directory to save the enhanced data')
    parser.add_argument("--ckpt", type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="pc",
                        help="Sampler type for the sampler (pc / ode).")
    parser.add_argument("--corrector", type=str,
                        choices=("ald", "langevin", "none"),
                        default="ald",
                        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1,
                        help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5,
                        help="SNR value for (annealed) Langevin dynamics")
    parser.add_argument("--N", type=int, default=50,
                        help="Number of reverse diffusion steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference (cuda / cpu)")
    parser.add_argument("--t_eps", type=float, default=0.03,
                        help="The minimum process time (0.03 by default)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load score model
    model = WhisperGuidedScoreModel.load_from_checkpoint(
        args.ckpt,
        map_location=device
    )
    model.t_eps = args.t_eps
    model.to(device)
    model.eval()

    # If the SDE has a sampler_type attribute, respect the CLI flag
    if hasattr(model.sde, "sampler_type"):
        model.sde.sampler_type = args.sampler_type

    # Sample rate the model was trained on
    target_sr = model.sr

    # Tokenizer / allowed tokens for Whisper (reused from model)
    tok = model.multilingual_tokenizer
    max_decode_len = 224
    allowed_ids_t = torch.as_tensor(
        model.allowed_toks, dtype=torch.long, device=device
    )

    def decode_constrained(signal_16k, target_mean, target_std):
        """
        Greedy, closed-set decode: at each step, use logits over the full vocab,
        but record the logits restricted to the allowed token set.

        Returns:
            pred_text: decoded string from constrained tokens
            mel: log-mel spectrogram used by Whisper
            step_logits_allowed: (S, |A|) logits over allowed tokens
            audio_embedding: encoder features (1, T', C) truncated to true length
        """
        # to device, 1D float tensor
        wav = torch.tensor(signal_16k, dtype=torch.float32, device=device)
        true_audio_length = wav.shape[-1]

        # Use the same helpers as in the model
        wav = model._pad_or_trim(wav)              # (T,)
        mel = model._log_mel_spectrogram(wav)      # (1, 80, Tm)
        feats = model.whisper.encoder(mel)         # (1, T', C)

        # Trim encoder outputs to true length
        hop_length = 160
        encoder_downsampling_factor = 2
        true_mel_length = (true_audio_length + hop_length - 1) // hop_length
        true_encoder_length = (true_mel_length + encoder_downsampling_factor - 1) // encoder_downsampling_factor
        audio_embedding = feats[:, :true_encoder_length, :]

        # Start with SOT + <|notimestamps|>
        prefix = torch.tensor(
            [tok.sot_sequence_including_notimestamps],
            dtype=torch.long,
            device=device
        )  # (1, L0)
        out = prefix.clone()
        per_step_allowed = []   # list of (1, |A|)
        constrained_tokens = []

        for _ in range(max_decode_len):
            # logits over full vocab for each position so far
            logits = model.whisper.decoder(out, feats).squeeze(0)  # (L, V)
            next_logits = logits[-1]                               # (V,)

            # best allowed token for tracking "closed set"
            allowed_logits = next_logits.index_select(0, allowed_ids_t)  # (|A|,)
            best_allowed_idx = int(torch.argmax(allowed_logits).item())
            constrained_token_id = allowed_ids_t[best_allowed_idx].item()
            constrained_tokens.append(constrained_token_id)

            # record logits over allowed set for this step
            allowed_step = allowed_logits.unsqueeze(0)  # (1, |A|)
            allowed_step = normalize_logits(
                allowed_step, target_mean, target_std
            )
            per_step_allowed.append(allowed_step)

            # greedy pick over full vocab (unconstrained)
            next_id = int(torch.argmax(next_logits).item())
            out = torch.cat(
                [out, torch.tensor([[next_id]], device=device)], dim=1
            )

            if constrained_token_id == tok.eot:
                break

        step_logits_allowed = torch.cat(per_step_allowed, dim=0)  # (S, |A|)

        # Strip trailing EOT from constrained token sequence (for text)
        if len(constrained_tokens) > 0 and constrained_tokens[-1] == tok.eot:
            constrained_tokens = constrained_tokens[:-1]
        text = tok.decode(constrained_tokens)

        return text, mel, step_logits_allowed, audio_embedding

    # Get list of noisy files
    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.flac')))

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        # Load wav (mono)
        y, sr = load(noisy_file)        # y: (1, T)
        y = y.squeeze().numpy()         # -> (T,)
        orig_sr = sr

        # Resample to model sample rate for enhancement
        if orig_sr != target_sr:
            y_model = resample(y, orig_sr=orig_sr, target_sr=target_sr).squeeze()
        else:
            y_model = y

        # Resample to 16kHz for Whisper front-end (ASR guidance)
        if orig_sr != 16000:
            y_16k = resample(y, orig_sr=orig_sr, target_sr=16000).squeeze()
        else:
            y_16k = y

        # --------- build STFT stats for logit normalization ----------
        y_tensor = torch.tensor(y_model, dtype=torch.float32, device=device)
        signal = y_tensor
        norm_factor = signal.abs().max().item()
        signal = signal / norm_factor

        # Shape: (1, 1, F, T) as in validation_step
        Y = torch.unsqueeze(
            model._forward_transform(model._stft(signal.to(device))),
            0
        ).unsqueeze(0)
        Y = pad_spec(Y)

        audio_mean = Y.mean(dim=(1, 2, 3), keepdim=True).abs()  # [B,1,1,1]
        audio_std = Y.std(dim=(1, 2, 3), keepdim=True)          # [B,1,1,1]
        target_mean = audio_mean.squeeze(-1).squeeze(-1)        # [B,1]
        target_std = audio_std.squeeze(-1).squeeze(-1)          # [B,1]

        # --------- closed-set Whisper decode on noisy signal -------
        corrupted_transcript, corrupted_mel, corrupted_logits, corrupted_audio_embedding = \
            decode_constrained(y_16k, target_mean, target_std)

        # Shape them for the model's enhance() API
        logits_cond = corrupted_logits.unsqueeze(0).to(device)        # [1, S, |A|]
        audio_embedding = corrupted_audio_embedding.to(device)        # [1, T', C]

        # --------- run enhancement using the model's enhance() ------
        # y_model is at target_sr; pass as [B, T]
        y_for_model = torch.tensor(y_model, dtype=torch.float32, device=device).unsqueeze(0)

        x_hat, _ = model.enhance(
            y_for_model,
            logits_cond,
            audio_embedding,
            sampler_type=args.sampler_type,
            predictor="reverse_diffusion",
            corrector=args.corrector,
            N=args.N,
            corrector_steps=args.corrector_steps,
            snr=args.snr,
        )

        # Write enhanced wav file
        out_path = join(args.enhanced_dir, filename)
        makedirs(dirname(out_path), exist_ok=True)
        write(out_path, x_hat, target_sr)
