# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch
import whisper
from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry
import torchaudio

__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y, logits_cond, audio_embedding_cond,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, to_audio_fn=None, pad_or_trim_fn=None, log_mel_spectogram_fn=None, sampling_mode="parallel",
    num_iterations=5, logits_diffusion_steps=10, T_orig=None, allowed_ids_t=None, sr=48000, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    # whisper initialization
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    whisper_model = whisper.load_model(f'base').to(y.device).eval()
    tokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=options.task)
    n_vocab = tokenizer.encoding.n_vocab
    # a mask we will add to logits each step: allowed=0, disallowed=-inf
    neg_inf = -1e9
    base_mask = torch.full((n_vocab,), neg_inf, device=y.device)
    base_mask[allowed_ids_t] = 0.0

    # maximum tokens we’ll allow to emit (you can expose this as a hyperparam)
    max_decode_len = 224
    if hasattr(whisper_model, 'alignment_heads') and whisper_model.alignment_heads.is_sparse:
    # Convert the sparse buffer to a dense buffer
        whisper_model.alignment_heads = whisper_model.alignment_heads.to_dense()    
    # Freeze Whisper parameters but keep gradient computation enabled
    for param in whisper_model.parameters():
        param.requires_grad = False

    def extract_logits(x_t, target_mean, target_std):
        xt_audio = to_audio_fn(x_t.squeeze(), T_orig)    
        if sr != 16000:
            xt_audio_16k = torchaudio.functional.resample(xt_audio.unsqueeze(0), sr, 16000).squeeze(0)
        else:
            xt_audio_16k = xt_audio
        true_audio_length_16k = xt_audio_16k.shape[-1]            
        xt_audio_padded = pad_or_trim_fn(xt_audio_16k)
        whisper_input = log_mel_spectogram_fn(xt_audio_padded)
        enc = whisper_model.encoder(whisper_input) 
        hop_length = 160
        encoder_downsampling_factor = 4
        true_mel_length = (true_audio_length_16k + hop_length - 1) // hop_length
        true_encoder_length = (true_mel_length + encoder_downsampling_factor - 1) // encoder_downsampling_factor
        audio_embedding = enc[:, :true_encoder_length, :]
        # start with SOT + <|notimestamps|>
        prefix = torch.tensor([tokenizer.sot_sequence_including_notimestamps],
                            dtype=torch.long, device=y.device)  # (1, L0)
        out = prefix.clone()
        per_step_allowed = []  # each item: (1, |A|)
        for _ in range(max_decode_len):
            # logits over full vocab for each position so far
            logits = whisper_model.decoder(out, enc).squeeze(0)  # (L, V)
            next_logits = logits[-1]  # (V,)
            best_allowed_idx = int(torch.argmax(next_logits.index_select(0, allowed_ids_t)).item())
            constrained_token_id = allowed_ids_t[best_allowed_idx].item()

            allowed_step = normalize_logits(next_logits.index_select(0, allowed_ids_t).unsqueeze(0), target_mean, target_std)  # (1, |A|) 
            per_step_allowed.append(allowed_step)

            # greedy pick
            next_id = int(torch.argmax(next_logits).item())
            out = torch.cat([out, torch.tensor([[next_id]], device=y.device)], dim=1)

            if constrained_token_id == tokenizer.eot:
                break
        
        step_logits_allowed = torch.cat(per_step_allowed, dim=0)  # (S, |A|)
        return step_logits_allowed.unsqueeze(0), audio_embedding
    
    def pad_align_logits_with_eot(
        clean_logits: torch.Tensor,  # [B, S1, C]
        noisy_logits: torch.Tensor,  # [B, S2, C]
    ):
        """
        Pads the *shorter* sequence (per batch *uniformly* to the global max S)
        with a 1-hot vector at the EOT index. Adjusts masks accordingly.
        Returns:
        clean_logits_p, clean_mask_p, noisy_logits_p, noisy_mask_p, union_mask
        Shapes after: [B, S_max, C] for logits, [B, S_max] for masks.
        """
        device = clean_logits.device
        dtype  = clean_logits.dtype
        B, S1, C = clean_logits.shape
        _, S2, C2 = noisy_logits.shape
        assert C == C2, f"Channel (|A|) mismatch: clean C={C}, noisy C={C2}"

        # ----- find EOT index inside allowed ids -----
        match = (allowed_ids_t == tokenizer.eot).nonzero(as_tuple=True)[0]
        if match.numel() == 0:
            raise ValueError("tok.eot is not in allowed_ids_t; cannot write 1-hot for padded rows.")
        eot_pos_in_allowed = int(match.item())

        # 1-hot vector at EOT for padding rows
        eot_onehot = torch.zeros(C, device=device, dtype=dtype)
        eot_onehot[eot_pos_in_allowed] = 1.0  

        S_max = max(S1, S2)
        if S_max == S1 == S2:
            # No padding needed
            return clean_logits, noisy_logits

        # ----- pad CLEAN up to S_max -----
        if S1 < S_max:
            pad_steps = S_max - S1
            pad_logits_clean = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
            clean_logits = torch.cat([clean_logits, pad_logits_clean], dim=1)
        else:
            # already S_max
            assert clean_logits.shape[1] == S_max 

        # ----- pad NOISY up to S_max -----
        if S2 < S_max:
            pad_steps = S_max - S2
            pad_logits_noisy = eot_onehot.view(1, 1, C).expand(B, pad_steps, C)
            noisy_logits = torch.cat([noisy_logits, pad_logits_noisy], dim=1)
        else:
            assert noisy_logits.shape[1] == S_max 

        return clean_logits, noisy_logits
    
    def normalize_logits(logits: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        logits_mean = logits.mean(dim=1, keepdim=True)  # [B,1]
        logits_std  = logits.std(dim=1, keepdim=True)   # [B,1]

        # avoid divide-by-zero
        eps = 1e-8
        normalized_logits = (logits - logits_mean) / (logits_std + eps)

        logits_rescaled = normalized_logits * target_std + target_mean  # [B,|A|]
        return logits_rescaled

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            if sampling_mode == "parallel":
                xt = sde.prior_sampling(y.shape, y, noise_scale=1.0, is_logits=False).to(y.device)
                audio_mean = y.mean(dim=(1, 2, 3), keepdim=True).abs()  # [B,1,1,1]
                audio_std  = y.std(dim=(1, 2, 3), keepdim=True)   # [B,1,1,1]
                target_mean = audio_mean.squeeze(-1).squeeze(-1)  # [B,1]
                target_std  = audio_std.squeeze(-1).squeeze(-1)   # [B,1]
                logits_cond_current = logits_cond
                logits_t_mean = logits_cond_current
                logits_t = sde.prior_sampling(logits_cond.shape, logits_cond, noise_scale=1.0, is_logits=True).to(y.device)
                timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
                for i in range(sde.N):
                    t = timesteps[i]
                    if i != len(timesteps) - 1:
                        stepsize = t - timesteps[i+1]
                    else:
                        stepsize = timesteps[-1] # from eps to 0
                    vec_t = torch.ones(y.shape[0], device=y.device) * t
                    if i >= sde.N-5:
                        xt, xt_mean = corrector.update_fn(xt, y, vec_t, is_logits=False, logits=logits0_hat, logits_cond=logits_cond_current)
                        xt, xt_mean, x0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=False, logits=logits0_hat, logits_cond=logits_cond_current)
                        kw_probs, audio_embedding = extract_logits(x0_hat, target_mean, target_std)
                        kw_probs, logits_t = pad_align_logits_with_eot(kw_probs, logits_t)
                        logits_t, logits_t_mean = corrector.update_fn(xt, y, vec_t, is_logits=True, logits=logits_t, logits_cond=kw_probs, audio_embedding=audio_embedding)
                        logits_t, logits_t_mean, logits0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=True, logits=logits_t, logits_cond=kw_probs, audio_embedding=audio_embedding)
                    else:
                        xt, xt_mean = corrector.update_fn(xt, y, vec_t, is_logits=False, logits=logits_cond_current, logits_cond=logits_cond_current)
                        xt, xt_mean, x0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=False, logits=logits_cond_current, logits_cond=logits_cond_current)
                        logits_t, logits_t_mean = corrector.update_fn(xt, y, vec_t, is_logits=True, logits=logits_t, logits_cond=logits_cond_current, audio_embedding=audio_embedding_cond)
                        logits_t, logits_t_mean, logits0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=True, logits=logits_t, logits_cond=logits_cond_current, audio_embedding=audio_embedding_cond)

                x_result = xt_mean if denoise else xt
                logits_result = logits_t_mean if denoise else logits_t
                ns = 2 * sde.N * (corrector.n_steps + 1)
            elif sampling_mode == "full":
                logits_cond_current = logits_cond
                logits_result = logits_cond_current
                audio_mean = y.mean(dim=(1, 2, 3), keepdim=True).abs()  # [B,1,1,1]
                audio_std  = y.std(dim=(1, 2, 3), keepdim=True)   # [B,1,1,1]
                target_mean = audio_mean.squeeze(-1).squeeze(-1)  # [B,1]
                target_std  = audio_std.squeeze(-1).squeeze(-1)   # [B,1]
                for j in range(num_iterations):
                    xt = sde.prior_sampling(y.shape, y, noise_scale=1.0, is_logits=False).to(y.device)
                    timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
                    for i in range(sde.N):
                        t = timesteps[i]
                        if i != len(timesteps) - 1:
                            stepsize = t - timesteps[i+1]
                        else:
                            stepsize = timesteps[-1] # from eps to 0
                        vec_t = torch.ones(y.shape[0], device=y.device) * t
                        xt, xt_mean = corrector.update_fn(xt, y, vec_t, is_logits=False, logits=logits_result, logits_cond=logits_result)
                        xt, xt_mean, x0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=False, logits=logits_result, logits_cond=logits_result)
                    x_result = xt_mean if denoise else xt
                    if j == num_iterations - 1:
                        break
                        
                    kw_probs, audio_embedding = extract_logits(x_result, target_mean, target_std)
                    logits_t = sde.prior_sampling(logits_cond.shape, logits_cond, noise_scale=1.0, is_logits=True).to(y.device)
                    timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
                    for i in range(sde.N):
                        t = timesteps[i]
                        if i != len(timesteps) - 1:
                            stepsize = t - timesteps[i+1]
                        else:
                            stepsize = timesteps[-1] # from eps to 0

                        vec_t = torch.ones(y.shape[0], device=y.device) * t
                        kw_probs, logits_t = pad_align_logits_with_eot(kw_probs, logits_t)
                        logits_t, logits_t_mean = corrector.update_fn(xt, y, vec_t, is_logits=True, logits=logits_t, logits_cond=kw_probs, audio_embedding=audio_embedding)
                        logits_t, logits_t_mean, logits0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=True, logits=logits_t, logits_cond=kw_probs, audio_embedding=audio_embedding)
                    logits_result = logits_t_mean if denoise else logits_t

                ns = 2 * sde.N * (corrector.n_steps + 1) * num_iterations - sde.N * (corrector.n_steps + 1) # minus last iteration that logits is not needed
            elif sampling_mode == "nested":
                logits_cond_current = logits_cond
                logits_t_mean = logits_cond_current
                audio_mean = y.mean(dim=(1, 2, 3), keepdim=True).abs()  # [B,1,1,1]
                audio_std  = y.std(dim=(1, 2, 3), keepdim=True)   # [B,1,1,1]
                target_mean = audio_mean.squeeze(-1).squeeze(-1)  # [B,1]
                target_std  = audio_std.squeeze(-1).squeeze(-1)   # [B,1]
                xt = sde.prior_sampling(y.shape, y, noise_scale=1.0, is_logits=False).to(y.device)
                logits_t = sde.prior_sampling(logits_cond.shape, logits_cond, noise_scale=1.0, is_logits=True).to(y.device)
                timesteps_logits = torch.linspace(sde.T, eps, sde.N, device=y.device)
                for i_logits in range(sde.N):
                    t_logits = timesteps_logits[i_logits]
                    if i_logits != len(timesteps_logits) - 1:
                        stepsize_logits = t_logits - timesteps_logits[i_logits+1]
                    else:
                        stepsize_logits = timesteps_logits[-1] # from eps to 0
                    vec_t_logits = torch.ones(y.shape[0], device=y.device) * t_logits
                    if i_logits >= sde.N - logits_diffusion_steps:
                        xt = sde.prior_sampling(y.shape, y, noise_scale=1.0, is_logits=False).to(y.device)
                        timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
                        for i in range(sde.N):
                            t = timesteps[i]
                            if i != len(timesteps) - 1:
                                stepsize = t - timesteps[i+1]
                            else:
                                stepsize = timesteps[-1] # from eps to 0
                            vec_t = torch.ones(y.shape[0], device=y.device) * t
                            xt, xt_mean = corrector.update_fn(xt, y, vec_t, is_logits=False, logits=logits0_hat, logits_cond=logits_cond_current)
                            xt, xt_mean, x0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=False, logits=logits0_hat, logits_cond=logits_cond_current)

                        x_result = xt_mean if denoise else xt
                        kw_probs, audio_embedding = extract_logits(x_result, target_mean, target_std)
                        kw_probs, logits_t = pad_align_logits_with_eot(kw_probs, logits_t)
                        logits_t, logits_t_mean = corrector.update_fn(xt, y, vec_t_logits, is_logits=True, logits=logits_t, logits_cond=kw_probs, audio_embedding=audio_embedding)
                        logits_t, logits_t_mean, logits0_hat = predictor.update_fn(xt, y, vec_t_logits, stepsize_logits, is_logits=True, logits=logits_t, logits_cond=kw_probs, audio_embedding=audio_embedding)
                    else:
                        logits_t, logits_t_mean = corrector.update_fn(xt, y, vec_t_logits, is_logits=True, logits=logits_t, logits_cond=logits_cond_current, audio_embedding=audio_embedding_cond)
                        logits_t, logits_t_mean, logits0_hat = predictor.update_fn(xt, y, vec_t_logits, stepsize_logits, is_logits=True, logits=logits_t, logits_cond=logits_cond_current, audio_embedding=audio_embedding_cond)

                logits_result = logits_t_mean if denoise else logits_t
                xt = sde.prior_sampling(y.shape, y, noise_scale=1.0, is_logits=False).to(y.device)
                timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
                for i in range(sde.N):
                    t = timesteps[i]
                    if i != len(timesteps) - 1:
                        stepsize = t - timesteps[i+1]
                    else:
                        stepsize = timesteps[-1] # from eps to 0
                    vec_t = torch.ones(y.shape[0], device=y.device) * t
                    xt, xt_mean = corrector.update_fn(xt, y, vec_t, is_logits=False, logits=logits_result, logits_cond=logits_cond_current)
                    xt, xt_mean, x0_hat = predictor.update_fn(xt, y, vec_t, stepsize, is_logits=False, logits=logits_result, logits_cond=logits_cond_current)
                x_result = xt_mean if denoise else xt
                ns = sde.N * (logits_diffusion_steps + 1) * (corrector.n_steps + 1) + logits_diffusion_steps * (corrector.n_steps + 1)

            return x_result, ns, logits_result
    
    return pc_sampler


def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None,
    denoise=True, rtol=1e-5, atol=1e-5,
    method='RK45', eps=3e-2, device='cuda', **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, y, vec_eps)
        return x

    def drift_fn(x, y, t):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, y, t)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(y.shape, y).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, y, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler

def get_sb_sampler(sde, model, y, eps=1e-4, n_steps=50, sampler_type="ode", **kwargs):
    # adapted from https://github.com/NVIDIA/NeMo/blob/78357ae99ff2cf9f179f53fbcb02c88a5a67defb/nemo/collections/audio/parts/submodules/schroedinger_bridge.py#L382
    def sde_sampler():
        """The SB-SDE sampler function."""
        with torch.no_grad():
            xt = y[:, [0], :, :] # special case for storm_2ch
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / (alpha_prev * sigma_prev**2 + sde.eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + sde.eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_z = weight_z[:, None, None, None]

                # Random sample
                z_norm = torch.randn_like(xt)
                
                if t == time_steps[-1]:
                    weight_z = 0.0

                # Update state: weighted sum of previous state, current estimate and noise
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps

    def ode_sampler():
        """The SB-ODE sampler function."""
        with torch.no_grad():
            xt = y
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bart / (alpha_prev * sigma_prev * sigma_bar_prev + sde.eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_T**2 + sde.eps)
                    * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + sde.eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_T * sigma_T**2 + sde.eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + sde.eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_prior_mean = weight_prior_mean[:, None, None, None]

                # Update state: weighted sum of previous state, current estimate and prior
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_prior_mean * y

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps
    
    if sampler_type == "sde":
        return sde_sampler
    elif sampler_type == "ode":
        return ode_sampler
    else:
        raise ValueError("Invalid type. Choose 'ode' or 'sde'.")
