import torch.nn as nn
import torch
from torchvision.transforms import GaussianBlur
import math
from tqdm import tqdm


LOGITS_MEAN = 0.0
LOGITS_STD  = 0.8   # same convention as your other CIFAR100 logits model


class MNISTDiffusion(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        time_embedding_dim=256,
        timesteps=150,
        base_dim=32,
        dim_mults=[1, 2, 4, 8],
        classifier=None,
        is_cond=False,
        is_y_cond=False,
        guiding_noise_level=0.15,
        y_initialization=50,
        inner_loop_jump_step=20,
        is_ddim=True,
        num_classes=100,
        mode="cifar100",
        device="cuda",
        pretrained_model_y_ckpt=None,
        corruption_type='gaussian_blur', 
        blur_sigma=2.0
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.guiding_noise_level = guiding_noise_level
        self.is_y_cond = is_y_cond
        self.y_initialization = y_initialization
        self.inner_loop_jump_step = inner_loop_jump_step
        self.is_ddim = is_ddim
        self.num_classes = num_classes
        self.mode = mode
        self.corruption_type = corruption_type
        if self.corruption_type == 'gaussian_blur':
            # Hardcoded kernel_size=5
            self.blur_transform = GaussianBlur(kernel_size=5, sigma=blur_sigma)

        # variance schedule
        betas = self._cosine_variance_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        if self.mode == "cifar100":
            mean = torch.tensor((0.5071, 0.4867, 0.4408)).view(1,3,1,1)
            std  = torch.tensor((0.2675, 0.2565, 0.2761)).view(1,3,1,1)
        else:
            mean = torch.tensor((0.4811, 0.4575, 0.4079)).view(1,3,1,1)
            std  = torch.tensor((0.2604, 0.2532, 0.2682)).view(1,3,1,1)
        self.register_buffer("data_mean", mean)
        self.register_buffer("data_std", std)

        # image branch
        if mode=="imagenet32_1k":
            from unet_y_cond_1000_classes import Unet
        else:
            from unet_y_cond import Unet

        self.model_x = Unet(
            timesteps,
            time_embedding_dim,
            in_channels,
            in_channels,
            base_dim,
            dim_mults,
            is_cond=is_cond,
            is_y_cond=is_y_cond,
            num_classes=num_classes,
        )

        # logits branch
        if mode=="cifar100" or mode=="imagenet32" or mode=="imagenet32_1k":
            if mode=="imagenet32_1k":
                from y_probs_model_imagenet_1k import ConditionalModel
            else:
                from y_probs_model_cifar100 import ConditionalModel
            self.model_y = ConditionalModel(
                feature_dim=512,
                hidden_dim=1024,
                n_input_channels=in_channels,
                num_classes=num_classes,
                timesteps=timesteps,
                num_heads=8,
            )
            # ---- load pretrained logits diffusion model_y if provided ----
            if pretrained_model_y_ckpt is not None:
                ckpt = torch.load(pretrained_model_y_ckpt, map_location=device)

                if "logits_model_state_dict" in ckpt:
                    # case: standalone logits pretrain
                    state_dict = ckpt["logits_model_state_dict"]
                elif "diffusion_state_dict" in ckpt:
                    # case: stored inside a bigger diffusion checkpoint, extract model_y.*
                    sd = ckpt["diffusion_state_dict"]
                    state_dict = {
                        k.replace("model_y.", ""): v
                        for k, v in sd.items()
                        if k.startswith("model_y.")
                    }
                else:
                    # very defensive fallback: try to treat whole ckpt as state_dict
                    state_dict = ckpt

                missing, unexpected = self.model_y.load_state_dict(state_dict, strict=False)
                print(f"[model_y] loaded pretrained weights from {pretrained_model_y_ckpt}")
                print(f"  missing keys   : {len(missing)}")
                print(f"  unexpected keys: {len(unexpected)}")
        elif mode == "cifar10":
            from y_probs_model_cifar10 import ConditionalModel
            self.model_y = ConditionalModel(
                n_input_channels=in_channels,
                num_classes=num_classes,
            )
        else:
            from y_probs_model import ConditionalModel
            self.model_y = ConditionalModel(
                n_input_channels=in_channels,
                num_classes=num_classes,
            )

        self.classifier = classifier

    # ---------------- CIFAR100 normalization helpers ---------------- #
    def _normalize_image(self, x):
        return (x - self.data_mean) / self.data_std

    # ---------------- forward ---------------- #
    def forward(self, x, noise, is_x=True, target=None,
                pre_train=False, given_t=None, x_cond=None, y_cond=None):
        """
        x      : NCHW CIFAR-normalized images
        noise  : for x-branch same shape as x; for y-branch same shape as target (logits)
        target : logits from classifier(image)
        """
        if pre_train:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        else:
            t = given_t

        y_0_pred = None

        if pre_train:
            # x pre-training, ignore classifier
            x_t = self._forward_diffusion(x, t, noise, is_x=True)
            if self.corruption_type == 'pixel_noise':
                noise_cond = torch.randn_like(x).to(x.device)
                mask = torch.bernoulli(torch.full(x.shape, self.guiding_noise_level, device=x.device))
                guiding_cond = x * (1 - mask) + noise_cond * mask
            elif self.corruption_type == 'gaussian_blur':
                guiding_cond = self.blur_transform(x)
            y_cond = torch.zeros(x.shape[0], self.num_classes, device=x.device)
            pred_noise = self.model_x(x_t, t, guiding_cond, y_cond=y_cond)
        else:
            if is_x:
                # x-branch
                x_t = self._forward_diffusion(x, t, noise, is_x=True)
                pred_noise = self.model_x(x_t, t, x_cond, y_cond=y_cond)
            else:
                # y-branch, logit diffusion
                with torch.no_grad():
                    class_logits = self.classifier(x_cond)  # [B, C]
                    # per-sample normalization to LOGITS_STD, no softmax
                    class_mean = class_logits.mean(dim=1, keepdim=True)
                    class_std  = class_logits.std(dim=1, keepdim=True)
                    class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
                    y_cond = class_logits

                    # forward diffuse target logits (raw classifier logits)
                    y_t = self._forward_diffusion(target, t, noise, is_x=False)

                # time normalization for model_y
                t_norm = t.float() / (self.timesteps - 1)
                pred_noise = self.model_y(y_t, t_norm, y_cond=y_cond, x=x_cond)

                alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(y_t.shape[0], 1)
                y_0_pred = (
                    torch.sqrt(1. / alpha_t_cumprod) * y_t
                    - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred_noise
                )
                # normalize y_0_pred to LOGITS_MEAN/STD, no softmax
                y_0_mean = y_0_pred.mean(dim=1, keepdim=True)
                y_0_std  = y_0_pred.std(dim=1, keepdim=True)
                y_0_pred = LOGITS_STD * (y_0_pred - y_0_mean) / (y_0_std + 1e-5) + LOGITS_MEAN

        return pred_noise, y_0_pred

    # ---------------- x0 reconstruction for nested scheme ---------------- #
    # @torch.no_grad()
    # def get_x_0(self, noisy_image, y_0):
    #     """
    #     reverse diffusion purely in x-space
    #     returns CIFAR-normalized x_0 (so classifier can be applied directly)
    #     """
    #     x_t = torch.randn_like(noisy_image).to(noisy_image.device)
    #     for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling x_0"):
    #         noise = torch.randn_like(x_t).to(noisy_image.device)
    #         t = torch.full((noisy_image.size(0),), i, device=noisy_image.device, dtype=torch.long)
    #         x_t = self._reverse_diffusion_of_x(x_t, t, noise, noisy_image, y_0)

    #     # original code: [-1,1] -> [0,1]
    #     x_t = (x_t + 1.) / 2.
    #     # then CIFAR-normalize, consistent with classifier expectations
    #     x_t = self._normalize_image(x_t)
    #     return x_t
    @torch.no_grad()
    def get_x_0(self, noisy_image, y_0, step_indices=None):
        # Allow passing pre-calculated indices
        if step_indices is None:
            step_indices = torch.arange(self.timesteps - 1, -1, -1, device=noisy_image.device)
            
        x_t = torch.randn_like(noisy_image).to(noisy_image.device)
        for i in tqdm(step_indices, desc="Sampling x_0", leave=False):
            noise = torch.randn_like(x_t).to(noisy_image.device)
            t = torch.full((noisy_image.size(0),), int(i.item()), device=noisy_image.device, dtype=torch.long)
            x_t = self._reverse_diffusion_of_x(x_t, t, noise, noisy_image, y_0)
        
        x_t = (x_t + 1.) / 2.
        x_t = self._normalize_image(x_t)
        return x_t

    @torch.no_grad()
    def _reverse_diffusion_of_x(self, x_t, t, noise, noisy_image, y_0):
        pred = self.model_x(x_t, t, noisy_image, y_0)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
        x_0_pred.clamp_(-1., 1.)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (
                (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred
                + ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t
            )
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred
            std = 0.0

        if self.is_ddim:
            std = 0.0

        return mean + std * noise

    # ---------------- nested sampling ---------------- #
    # @torch.no_grad()
    # def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda", cond=None):
    #     """
    #     cond : CIFAR-normalized images (NCHW)
    #     returns:
    #       x_0     : CIFAR-normalized reconstruction
    #       y_logits: logits in normalized logit-space (LOGITS_MEAN/STD)
    #     """
    #     y_t = torch.randn((n_samples, self.num_classes), device=device)
    #     y_0 = torch.zeros(n_samples, self.num_classes, device=device)

    #     if self.corruption_type == 'pixel_noise':
    #         noise_cond = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size), device=device)
    #         mask = torch.bernoulli(
    #             torch.full((n_samples, self.in_channels, self.image_size, self.image_size),
    #                        self.guiding_noise_level, device=device)
    #         )
    #         x_cond = cond * (1 - mask) + noise_cond * mask
    #     elif self.corruption_type == 'gaussian_blur':
    #         x_cond = self.blur_transform(cond)

    #     # initial x_0
    #     x_0 = self.get_x_0(x_cond, y_0)
    #     y_image_cond = x_0

    #     with torch.no_grad():
    #         class_logits = self.classifier(x_0)
    #         class_mean = class_logits.mean(dim=1, keepdim=True)
    #         class_std  = class_logits.std(dim=1, keepdim=True)
    #         class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
    #         y_cond = class_logits

    #     for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling y"):
    #         noise_y = torch.randn_like(y_t).to(device)
    #         t = torch.full((n_samples,), i, device=device, dtype=torch.long)

    #         if (i + 1) <= self.y_initialization and (i + 1) % self.inner_loop_jump_step == 0:
    #             x_0 = self.get_x_0(x_cond, y_0)
    #             y_image_cond = x_0
    #             with torch.no_grad():
    #                 class_logits = self.classifier(x_0)
    #                 class_mean = class_logits.mean(dim=1, keepdim=True)
    #                 class_std  = class_logits.std(dim=1, keepdim=True)
    #                 class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
    #                 y_cond = class_logits

    #         if clipped_reverse_diffusion:
    #             y_t, y_0 = self._reverse_diffusion_with_clip(
    #                 y_t, t, noise_y, y_image_cond, y_cond, is_x=False
    #             )
    #         else:
    #             raise NotImplementedError

    #     # final normalization of y_t to LOGITS_MEAN/STD, no softmax
    #     y_mean = y_t.mean(dim=1, keepdim=True)
    #     y_std  = y_t.std(dim=1, keepdim=True)
    #     y_t = LOGITS_STD * (y_t - y_mean) / (y_std + 1e-5) + LOGITS_MEAN

    #     return x_0, y_t
    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda", cond=None, sampling_steps=None):
        # Setup step indices
        if sampling_steps is None or sampling_steps == self.timesteps:
            step_indices = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            step_indices = torch.linspace(self.timesteps - 1, 0, steps=int(sampling_steps), device=device).long()

        y_t = torch.randn((n_samples, self.num_classes), device=device)
        y_0 = torch.zeros(n_samples, self.num_classes, device=device)
        
        if self.corruption_type == 'pixel_noise':
            noise_cond = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size), device=device)
            mask = torch.bernoulli(torch.full((n_samples, self.in_channels, self.image_size, self.image_size), self.guiding_noise_level, device=device))
            x_cond = cond * (1 - mask) + noise_cond * mask
        elif self.corruption_type == 'gaussian_blur':
            x_cond = self.blur_transform(cond)
            
        # initial x_0 (Pass the step_indices to make this faster too!)
        x_0 = self.get_x_0(x_cond, y_0, step_indices=step_indices)
        
        y_image_cond = x_0
        with torch.no_grad():
            class_logits = self.classifier(x_0)
            class_mean = class_logits.mean(dim=1, keepdim=True)
            class_std  = class_logits.std(dim=1, keepdim=True)
            class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
            y_cond = class_logits
            
        for i in tqdm(step_indices, desc=f"Sampling y ({len(step_indices)} steps)"):
            noise_y = torch.randn_like(y_t).to(device)
            
            val_i = int(i.item())
            t = torch.full((n_samples,), val_i, device=device, dtype=torch.long)
            
            # Use val_i (actual timestep) for logic checks
            if (val_i + 1) <= self.y_initialization and (val_i + 1) % self.inner_loop_jump_step == 0:
                # We also pass step_indices to the inner loop for consistency and speed
                x_0 = self.get_x_0(x_cond, y_0, step_indices=step_indices)
                y_image_cond = x_0
                with torch.no_grad():
                    class_logits = self.classifier(x_0)
                    class_mean = class_logits.mean(dim=1, keepdim=True)
                    class_std  = class_logits.std(dim=1, keepdim=True)
                    class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
                    y_cond = class_logits
                    
            if clipped_reverse_diffusion:
                y_t, y_0 = self._reverse_diffusion_with_clip(
                    y_t, t, noise_y, y_image_cond, y_cond, is_x=False
                )
            else:
                raise NotImplementedError
                
        y_mean = y_t.mean(dim=1, keepdim=True)
        y_std  = y_t.std(dim=1, keepdim=True)
        y_t = LOGITS_STD * (y_t - y_mean) / (y_std + 1e-5) + LOGITS_MEAN
        return x_0, y_t

    # ---------------- diffusion utilities ---------------- #
    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise, is_x=True):
        assert x_0.shape == noise.shape
        if is_x:
            return (
                self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0
                + self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * noise
            )
        else:
            return (
                self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1) * x_0
                + self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1) * noise
            )

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise, x_cond=None, y_cond=None, is_x=True):
        if is_x:
            pred = self.model_x(x_t, t, x_cond, y_cond)
            alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
            alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
            beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
                -1, t
            ).reshape(x_t.shape[0], 1, 1, 1)
            mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)

            if t.min() > 0:
                alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
                std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
            else:
                std = 0.0

            return mean + std * noise
        else:
            # logits branch with t_norm
            t_norm = t.float() / (self.timesteps - 1)
            pred = self.model_y(x_t, t_norm, y_cond, x_cond)
            alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1)
            alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1)
            beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
                -1, t
            ).reshape(x_t.shape[0], 1)
            mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)

            if t.min() > 0:
                alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1)
                std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
            else:
                std = 0.0

            if self.is_ddim:
                std = 0.0

            return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise, x_cond=None, y_cond=None, is_x=True):
        if is_x:
            pred = self.model_x(x_t, t, x_cond, y_cond)
            alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
            alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
            beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

            x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
            x_0_pred.clamp_(-1., 1.)

            if t.min() > 0:
                alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
                mean = (
                    (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred
                    + ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t
                )
                std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
            else:
                mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred
                std = 0.0

            if self.is_ddim:
                std = 0.0

            return mean + std * noise
        else:
            # logits branch
            t_norm = t.float() / (self.timesteps - 1)
            pred = self.model_y(x_t, t_norm, y_cond, x_cond)
            alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1)
            alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1)
            beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1)

            x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred

            # normalize to LOGITS_STD, no softmax
            x_0_mean = x_0_pred.mean(dim=1, keepdim=True)
            x_0_std  = x_0_pred.std(dim=1, keepdim=True)
            x_0_pred = LOGITS_STD * (x_0_pred - x_0_mean) / (x_0_std + 1e-5) + LOGITS_MEAN

            if t.min() > 0:
                alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1)
                mean = (
                    (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred
                    + ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t
                )
                std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
            else:
                mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred
                std = 0.0

            if self.is_ddim:
                std = 0.0

            # here x_0_pred is actually y_0_pred in normalized logit-space
            y_0_pred = x_0_pred
            return mean + std * noise, y_0_pred
