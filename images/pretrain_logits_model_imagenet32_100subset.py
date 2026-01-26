import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier_imagenet32_100subset import (
    ImageNet32ResNet,
    build_imagenet32_datasets,
)
from y_probs_model_cifar100 import ConditionalModel  # logits model file


# -----------------------
# Data: ImageNet-32 (100-class subset, NPZ)
# -----------------------
def create_imagenet32_100_dataloaders(
    batch_size,
    image_size=32,          # kept for interface compatibility
    num_workers=4,
    root="",
    num_subset_classes=100,
    seed=42,
):
    """
    Uses the SAME subset selection + transforms as in classifier_imagenet32_100subset.py
    via build_imagenet32_datasets, which reads NPZ files:
      root/Imagenet32_train_npz/train_data_batch_1.npz ... _10.npz
      root/Imagenet32_val_npz/val_data.npz
    """
    train_dataset, test_dataset = build_imagenet32_datasets(
        root=root,
        num_subset_classes=num_subset_classes,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# -----------------------
# Diffusion in logits space
# -----------------------
class LogitsDiffusion(nn.Module):
    """
    Simple DDPM-style diffusion on logits:

      y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1 - alpha_bar_t) * eps

    The model predicts eps given (y_t, t, y_cond, x).
    """
    def __init__(
        self,
        logits_model: nn.Module,
        timesteps: int = 150,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = logits_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Sample y_t from q(y_t | y_0, t).
        y0: [B, num_classes]
        t:  [B] integer timesteps
        """
        if noise is None:
            noise = torch.randn_like(y0)

        # [B, 1]
        sqrt_alpha_bar = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)

        y_t = sqrt_alpha_bar * y0 + sqrt_one_minus * noise
        return y_t, noise

    def forward(
        self,
        y0: torch.Tensor,
        y_cond: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        One training step:
          - sample y_t and eps
          - model predicts eps_hat
        """
        y_t, noise = self.q_sample(y0, t)

        # normalize t to [0, 1] for sinusoidal / MLP embedding
        t_norm = t.float() / (self.timesteps - 1)

        eps_pred = self.model(
            y=y_t,
            t=t_norm,
            y_cond=y_cond,
            x=x,
        )
        return eps_pred, noise
    
    @torch.no_grad()
    def sample(
        self,
        y_cond: torch.Tensor,
        x_cond: torch.Tensor,
        ddim: bool = True,
    ) -> torch.Tensor:
        """
        Run reverse diffusion to get enhanced logits y_0_hat from noisy logits condition.

        Args:
            y_cond: [B, num_classes]  - logits of the noisy image from the classifier
            x_cond: [B, 3, H, W]      - noisy images used as extra condition
            ddim:   whether to use deterministic DDIM-style sampling (variance=0)

        Returns:
            y_0_hat: [B, num_classes] - enhanced logits
        """
        device = y_cond.device
        B, num_classes = y_cond.shape

        # Start from pure Gaussian noise (standard conditional diffusion)
        y_t = torch.randn(B, num_classes, device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            t_norm = t.float() / (self.timesteps - 1)

            # Predict epsilon
            eps_pred = self.model(
                y=y_t,
                t=t_norm,
                y_cond=y_cond,
                x=x_cond,
            )  # [B, num_classes]

            # scalar schedules, [B, 1] for broadcasting
            alpha_t     = self.alphas[t].unsqueeze(-1)           # [B, 1]
            alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)   # [B, 1]
            beta_t      = self.betas[t].unsqueeze(-1)            # [B, 1]

            if step > 0:
                alpha_bar_prev = self.alphas_cumprod[t - 1].unsqueeze(-1)  # [B,1]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)  # conceptually at t=-1

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            y0_pred = (y_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

            if step > 0:
                # posterior mean coefficients
                coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
                coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)

                mean = coef1 * y0_pred + coef2 * y_t  # [B, num_classes]

                if ddim:
                    # Deterministic DDIM-style update
                    y_t = mean
                else:
                    # Sample from posterior
                    var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    std = torch.sqrt(var)
                    z = torch.randn_like(y_t)
                    y_t = mean + std * z
            else:
                # At t=0 just take y0_pred
                y_t = y0_pred

        # y_t ~ y_0_hat
        return y_t


# -----------------------
# Args
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain logits diffusion model on ImageNet-32 (100-class subset)"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--classifier_ckpt", type=str, required=True, help="Path to pretrained classifier checkpoint")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--guiding_noise_level", type=float, default=0.3)
    parser.add_argument('--corruption_type', type=str, default='gaussian_blur', 
                        choices=['pixel_noise', 'gaussian_blur'], 
                        help='Choose "pixel_noise" or "gaussian_blur" (default)')
    parser.add_argument('--blur_sigma', type=float, default=2.0, help='Standard deviation for Gaussian blur')

    # ImageNet-32 root + subset
    parser.add_argument("--imagenet32_root", type=str, required=True, help="Path to the root directory of ImageNet32 dataset")
    parser.add_argument("--subset_seed", type=int, default=42)

    return parser.parse_args()


# -----------------------
# Main training
# -----------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    os.makedirs(args.results_dir, exist_ok=True)

    # 1) Data: ImageNet-32 100-class subset
    train_loader, test_loader = create_imagenet32_100_dataloaders(
        batch_size=args.batch_size,
        image_size=32,
        root=args.imagenet32_root,
        num_subset_classes=args.num_classes,
        seed=args.subset_seed,
    )

    # 2) Classifier (frozen, ImageNet-32 ResNet)
    classifier = ImageNet32ResNet(num_classes=args.num_classes)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location="cpu"))
    classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    # 3) Logits model + diffusion wrapper
    logits_model = ConditionalModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        n_input_channels=3,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        num_heads=8,
    ).to(device)

    diffusion = LogitsDiffusion(
        logits_model=logits_model,
        timesteps=args.timesteps,
        beta_start=1e-4,
        beta_end=0.02,
    ).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # -----------------------
    # Training loop
    # -----------------------
    global_step = 0
    best_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(args.epochs):
        diffusion.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        running_loss = 0.0

        for images, _ in pbar:
            images = images.to(device)

            if args.corruption_type == 'pixel_noise':
                guiding_noise_level = args.guiding_noise_level
                noise_cond = torch.randn_like(images, device=device)
                mask = torch.bernoulli(torch.full_like(images, guiding_noise_level, device=device))
                noisy_images = images * (1 - mask) + noise_cond * mask
            elif args.corruption_type == 'gaussian_blur':
                # Hardcoded kernel_size=5, sigma from args
                noisy_images = TF.gaussian_blur(images, kernel_size=5, sigma=[args.blur_sigma, args.blur_sigma])

            # --- classifier logits (clean + noisy) ---
            with torch.no_grad():
                logits_clean = classifier(images)
                logits_noisy = classifier(noisy_images)

            y0 = logits_clean          # "clean" logits to denoise towards
            y_cond = logits_noisy      # noisy logits as condition
            x_cond = noisy_images      # noisy images as additional condition

            B = images.size(0)
            t = torch.randint(low=0, high=args.timesteps, size=(B,), device=device)

            eps_pred, noise = diffusion(
                y0=y0, y_cond=y_cond, x=x_cond, t=t
            )

            loss = loss_fn(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Epoch loss
        epoch_avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {epoch_avg_loss:.6f}")

        # ---- Save regular checkpoint ----
        ckpt_path = os.path.join(
            args.results_dir,
            f"epoch_{epoch+1:03d}_steps_{global_step:08d}.pt"
        )
        torch.save(
            {
                "diffusion_state_dict": diffusion.state_dict(),
                "logits_model_state_dict": logits_model.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
            },
            ckpt_path
        )

        # ---- Save BEST checkpoint ----
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_ckpt_path = os.path.join(
                args.results_dir,
                f"BEST_epoch_{epoch+1:03d}_loss_{best_loss:.6f}.pt"
            )

            torch.save(
                {
                    "diffusion_state_dict": diffusion.state_dict(),
                    "logits_model_state_dict": logits_model.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_loss": best_loss,
                },
                best_ckpt_path
            )

            print(f"⭐ New best checkpoint saved: {best_ckpt_path} with loss {best_loss:.6f}")

    print("Finished pretraining logits diffusion model on ImageNet-32.")


if __name__ == "__main__":
    main()
