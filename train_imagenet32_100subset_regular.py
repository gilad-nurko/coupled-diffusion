import os
import math
import argparse

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image
from tqdm import tqdm

from utils import ExponentialMovingAverage
from model_cifar10_regular import MNISTDiffusion  
from classifier_imagenet32_100subset import (
    ImageNet32ResNet,
    build_imagenet32_datasets,  
)


# --------- ImageNet-32 (100-class subset, NPZ) dataloaders ---------
def create_imagenet32_100_dataloaders(
    batch_size,
    image_size=32,              # kept for interface compatibility, not used internally
    num_workers=4,
    root="/mlspeech/data/gilad/imagenet32",
    num_subset_classes=100,
    seed=42,
):
    """
    Uses the SAME subset selection + transforms as in classifier_imagenet32_100subset.py
    via build_imagenet32_datasets, which now reads NPZ files:
      root/Imagenet32_train_npz/train_data_batch_1.npz ... _10.npz
      root/Imagenet32_val_npz/val_data.npz
    """
    train_dataset, test_dataset = build_imagenet32_datasets(
        root=root,
        num_subset_classes=num_subset_classes,
        seed=seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Training ImageNet-32 (100-class subset) diffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=150)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence', default=10)
    parser.add_argument('--no_clip', action='store_true',
                        help='set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--is_ddim', type=bool, default=True)
    parser.add_argument('--guiding_noise_level', type=float, default=0.3)
    parser.add_argument('--corruption_type', type=str, default='gaussian_blur', 
                        choices=['pixel_noise', 'gaussian_blur'], 
                        help='Choose "pixel_noise" or "gaussian_blur" (default)')
    parser.add_argument('--blur_sigma', type=float, default=2.0, 
                        help='Standard deviation for Gaussian blur')
    # imagenet32 NPZ root / seed
    parser.add_argument('--imagenet32_root', type=str, default='/mlspeech/data/gilad/imagenet32')
    parser.add_argument('--subset_seed', type=int, default=42)
    args = parser.parse_args()

    return args


def main(args):
    # keep your device choice, adjust if needed
    device = torch.device('cuda:6' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using device:", device)

    # ----- Dataloaders: ImageNet-32 (100-class subset) -----
    train_dataloader, test_dataloader = create_imagenet32_100_dataloaders(
        batch_size=args.batch_size,
        image_size=32,
        root=args.imagenet32_root,
        num_subset_classes=100,
        seed=args.subset_seed,
    )

    # ----- Load the classifier (ImageNet-32 100-subset) -----
    classifier = ImageNet32ResNet(num_classes=100)
    classifier_path = "./classifier/classifier_weights_imagenet32_100subset.pth"
    classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
    classifier.to(device)
    classifier.eval()  # Set classifier to evaluation mode

    # ----- Diffusion model (unchanged, just different data) -----
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=32,
        in_channels=3,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        guiding_noise_level=args.guiding_noise_level,
        is_ddim=args.is_ddim,
        corruption_type=args.corruption_type,
        blur_sigma=args.blur_sigma,
    ).to(device)

    # EMA setup
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy='cos'
    )
    loss_fn = nn.MSELoss(reduction='mean')

    # load checkpoint if provided
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    results_dir = "results_imagenet32_100subset_regular_gaussian_blur_kernel_5_sigma_2"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for j, (image, target) in enumerate(train_dataloader):
            image = image.to(device)
            noise = torch.randn_like(image).to(device)

            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1

            if j % args.log_freq == 0:
                print(
                    "Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(
                        epoch + 1,
                        args.epochs,
                        j,
                        len(train_dataloader),
                        loss.detach().cpu().item(),
                        scheduler.get_last_lr()[0],
                    )
                )

        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }
        torch.save(ckpt, f"{results_dir}/steps_{global_steps:08}.pt")

        # --- Evaluation every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            model_ema.eval()
            classifier.eval()
            all_predictions = []
            all_targets = []

            for batch_idx, (test_images, test_labels) in enumerate(
                tqdm(test_dataloader, desc="Evaluation")
            ):
                num_samples = test_images.size(0)
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                if args.corruption_type == 'pixel_noise':
                    noise_cond = torch.randn_like(test_images).to(device)
                    mask = torch.bernoulli(
                        torch.full(test_images.shape, args.guiding_noise_level, device=device)
                    )
                    x_cond = test_images * (1 - mask) + noise_cond * mask
                elif args.corruption_type == 'gaussian_blur':
                    # Hardcoded kernel_size=5, sigma from args
                    x_cond = TF.gaussian_blur(test_images, kernel_size=5, sigma=[args.blur_sigma, args.blur_sigma])

                # sampling
                samples = model_ema.module.sampling(
                    num_samples,
                    clipped_reverse_diffusion=not args.no_clip,
                    device=device,
                    x_cond=x_cond,
                )

                # classifier on samples (ImageNet32 classifier)
                samples_logits = classifier(samples)
                samples_probs = torch.softmax(samples_logits, dim=1)

                # Save some visualizations only for the first batch
                if batch_idx == 0:
                    x_cond_file_path = os.path.join(results_dir, f"epoch_{epoch + 1}_x_cond.png")
                    samples_file_path = os.path.join(results_dir, f"epoch_{epoch + 1}_samples.png")
                    save_image(x_cond[:36], x_cond_file_path, nrow=6)
                    save_image(samples[:36], samples_file_path, nrow=6)
                    prob_file_path = os.path.join(results_dir, f"epoch_{epoch + 1}_sample_probs.txt")
                    with open(prob_file_path, 'w') as f:
                        f.write(str(samples_probs[:36].detach().cpu().numpy()))

                predictions = torch.argmax(samples_probs, dim=1)
                all_predictions.append(predictions)
                all_targets.append(test_labels)

            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            correct_predictions = (all_predictions == all_targets).sum().item()
            accuracy = correct_predictions / all_targets.size(0)

            acc_file_path = os.path.join(
                results_dir,
                f"epoch_{epoch + 1}_test_accuracy.txt"
            )
            with open(acc_file_path, 'w') as f:
                f.write(f"Test Accuracy: {accuracy}\n")
            print(f"Epoch {epoch + 1} evaluation: Test Accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
