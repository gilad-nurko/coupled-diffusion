import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from model_y_cond_iterative_diffusion_true_noising_cifar100 import MNISTDiffusion
from classifier_imagenet32_100subset import (
    ImageNet32ResNet,
    build_imagenet32_datasets,
)
from utils import ExponentialMovingAverage


# # -------------------------------------------------------------------
# # ImageNet-32 normalization (set these to what you used for classifier)
# # -------------------------------------------------------------------
# IMAGENET32_MEAN = (0.4811, 0.4575, 0.4079)  
# IMAGENET32_STD  = (0.2604, 0.2532, 0.2682)  


def create_imagenet32_100_dataloaders(
    batch_size,
    image_size=32,      # kept for interface compatibility
    num_workers=4,
    root="/mlspeech/data/gilad/imagenet32",
    num_subset_classes=100,
    seed=42,
):
    """
    Uses build_imagenet32_datasets from classifier_imagenet32_100subset.py
    to get the 100-class subset (train/test) from NPZ files.
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
    parser = argparse.ArgumentParser(
        description="Training iterative MNISTDiffusion for ImageNet-32 (100-class subset, logits y-space)"
    )
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint path')
    parser.add_argument('--n_samples', type=int, default=36, help='sampling amount after every epoch')
    parser.add_argument('--model_base_dim', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=150)
    parser.add_argument('--model_ema_steps', type=int, default=10)
    parser.add_argument('--model_ema_decay', type=float, default=0.995)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--no_clip', action='store_true', help='use normal sampling method without clip x_0')
    parser.add_argument('--cpu', action='store_true', help='cpu training')

    parser.add_argument('--is_cond', type=bool, default=True)
    parser.add_argument('--is_y_cond', type=bool, default=True)

    parser.add_argument('--pre_training_epochs', type=int, default=10)
    parser.add_argument('--guiding_noise_level', type=float, default=0.3)
    parser.add_argument('--iteration_steps', type=int, default=5)
    parser.add_argument('--steps_within_iteration', type=int, default=30)

    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--mode', type=str, default="imagenet32")
    parser.add_argument('--pretrained_model_y_ckpt', type=str, default="")
    parser.add_argument('--corruption_type', type=str, default='pixel_noise', 
                        choices=['pixel_noise', 'gaussian_blur'], 
                        help='Choose "pixel_noise" or "gaussian_blur" ')
    parser.add_argument('--blur_sigma', type=float, default=2.0, 
                        help='Standard deviation for Gaussian blur')

    # ImageNet-32 specific
    parser.add_argument('--imagenet32_root', type=str, default='/mlspeech/data/gilad/imagenet32')
    parser.add_argument('--subset_seed', type=int, default=42)
    parser.add_argument(
        '--classifier_ckpt',
        type=str,
        default="/home/gilad/diffusion_EM/toy_problem_implementation/"
                "MNISTDiffusion/classifier/classifier_weights_imagenet32_100subset.pth",
    )

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda:5' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using device:", device)

    # --- Dataloaders (ImageNet-32 100-subset) ---
    train_dataloader, test_dataloader = create_imagenet32_100_dataloaders(
        batch_size=args.batch_size,
        image_size=32,
        root=args.imagenet32_root,
        num_subset_classes=args.num_classes,
        seed=args.subset_seed,
    )

    # --- Classifier (ImageNet-32) ---
    classifier = ImageNet32ResNet(num_classes=args.num_classes)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location='cpu'))
    classifier.to(device)
    classifier.eval()

    # --- Diffusion model (iterative scheme, logits y-space) ---
    if args.pretrained_model_y_ckpt == "":
        pretrained_ckpt = "./logits_diffusion_pretrain_imagenet32_100_guiding_noise_level_0.3/BEST_epoch_080_loss_0.104026.pt"
    else:
        pretrained_ckpt = args.pretrained_model_y_ckpt

    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=32,
        in_channels=3,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        classifier=classifier,
        is_cond=args.is_cond,
        is_y_cond=args.is_y_cond,
        guiding_noise_level=args.guiding_noise_level,
        y_initialization=700,              # kept from original
        inner_loop_jump_step=20,           # kept for compatibility
        is_ddim=True,
        iteration_steps=args.iteration_steps,
        num_classes=args.num_classes,
        mode=args.mode,                    # "imagenet32"
        device=device,
        pretrained_model_y_ckpt=pretrained_ckpt,
        corruption_type=args.corruption_type,
        blur_sigma=args.blur_sigma
    ).to(device)

    # --- EMA ---
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    pre_training_scheduler = OneCycleLR(
        optimizer, args.lr,
        total_steps=args.pre_training_epochs * len(train_dataloader),
        pct_start=0.25, anneal_strategy='cos'
    )

    scheduler = OneCycleLR(
        optimizer, args.lr,
        total_steps=args.epochs * len(train_dataloader) *
                    args.iteration_steps * args.steps_within_iteration,
        pct_start=0.25, anneal_strategy='cos'
    )

    loss_fn = nn.MSELoss(reduction='mean')

    # --- Optional checkpoint load ---
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    directory = (
        "/home/gilad/diffusion_EM/toy_problem_implementation/"
        "MNISTDiffusion/results_true_noising_y_cond_imagenet32_100_"
        "iterative_diffusion_full_x_full_y_150_steps_many_steps_within_iteration_noise_0.3_try2"
    )
    os.makedirs(directory, exist_ok=True)

    # -------- Pre-training: x-branch only --------
    print("Starting pre-training (x-branch only)...")
    for epoch in range(args.pre_training_epochs):
        model.train()
        for (image, _) in tqdm(
            train_dataloader, desc=f"Pretrain epoch {epoch+1}/{args.pre_training_epochs}"
        ):
            image = image.to(device)
            noise_x = torch.randn_like(image).to(device)

            pred_x_noise, _ = model(image, noise_x, is_x=True, pre_train=True)
            loss_x = loss_fn(pred_x_noise, noise_x)

            loss_x.backward()
            optimizer.step()
            optimizer.zero_grad()
            pre_training_scheduler.step()

        print(
            f"[Pretrain] Epoch[{epoch+1}/{args.pre_training_epochs}], "
            f"Loss X:{loss_x.item():.5f}, "
            f"lr:{pre_training_scheduler.get_last_lr()[0]:.5f}"
        )

    # -------- Main training loop --------
    print("Starting main training...")
    for epoch in range(args.epochs):
        model.train()
        for j, (image, _) in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Train epoch {epoch+1}/{args.epochs}"
        ):
            image = image.to(device)

            # Target = classifier logits on clean image
            with torch.no_grad():
                target = classifier(image).detach()    # [B, num_classes]

            if args.corruption_type == 'pixel_noise':
                noise_cond = torch.randn_like(image).to(device)
                mask = torch.bernoulli(
                    torch.full(image.shape, args.guiding_noise_level, device=image.device)
                )
                noisy_image = image * (1 - mask) + noise_cond * mask
            elif args.corruption_type == 'gaussian_blur':
                # Hardcoded kernel_size=5, sigma from args
                noisy_image = TF.gaussian_blur(image, kernel_size=5, sigma=[args.blur_sigma, args.blur_sigma])

            # Initial y_0 (logits space) and x_0
            y_0 = torch.zeros(image.size(0), args.num_classes, device=device)
            x_0 = model_ema.module.get_x_0(noisy_image, y_0)
            y_0 = model_ema.module.get_y_0(noisy_image, x_0).detach()

            for it in range(args.iteration_steps):
                for _ in range(args.steps_within_iteration):
                    noise_y = torch.randn_like(target).to(device)
                    noise_x = torch.randn_like(noisy_image).to(device)

                    # x branch: denoise image, conditioned on noisy_image + current y_0
                    pred_x_noise, _ = model(
                        image, noise_x,
                        is_x=True,
                        target=target,
                        pre_train=False,
                        x_cond=noisy_image,
                        y_cond=y_0,
                    )
                    loss_x = loss_fn(pred_x_noise, noise_x)

                    # y branch: logits diffusion, conditioned on current x_0
                    pred_y_noise, _ = model(
                        image, noise_y,
                        is_x=False,
                        target=target,
                        pre_train=False,
                        x_cond=x_0,
                        y_cond=None,
                    )
                    loss_y = loss_fn(pred_y_noise, noise_y)

                    total_loss = loss_x + loss_y
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # Recompute x_0 and y_0 using EMA model
                x_0 = model_ema.module.get_x_0(noisy_image, y_0).detach()
                y_0 = model_ema.module.get_y_0(noisy_image, x_0).detach()

            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1

            if j % args.log_freq == 0:
                print(
                    f"Epoch[{epoch+1}/{args.epochs}], Step[{j}/{len(train_dataloader)}], "
                    f"Loss Y:{loss_y.item():.5f}, Loss X:{loss_x.item():.5f}, "
                    f"Total Loss:{total_loss.item():.5f}, "
                    f"lr:{scheduler.get_last_lr()[0]:.5f}"
                )

        # Save checkpoint after each epoch.
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
        torch.save(ckpt, f"{directory}/steps_{global_steps:08}.pt")

        # -------- Test Accuracy Every 2 Epochs (two metrics) --------
        if (epoch + 1) % 2 == 0:
            model_ema.eval()
            classifier.eval()

            all_predictions_logits = []      # argmax over diffusion y_0 logits
            all_predictions_classifier = []  # argmax over classifier on generated x_0
            all_targets = []

            # # tensors for ImageNet-32 normalization (for classifier on generated x_0)
            # mean = torch.tensor(IMAGENET32_MEAN, device=device).view(1, 3, 1, 1)
            # std  = torch.tensor(IMAGENET32_STD,  device=device).view(1, 3, 1, 1)

            with torch.no_grad():
                for i, (test_images, test_labels) in enumerate(
                    tqdm(test_dataloader, desc="Eval")
                ):
                    num_samples = test_images.size(0)
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    if args.corruption_type == 'pixel_noise':
                        noise_cond = torch.randn_like(test_images).to(device)
                        mask = torch.bernoulli(
                            torch.full(test_images.shape, args.guiding_noise_level,
                                    device=test_images.device)
                        )
                        x_cond = test_images * (1 - mask) + noise_cond * mask
                    elif args.corruption_type == 'gaussian_blur':
                        x_cond = TF.gaussian_blur(test_images, kernel_size=5, sigma=[args.blur_sigma, args.blur_sigma])

                    # save a grid of conditional and sampled images for the first batch
                    if i == 0:
                        x_cond_file_path = os.path.join(
                            directory, f"epoch_{epoch+1}_x_cond.png"
                        )
                        samples_file_path = os.path.join(
                            directory, f"epoch_{epoch+1}_samples.png"
                        )
                        save_image(x_cond[:36], x_cond_file_path, nrow=6)

                    # sampling: returns x_0 (images in [0,1]) and y_0 (logits)
                    samples_x0, samples_logits = model_ema.module.sampling(
                        num_samples,
                        clipped_reverse_diffusion=not args.no_clip,
                        device=device,
                        x_cond=x_cond,
                    )

                    if i == 0:
                        save_image(samples_x0[:36], samples_file_path, nrow=6)
                        logits_file_path = os.path.join(
                            directory, f"epoch_{epoch+1}_sample_logits.txt"
                        )
                        with open(logits_file_path, 'w') as f:
                            f.write(str(samples_logits[:36].detach().cpu().numpy()))

                    # ----- Accuracy 1: argmax over diffusion logits y_0 -----
                    preds_logits = torch.argmax(samples_logits, dim=1)
                    all_predictions_logits.append(preds_logits)

                    # ----- Accuracy 2: classifier on generated images x_0 -----
                    # # clamp to [0,1], then normalize like ImageNet-32 train pipeline
                    # gen_imgs = samples_x0.clamp(0.0, 1.0)
                    # gen_imgs_norm = (gen_imgs - mean) / std
                    cls_logits = classifier(samples_x0)
                    preds_cls = torch.argmax(cls_logits, dim=1)

                    all_predictions_classifier.append(preds_cls)
                    all_targets.append(test_labels)

            # concat everything
            all_predictions_logits = torch.cat(all_predictions_logits)
            all_predictions_classifier = torch.cat(all_predictions_classifier)
            all_targets = torch.cat(all_targets)

            # accuracies
            correct_logits = (all_predictions_logits == all_targets).sum().item()
            correct_cls = (all_predictions_classifier == all_targets).sum().item()
            acc_logits = correct_logits / all_targets.size(0)
            acc_cls = correct_cls / all_targets.size(0)

            acc_file_path = os.path.join(
                directory, f"epoch_{epoch+1}_test_accuracy.txt"
            )
            with open(acc_file_path, 'w') as f:
                f.write(f"Test Accuracy (argmax diffusion logits y_0): {acc_logits}\n")
                f.write(f"Test Accuracy (classifier on generated x_0): {acc_cls}\n")

            print(
                f"[Eval epoch {epoch+1}] "
                f"Acc_logits(y_0): {acc_logits:.4f}, "
                f"Acc_classifier(x_0): {acc_cls:.4f}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
