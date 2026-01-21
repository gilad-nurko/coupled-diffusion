import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from model_y_cond_nested_diffusion_true_noising_cifar100 import MNISTDiffusion
from classifier_imagenet32_100subset import (
    ImageNet32ResNet,
    build_imagenet32_datasets,
)
from utils import ExponentialMovingAverage


# -------------------------
# ImageNet-32 dataloaders
# -------------------------
def create_imagenet32_100_dataloaders(
    batch_size,
    image_size=32,      # kept for interface compatibility
    num_workers=4,
    root="/mlspeech/data/gilad/imagenet32",
    num_subset_classes=100,
    seed=42,
):
    """
    ImageNet-32 (100-class subset) dataloaders using build_imagenet32_datasets.
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
        shuffle=True,     # keep same behaviour as CIFAR script
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested MNISTDiffusion for ImageNet-32 (100-class subset) with logits y-space"
    )
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=150)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequency', default=10)
    parser.add_argument('--no_clip', action='store_true', help='use normal sampling method without clip x_0')
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--is_cond', type=bool, default=True)
    parser.add_argument('--is_y_cond', type=bool, default=True)
    parser.add_argument('--pre_training_epochs', type=int, default=10)
    parser.add_argument('--inner_loop_jump_step', type=int, default=20)
    parser.add_argument('--y_initialization', type=int, default=0)
    parser.add_argument('--guiding_noise_level', type=float, default=0.3)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--mode', type=str, default="imagenet32")
    parser.add_argument('--pretrained_model_y_ckpt', type=str, default="")
    parser.add_argument('--corruption_type', type=str, default='pixel_noise', 
                        choices=['pixel_noise', 'gaussian_blur'], 
                        help='Choose "pixel_noise" or "gaussian_blur" (default)')
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
    device = torch.device('cuda:7' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using device:", device)

    # Dataloaders: ImageNet-32 100-subset
    train_dataloader, test_dataloader = create_imagenet32_100_dataloaders(
        batch_size=args.batch_size,
        image_size=32,
        root=args.imagenet32_root,
        num_subset_classes=args.num_classes,
        seed=args.subset_seed,
    )

    # Classifier on ImageNet-32 (logits)
    classifier = ImageNet32ResNet(num_classes=args.num_classes)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location='cpu'))
    classifier.to(device)
    classifier.eval()

    # Pretrained logits-diffusion ckpt
    pretrained_ckpt = (
        args.pretrained_model_y_ckpt
        or "./logits_diffusion_pretrain_imagenet32_100_guiding_noise_level_0.3/BEST_epoch_080_loss_0.104026.pt"
    )

    # Nested diffusion model in ImageNet32 + logits y-space
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
        y_initialization=args.y_initialization,
        inner_loop_jump_step=args.inner_loop_jump_step,
        is_ddim=True,
        num_classes=args.num_classes,
        mode=args.mode,                 # "imagenet32"
        device=device,
        pretrained_model_y_ckpt=pretrained_ckpt,
        corruption_type=args.corruption_type,
        blur_sigma=args.blur_sigma
    ).to(device)

    # EMA
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    pre_training_scheduler = OneCycleLR(
        optimizer, args.lr,
        total_steps=(args.pre_training_epochs) * len(train_dataloader),
        pct_start=0.25, anneal_strategy='cos'
    )

    total_steps_scheduler = args.epochs * len(train_dataloader) * (1 + args.timesteps // args.inner_loop_jump_step)
    scheduler = OneCycleLR(
        optimizer, args.lr,
        total_steps=total_steps_scheduler,
        pct_start=0.25, anneal_strategy='cos'
    )
    loss_fn = nn.MSELoss(reduction='mean')

    # Optional checkpoint load
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    directory = (
        "/home/gilad/diffusion_EM/toy_problem_implementation/"
        "MNISTDiffusion/results_nested_diffusion_imagenet32_100_logits_ddim_"
        f"{args.timesteps}_steps_noise_level_0.3_y_initialization_{args.y_initialization}"
    )
    os.makedirs(directory, exist_ok=True)

    # ------------------- Pre-training: x-branch only -------------------
    for epoch in range(args.pre_training_epochs):
        model.train()
        for (image, _) in tqdm(
            train_dataloader,
            desc=f"Pretrain Epoch {epoch+1}/{args.pre_training_epochs}"
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
            f"Pretrain Epoch[{epoch+1}/{args.pre_training_epochs}], "
            f"Loss X:{loss_x.item():.5f}, "
            f"lr:{pre_training_scheduler.get_last_lr()[0]:.5f}"
        )

    # ------------------- Main training loop -------------------
    for epoch in range(args.epochs):
        model.train()
        for j, (image, _) in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Train Epoch {epoch+1}/{args.epochs}"
        ):
            image = image.to(device)

            # target = classifier logits (clean)
            with torch.no_grad():
                target = classifier(image).detach()   # [B, num_classes]

            if args.corruption_type == 'pixel_noise':
                noise_image = torch.randn_like(image).to(device)
                mask = torch.bernoulli(
                    torch.full(image.shape, args.guiding_noise_level, device=device)
                )
                noisy_image = image * (1 - mask) + noise_image * mask
            elif args.corruption_type == 'gaussian_blur':
                # Hardcoded kernel_size=5, sigma from args
                noisy_image = TF.gaussian_blur(image, kernel_size=5, sigma=[args.blur_sigma, args.blur_sigma])

            # y_0 initialized as zeros in logits-space
            y_0 = torch.zeros(image.size(0), args.num_classes, device=device)

            # x_0 from EMA model
            x_0 = model_ema.module.get_x_0(noisy_image, y_0)

            # Nested inner loop over timesteps
            for i in range(args.timesteps - 1, -1, -args.inner_loop_jump_step):
                t = torch.tensor(
                    [
                        torch.randint(max(0, i + 1 - args.inner_loop_jump_step), i + 1, (1,)).item()
                        for _ in range(image.size(0))
                    ],
                    device=device,
                    dtype=torch.long
                )

                noise_y = torch.randn_like(target).to(device)
                noise_x = torch.randn_like(noisy_image).to(device)

                # x-branch
                pred_x_noise, _ = model(
                    image, noise_x, is_x=True, target=target,
                    given_t=t, x_cond=noisy_image, y_cond=y_0
                )
                loss_x = loss_fn(pred_x_noise, noise_x)

                # Early timesteps: refresh x_0 from EMA
                if (i + 1) <= args.y_initialization:
                    x_0 = model_ema.module.get_x_0(noisy_image, y_0).detach()

                # y-branch (logits)
                pred_y_noise, y_0_pred = model(
                    image, noise_y, is_x=False, target=target,
                    given_t=t, x_cond=x_0
                )
                y_0 = y_0_pred.detach()
                loss_y = loss_fn(pred_y_noise, noise_y)

                total_loss = loss_x + loss_y
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

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

        # Save checkpoint
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
        torch.save(ckpt, f"{directory}/steps_{global_steps:08}.pt")

        # ------------------- Eval every 2 epochs -------------------
        if (epoch + 1) % 2 == 0:
            model_ema.eval()
            classifier.eval()

            all_predictions = []
            all_predictions_clf = []
            all_targets = []

            for test_images, test_labels in tqdm(test_dataloader, desc="Eval"):
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                num_samples = test_images.size(0)

                # Nested sampling conditioned on normalized test images
                samples_x0, samples_logits = model_ema.module.sampling(
                    num_samples,
                    clipped_reverse_diffusion=not args.no_clip,
                    device=device,
                    cond=test_images
                )

                # logits -> argmax (no softmax needed)
                predictions = torch.argmax(samples_logits, dim=1)

                with torch.no_grad():
                    clf_logits = classifier(samples_x0)        # samples_x0 expected in same normalization domain
                    preds_clf = torch.argmax(clf_logits, dim=1)

                all_predictions.append(predictions)
                all_predictions_clf.append(preds_clf)
                all_targets.append(test_labels)

            all_predictions = torch.cat(all_predictions)
            all_predictions_clf = torch.cat(all_predictions_clf)
            all_targets = torch.cat(all_targets)

            correct_predictions = (all_predictions == all_targets).sum().item()
            accuracy = correct_predictions / all_targets.size(0)

            correct_clf = (all_predictions_clf == all_targets).sum().item()
            accuracy_clf = correct_clf / all_targets.size(0)

            prob_file_path = os.path.join(directory, f"epoch_{epoch+1}_test_accuracy.txt")
            with open(prob_file_path, 'w') as f:
                f.write(f"Test Accuracy (argmax over logits): {accuracy}\n")
                f.write(f"Test Accuracy (classifier on generated x0): {accuracy_clf}\n")

            print(
                f"[Eval] Epoch {epoch+1}: "
                f"diffusion_logits_acc={accuracy:.4f}, "
                f"classifier_on_x0_acc={accuracy_clf:.4f}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
