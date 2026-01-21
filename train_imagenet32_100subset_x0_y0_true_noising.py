import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from utils import ExponentialMovingAverage
import os
import argparse
from tqdm import tqdm

from classifier_imagenet32_100subset import (
    ImageNet32ResNet,
    build_imagenet32_datasets,
)
from model_y_cond_our_working_x0_y0_true_noising_cifar100 import MNISTDiffusion


# ---------- ImageNet-32 (100-class subset) dataloaders ----------
def create_imagenet32_100_dataloaders(
    batch_size,
    image_size=32,          # kept for interface compatibility
    num_workers=4,
    root="/mlspeech/data/gilad/imagenet32",
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
        shuffle=True,   # keep behavior similar to your CIFAR100 script
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def label_to_noisy_probabilities(labels, num_classes, correct_prob=0.99):
    batch_size = len(labels)
    remaining_prob = 1.0 - correct_prob
    noise = torch.rand(batch_size, num_classes)
    noise[range(batch_size), labels] = 0
    noise_sum = noise.sum(dim=1, keepdim=True)
    noise = noise / noise_sum
    noise *= remaining_prob
    probabilities = noise.clone()
    probabilities[range(batch_size), labels] = correct_prob
    return probabilities


def label_to_one_hot(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion for ImageNet-32 (100-class subset)")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=500) # was 150
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequency', default=10)
    parser.add_argument('--no_clip', action='store_true', help='use normal sampling method without clip x_0')
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--is_cond', type=bool, default=True)
    parser.add_argument('--is_y_cond', type=bool, default=True)
    parser.add_argument('--iter_per_epoch', type=int, default=5)
    parser.add_argument('--guiding_noise_level', type=float, default=0.15)
    parser.add_argument('--y_loss_importance', type=float, default=0.1)
    parser.add_argument('--is_ddim', type=bool, default=False) # was True
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--mode', type=str, default="imagenet32")

    # ImageNet-32 specific
    parser.add_argument('--imagenet32_root', type=str, default='/mlspeech/data/gilad/imagenet32')
    parser.add_argument('--subset_seed', type=int, default=42)
    parser.add_argument('--classifier_path', type=str,
                        default="./classifier/classifier_weights_imagenet32_100subset.pth")
    parser.add_argument('--pretrained_model_y_ckpt', type=str,
                        default="./logits_diffusion_pretrain_imagenet32_100_guiding_noise_level_0.15/BEST_epoch_080_loss_0.102834.pt")
    parser.add_argument('--corruption_type', type=str, default='pixel_noise', 
                        choices=['pixel_noise', 'gaussian_blur'], 
                        help='Choose "pixel_noise" or "gaussian_blur" (default)')
    parser.add_argument('--blur_sigma', type=float, default=2.0, 
                        help='Standard deviation for Gaussian blur')

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda:7' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Using device:", device)

    # --------- Dataloaders for ImageNet-32 100-subset ---------
    train_dataloader, test_dataloader = create_imagenet32_100_dataloaders(
        batch_size=args.batch_size,
        image_size=32,
        root=args.imagenet32_root,
        num_subset_classes=args.num_classes,
        seed=args.subset_seed,
    )

    # --------- Load the ImageNet-32 classifier ---------
    classifier = ImageNet32ResNet(num_classes=args.num_classes)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location='cpu'))
    classifier.to(device)
    classifier.eval()

    # --------- Diffusion model (x/y alternating, now in 'imagenet32' mode) ---------
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=32,
        in_channels=3,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        classifier=classifier,
        is_cond=args.is_cond,
        is_y_cond=args.is_y_cond,
        is_ddim=args.is_ddim,
        mode=args.mode,                         # "imagenet32"
        guiding_noise_level=args.guiding_noise_level,
        num_classes=args.num_classes,
        device=device,
        pretrained_model_y_ckpt=args.pretrained_model_y_ckpt,
        corruption_type=args.corruption_type,
        blur_sigma=args.blur_sigma
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
        total_steps=args.epochs * len(train_dataloader) * args.iter_per_epoch,
        pct_start=0.25,
        anneal_strategy='cos',
    )
    loss_fn = nn.MSELoss(reduction='mean')

    # Load checkpoint if provided
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    # --------- Results directory (ImageNet-32 specific) ---------
    if args.is_y_cond:
        directory = (
            "/home/gilad/diffusion_EM/toy_problem_implementation/MNISTDiffusion/"
            f"results_true_noising_y_cond_imagenet32_100_x_0_y_0_ddpm_{args.timesteps}_"
            f"steps_gaussian_noise_0.15_weighted_loss_"
            f"{args.y_loss_importance}_on_y"
        )
    else:
        directory = "results_imagenet32_100"
    os.makedirs(directory, exist_ok=True)

    global_steps = 0

    for epoch in range(args.epochs):
        model.train()
        for j, (image, _) in enumerate(train_dataloader):
            image = image.to(device)
            # classifier logits as target (clean logits)
            target = classifier(image).detach()

            y_0 = torch.zeros_like(target, device=device)

            for i in range(args.iter_per_epoch):
                t = torch.randint(0, args.timesteps, (image.shape[0],), device=device)
                noise_x = torch.randn_like(image, device=device)
                noise_y = torch.randn_like(target, device=device)

                # x diffusion step
                pred_x, x_0 = model(
                    image,
                    noise_x,
                    is_x=True,
                    target=target,
                    cond=y_0,
                    t=t,
                )
                x_0 = x_0.detach()

                # y diffusion step
                pred_y, y_0 = model(
                    image,
                    noise_y,
                    is_x=False,
                    target=target,
                    cond=x_0,
                    t=t,
                )
                y_0 = y_0.detach()

                loss_x = loss_fn(pred_x, noise_x)
                loss_y = loss_fn(pred_y, noise_y)
                loss = (1 - args.y_loss_importance) * loss_x + args.y_loss_importance * loss_y

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_steps += 1

                if global_steps % args.model_ema_steps == 0:
                    model_ema.update_parameters(model)

                if j % args.log_freq == 0:
                    print(
                        f"Epoch[{epoch+1}/{args.epochs}], "
                        f"Step[{j}/{len(train_dataloader)}], "
                        f"loss: {loss.item():.5f}, "
                        f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    )

        # Save checkpoint after each epoch
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
        torch.save(ckpt, f"{directory}/steps_{global_steps:08}.pt")

        # --------- Test Accuracy Computation Every 5th Epoch ---------
        if (epoch + 1) % 5 == 0:
            model_ema.eval()
            all_predictions = []
            all_predictions_clf = []
            all_targets = []
            misclassified_saved = False  # save only one misclassified example

            for i, (test_images, test_labels) in enumerate(tqdm(test_dataloader)):
                num_samples = test_images.size(0)
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                if args.corruption_type == 'pixel_noise':
                    noise_cond = torch.randn_like(test_images, device=device)
                    mask = torch.bernoulli(
                        torch.full(test_images.shape, args.guiding_noise_level, device=device)
                    )
                    x_cond = test_images * (1 - mask) + noise_cond * mask
                elif args.corruption_type == 'gaussian_blur':
                    # Hardcoded kernel_size=5, sigma from args
                    x_cond = TF.gaussian_blur(test_images, kernel_size=5, sigma=[args.blur_sigma, args.blur_sigma])

                # Save corrupted images (first 36) for first batch
                if i == 0:
                    xcond_file_path = os.path.join(directory, f"epoch_{epoch+1}_xcond.png")
                    save_image(x_cond[:36], xcond_file_path, nrow=6)

                # Generate samples and probs with EMA model
                samples, samples_probs = model_ema.module.sampling(
                    num_samples,
                    clipped_reverse_diffusion=not args.no_clip,
                    device=device,
                    x_cond=x_cond,
                )

                # Save generated samples for first batch
                if i == 0:
                    samples_file_path = os.path.join(directory, f"epoch_{epoch+1}_samples.png")
                    save_image(samples[:36], samples_file_path, nrow=6)

                predictions = torch.argmax(samples_probs, dim=1)
                all_predictions.append(predictions)

                with torch.no_grad():
                    clf_logits = classifier(samples)  # samples are in the same normalization domain
                    clf_preds = torch.argmax(clf_logits, dim=1)

                all_predictions_clf.append(clf_preds)
                all_targets.append(test_labels)

                # Save one misclassified triplet (original, x_cond, sample)
                if not misclassified_saved:
                    for j in range(num_samples):
                        if predictions[j].item() != test_labels[j].item():
                            original_path = os.path.join(
                                directory, f"epoch_{epoch+1}_misclassified_original.png"
                            )
                            xcond_path = os.path.join(
                                directory, f"epoch_{epoch+1}_misclassified_xcond.png"
                            )
                            sample_path = os.path.join(
                                directory, f"epoch_{epoch+1}_misclassified_sample.png"
                            )

                            save_image(test_images[j], original_path)
                            save_image(x_cond[j], xcond_path)
                            save_image(samples[j], sample_path)

                            misclassified_saved = True
                            break

            all_predictions = torch.cat(all_predictions)
            all_predictions_clf = torch.cat(all_predictions_clf)
            all_targets = torch.cat(all_targets)

            correct_predictions = (all_predictions == all_targets).sum().item()
            accuracy_probs = correct_predictions / all_targets.size(0)

            correct_predictions_clf = (all_predictions_clf == all_targets).sum().item()
            accuracy_clf = correct_predictions_clf / all_targets.size(0)

            prob_file_path = os.path.join(directory, f"epoch_{epoch+1}_test_accuracy.txt")
            with open(prob_file_path, 'w') as f:
                f.write(f"Test Accuracy (argmax(samples_probs)): {accuracy_probs}\n")
                f.write(f"Test Accuracy (classifier on samples): {accuracy_clf}\n")

            print(
                f"Epoch {epoch+1}: "
                f"Test Accuracy probs={accuracy_probs:.4f}, "
                f"classifier={accuracy_clf:.4f}"
            )

    print(
        f"Final test accuracy at epoch {epoch+1}: "
        f"probs={accuracy_probs:.4f}, classifier={accuracy_clf:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
