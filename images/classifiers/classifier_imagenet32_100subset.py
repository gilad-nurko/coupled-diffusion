import os
import random
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


class ImageNet32ResNet(nn.Module):
    """
    ImageNet-32 subset classifier:
      - ResNet-50 backbone 
      - 3x3 conv1 and no maxpool (better for 32x32)
      - Extra MLP head on top of features.
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.backbone = models.resnet50(weights=None)

        # Adapt first conv & remove maxpool for 32x32
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        # Replace the original FC with Identity; we build our own head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        # Optional: initialize last layer a bit smaller
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)   # [B, in_features]
        logits = self.head(feats)  # [B, num_classes]
        return logits


class ImageNet32NPZ(Dataset):
    """
    Dataset that reads ImageNet32 from NPZ files:

    - Train files:
        Imagenet32_train_npz/train_data_batch_1.npz
        ...
        Imagenet32_train_npz/train_data_batch_10.npz
    - Val file:
        Imagenet32_val_npz/val_data.npz

    Each NPZ is expected to have:
        'data'   : uint8 array of shape (N, 3072) or (N, 3*32*32)
        'labels' : int labels in [1..1000] (we'll shift to [0..999])
    """
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            # 10 training batches
            file_list = [
                os.path.join(self.root, "Imagenet32_train_npz", f"train_data_batch_{i}.npz")
                for i in range(1, 11)
            ]
        else:
            # Single validation file
            file_list = [
                os.path.join(self.root, "Imagenet32_val_npz", "val_data.npz")
            ]

        data_list = []
        labels_list = []

        for fname in file_list:
            print(f"Loading {fname} ...")
            entry = np.load(fname)
            imgs = entry["data"]     
            labels = entry["labels"]  

            # reshape to (N, 3, 32, 32) then to (N, 32, 32, 3) for PIL/ToTensor
            imgs = imgs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            data_list.append(imgs)
            labels_list.append(labels)

        self.data = np.concatenate(data_list, axis=0)
        # Original labels are 1..1000, shift to 0..999
        self.targets = np.concatenate(labels_list, axis=0) - 1  

        print(
            f"{'Train' if self.train else 'Val'} dataset loaded: "
            f"{self.data.shape[0]} samples, label min={self.targets.min()}, max={self.targets.max()}"
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]          # (32, 32, 3), uint8
        target = int(self.targets[idx])

        img = Image.fromarray(img)    # PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class RemappedSubset(Dataset):
    """
    Wrap a base dataset and:
      - keep only samples whose label is in keep_labels (original indices)
      - remap labels using old_to_new (e.g., 1000 -> 100 classes)
    Assumes base_dataset.__getitem__ returns (img, label) and base_dataset.targets exists.
    """
    def __init__(self, base_dataset, keep_labels, old_to_new):
        self.base = base_dataset
        self.keep_labels = set(keep_labels)
        self.old_to_new = old_to_new

        self.indices = [
            i for i, y in enumerate(self.base.targets)
            if y in self.keep_labels
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, old_label = self.base[base_idx]
        new_label = self.old_to_new[int(old_label)]
        return img, new_label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet32_root", type=str, required=True, help="Path to the root directory of ImageNet32 dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save checkpoints")
    return parser.parse_args()

def build_imagenet32_datasets(root, num_subset_classes=100, seed=12345):
    """
    root: path that contains:
        root/Imagenet32_train_npz/train_data_batch_1.npz ... _10.npz
        root/Imagenet32_val_npz/val_data.npz
    """
    # Standard ImageNet32 mean/std (for 32x32 version)
    IMAGENET32_MEAN = (0.4811, 0.4575, 0.4079)
    IMAGENET32_STD  = (0.2604, 0.2532, 0.2682)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET32_MEAN, IMAGENET32_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET32_MEAN, IMAGENET32_STD),
    ])

    # Load full train/val from NPZ
    train_full = ImageNet32NPZ(root=root, train=True, transform=train_transform)
    test_full  = ImageNet32NPZ(root=root, train=False, transform=test_transform)

    # We have labels in [0..999]; infer number of classes from unique labels
    all_labels = np.unique(train_full.targets)
    num_all_classes = all_labels.shape[0]
    assert num_all_classes >= num_subset_classes, \
        f"Dataset has only {num_all_classes} classes, need at least {num_subset_classes}"

    # ----- fixed random subset of classes -----
    rng = random.Random(seed)
    all_class_indices = list(all_labels)  # e.g. [0, 1, ..., 999]
    subset_original_labels = sorted(rng.sample(all_class_indices, num_subset_classes))

    # Mapping from original label (0..999) to new label (0..99)
    old_to_new = {old: new for new, old in enumerate(subset_original_labels)}

    print(f"Chosen {num_subset_classes} classes (fixed seed = {seed}):")

    # Wrap datasets with subset + label remap
    train_dataset = RemappedSubset(train_full, subset_original_labels, old_to_new)
    test_dataset  = RemappedSubset(test_full,  subset_original_labels, old_to_new)

    return train_dataset, test_dataset


def main():
    args = parse_args()

    imagenet32_root = args.imagenet32_root
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "classifier_weights_imagenet32_100subset.pth")

    # Build subset datasets (100 fixed classes)
    train_dataset, test_dataset = build_imagenet32_datasets(
        imagenet32_root, num_subset_classes=100, seed=42
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ImageNet32ResNet(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    epochs = 200
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # ---- Eval every epoch ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = correct / total
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Test Acc: {acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best acc {best_acc:.4f}, saved to {model_path}")

    print("Best test accuracy:", best_acc)


if __name__ == "__main__":
    main()
