"""
train_segmentation.py
#d
Training script for lung segmentation models in the pneumonia pipeline.

Features:
- Binary lung segmentation
- Grayscale CXR support
- Train/validation split
- Dice + BCE combined loss
- Metrics: Dice, IoU, Precision, Recall
- Best checkpoint saving
- CSV training log
- Simple built-in U-Net
- Easy integration with project-specific segmentation models later

Outputs:
checkpoints/segmentation/best.pt
checkpoints/segmentation/last.pt
outputs/segmentation/train_log.csv
"""

import os
import random
import argparse
from typing import Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================================
# REPRODUCIBILITY
# =========================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train lung segmentation model")

    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/segmentation/montgomery/images",
        help="Path to segmentation training images"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="data/segmentation/montgomery/masks",
        help="Path to segmentation masks"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet",
        choices=["unet"],
        help="Segmentation model name"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary mask metrics"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/segmentation",
        help="Checkpoint output directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/segmentation",
        help="CSV/log output directory"
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.485,
        help="Normalization mean"
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.229,
        help="Normalization std"
    )

    return parser.parse_args()


# =========================================================
# DATA UTILITIES
# =========================================================

def list_image_mask_pairs(image_dir: str, mask_dir: str) -> List[Dict[str, str]]:
    """
    Collect image-mask pairs based on identical filenames.
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_files = sorted(os.listdir(image_dir))
    pairs = []

    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.isfile(image_path):
            continue

        if not os.path.exists(mask_path):
            continue

        pairs.append({
            "image_id": os.path.splitext(fname)[0],
            "image_path": image_path,
            "mask_path": mask_path
        })

    if len(pairs) == 0:
        raise RuntimeError("No image-mask pairs found.")

    return pairs


# =========================================================
# AUGMENTATIONS
# =========================================================

def get_train_transforms(img_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=7,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2()
    ])


# =========================================================
# DATASET
# =========================================================

class LungSegmentationDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample['image_path']}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {sample['mask_path']}")

        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
            image = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()

        return {
            "image_id": sample["image_id"],
            "image": image.float(),
            "mask": mask
        }


# =========================================================
# MODEL: U-NET
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = nn.functional.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def build_model(model_name: str):
    if model_name == "unet":
        return UNet(in_channels=1, out_channels=1)
    raise ValueError(f"Unsupported model_name: {model_name}")


# =========================================================
# LOSSES
# =========================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =========================================================
# METRICS
# =========================================================

def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item())
    }


# =========================================================
# TRAIN / VALIDATION
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    amp: bool = False,
    threshold: float = 0.5
):
    model.train()

    running_loss = 0.0
    metric_sum = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0
    }

    num_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        masks = batch["mask"].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        metrics = compute_segmentation_metrics(logits, masks, threshold=threshold)

        running_loss += loss.item() * images.size(0)
        for k in metric_sum:
            metric_sum[k] += metrics[k]
        num_batches += 1

    epoch_loss = running_loss / len(loader.dataset)
    epoch_metrics = {k: metric_sum[k] / max(num_batches, 1) for k in metric_sum}
    epoch_metrics["loss"] = epoch_loss

    return epoch_metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    threshold: float = 0.5
):
    model.eval()

    running_loss = 0.0
    metric_sum = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0
    }
    num_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        masks = batch["mask"].to(device, dtype=torch.float32)

        logits = model(images)
        loss = criterion(logits, masks)

        metrics = compute_segmentation_metrics(logits, masks, threshold=threshold)

        running_loss += loss.item() * images.size(0)
        for k in metric_sum:
            metric_sum[k] += metrics[k]
        num_batches += 1

    epoch_loss = running_loss / len(loader.dataset)
    epoch_metrics = {k: metric_sum[k] / max(num_batches, 1) for k in metric_sum}
    epoch_metrics["loss"] = epoch_loss

    return epoch_metrics


# =========================================================
# CHECKPOINT
# =========================================================

def save_checkpoint(path, model, optimizer, epoch, best_score, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
        "args": vars(args)
    }, path)


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}")

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------
    samples = list_image_mask_pairs(args.image_dir, args.mask_dir)

    train_samples, val_samples = train_test_split(
        samples,
        test_size=args.val_size,
        random_state=args.seed
    )

    print(f"[INFO] Total samples: {len(samples)}")
    print(f"[INFO] Train samples: {len(train_samples)}")
    print(f"[INFO] Val samples: {len(val_samples)}")

    train_transform = get_train_transforms(args.img_size, args.mean, args.std)
    val_transform = get_val_transforms(args.img_size, args.mean, args.std)

    train_dataset = LungSegmentationDataset(train_samples, transform=train_transform)
    val_dataset = LungSegmentationDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # -----------------------------------------------------
    # MODEL / LOSS / OPTIMIZER
    # -----------------------------------------------------
    model = build_model(args.model_name).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # -----------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------
    best_dice = -1.0
    log_rows = []

    best_ckpt_path = os.path.join(args.save_dir, "best.pt")
    last_ckpt_path = os.path.join(args.save_dir, "last.pt")
    log_csv_path = os.path.join(args.output_dir, "train_log.csv")

    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp=(args.amp and device.type == "cuda"),
            threshold=args.threshold
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],

            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"]
        }
        log_rows.append(row)

        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)

        print(
            f"[TRAIN] loss={train_metrics['loss']:.4f} "
            f"dice={train_metrics['dice']:.4f} "
            f"iou={train_metrics['iou']:.4f} "
            f"precision={train_metrics['precision']:.4f} "
            f"recall={train_metrics['recall']:.4f}"
        )

        print(
            f"[VAL]   loss={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} "
            f"iou={val_metrics['iou']:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f}"
        )

        save_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_dice,
            args=args
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_dice,
                args=args
            )

            print(f"[INFO] Best model updated. New best val_dice={best_dice:.4f}")

    print("\n[INFO] Segmentation training completed.")
    print(f"[INFO] Best val Dice: {best_dice:.4f}")
    print(f"[INFO] Best checkpoint: {best_ckpt_path}")
    print(f"[INFO] Last checkpoint: {last_ckpt_path}")
    print(f"[INFO] Train log: {log_csv_path}")


if __name__ == "__main__":
    main()