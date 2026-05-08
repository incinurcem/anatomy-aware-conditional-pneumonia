"""
train_classifier.py

Binary pneumonia classifier training script for RSNA CXR pipeline.

Updated features
----------------
- Reads default paths from configs/paths.yaml
- Uses preprocess outputs by default:
    data/processed_pre/train/images_png
- Supports full image or ROI training
- Patient-level target building from RSNA labels CSV
- BCEWithLogitsLoss or Focal Loss
- Validation metrics:
    accuracy, auc, precision, recall, f1, specificity, sensitivity
- Checkpoint saving and CSV logging
"""

import os
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import yaml
except ImportError:
    yaml = None


# =========================================================
# PATH / YAML HELPERS
# =========================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(yaml_path: Optional[str]) -> Dict[str, Any]:
    if yaml_path is None:
        return {}

    yaml_file = Path(yaml_path)
    if not yaml_file.is_absolute():
        yaml_file = get_project_root() / yaml_file

    if not yaml_file.exists():
        return {}

    if yaml is None:
        raise ImportError("PyYAML is required to load paths.yaml")

    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(d: Dict[str, Any], keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_default_paths(paths_cfg: Dict[str, Any]) -> Dict[str, str]:
    project_root = get_project_root()

    return {
        "csv_path": deep_get(paths_cfg, ["data", "train_labels_csv"], str(project_root / "data/rsna/stage_2_train_labels.csv")),
        "image_dir": deep_get(paths_cfg, ["data", "train_png_dir"], str(project_root / "data/processed_pre/train/images_png")),
        "roi_dir": deep_get(paths_cfg, ["roi", "nnunet_dir"], str(project_root / "data/roi_images")),
        "save_dir": deep_get(paths_cfg, ["classifier", "checkpoints_dir"], str(project_root / "checkpoints/classifier")),
        "output_dir": deep_get(paths_cfg, ["classifier", "outputs_dir"], str(project_root / "outputs/classifier")),
    }


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
    parser = argparse.ArgumentParser(description="Train pneumonia classifier")

    parser.add_argument("--paths_yaml", type=str, default="configs/paths.yaml")

    # first parse only yaml path
    temp_args, _ = parser.parse_known_args()
    paths_cfg = load_yaml_config(temp_args.paths_yaml)
    defaults = resolve_default_paths(paths_cfg)

    parser.add_argument("--csv_path", type=str, default=defaults["csv_path"])
    parser.add_argument("--image_dir", type=str, default=defaults["image_dir"])
    parser.add_argument("--roi_dir", type=str, default=defaults["roi_dir"])
    parser.add_argument("--use_roi", action="store_true")

    parser.add_argument("--id_col", type=str, default="patientId")
    parser.add_argument("--target_col", type=str, default="Target")
    parser.add_argument("--ext", type=str, default=".png")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=["simplecnn", "resnet18", "resnet34", "efficientnet_b0"],
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--loss_name", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default=defaults["save_dir"])
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"])

    parser.add_argument("--mean", type=float, default=0.485)
    parser.add_argument("--std", type=float, default=0.229)

    return parser.parse_args()


# =========================================================
# LOSSES
# =========================================================

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =========================================================
# DATA PREPARATION
# =========================================================

def build_rsna_dataframe(
    csv_path: str,
    id_col: str = "patientId",
    target_col: str = "Target",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if id_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {id_col}, {target_col}")

    df_cls = df.groupby(id_col)[target_col].max().reset_index()
    df_cls[id_col] = df_cls[id_col].astype(str)
    df_cls[target_col] = df_cls[target_col].astype(int)
    return df_cls


def filter_existing_images(
    df: pd.DataFrame,
    image_dir: str,
    ext: str,
    id_col: str,
) -> pd.DataFrame:
    exists_mask = []
    for pid in df[id_col].tolist():
        path = os.path.join(image_dir, f"{pid}{ext}")
        exists_mask.append(os.path.exists(path))
    return df.loc[exists_mask].reset_index(drop=True)


# =========================================================
# AUGMENTATIONS
# =========================================================

def get_train_transforms(img_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=7,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3,
        ),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])


# =========================================================
# DATASET
# =========================================================

class PneumoClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        roi_dir: Optional[str],
        id_col: str,
        target_col: str,
        transform=None,
        use_roi: bool = False,
        ext: str = ".png",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.roi_dir = roi_dir
        self.id_col = id_col
        self.target_col = target_col
        self.transform = transform
        self.use_roi = use_roi
        self.ext = ext

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_id: str) -> str:
        filename = f"{image_id}{self.ext}"
        if self.use_roi and self.roi_dir is not None:
            roi_path = os.path.join(self.roi_dir, filename)
            if os.path.exists(roi_path):
                return roi_path
        return os.path.join(self.image_dir, filename)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_id = str(row[self.id_col])
        label = float(row[self.target_col])

        image_path = self._resolve_path(image_id)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            image = torch.tensor(image, dtype=torch.float32)

        return {
            "image_id": image_id,
            "image": image,
            "target": torch.tensor([label], dtype=torch.float32),
        }


# =========================================================
# MODELS
# =========================================================

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)


def build_model(model_name: str, pretrained: bool = False, dropout: float = 0.3):
    if model_name == "simplecnn":
        return SimpleCNN(in_channels=1, dropout=dropout)

    try:
        import torchvision.models as models
    except Exception as e:
        raise ImportError("torchvision is required for resnet/efficientnet models") from e

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    if model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


# =========================================================
# METRICS
# =========================================================

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


# =========================================================
# TRAIN / VALIDATION
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp: bool = False):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        targets = batch["target"].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        targs = targets.detach().cpu().numpy().reshape(-1)

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_targets.extend(targs.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_binary_metrics(
        np.array(all_targets, dtype=np.int32),
        np.array(all_probs, dtype=np.float32),
    )
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    running_loss = 0.0
    all_ids = []
    all_targets = []
    all_probs = []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        targets = batch["target"].to(device, dtype=torch.float32)
        image_ids = batch["image_id"]

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        targs = targets.detach().cpu().numpy().reshape(-1)

        running_loss += loss.item() * images.size(0)
        all_ids.extend(image_ids)
        all_probs.extend(probs.tolist())
        all_targets.extend(targs.tolist())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets_np = np.array(all_targets, dtype=np.int32)
    all_probs_np = np.array(all_probs, dtype=np.float32)
    metrics = compute_binary_metrics(all_targets_np, all_probs_np, threshold=threshold)
    metrics["loss"] = epoch_loss

    pred_df = pd.DataFrame({
        "image_id": all_ids,
        "true_label": all_targets_np,
        "pred_prob": all_probs_np,
        "pred_label": (all_probs_np >= threshold).astype(np.int32),
    })
    return metrics, pred_df


# =========================================================
# SAVE / LOAD
# =========================================================

def save_checkpoint(path, model, optimizer, epoch, best_score, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
        "args": vars(args),
    }, path)


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CSV path: {args.csv_path}")
    print(f"[INFO] Image dir: {args.image_dir}")
    if args.use_roi:
        print(f"[INFO] ROI dir: {args.roi_dir}")

    df = build_rsna_dataframe(
        csv_path=args.csv_path,
        id_col=args.id_col,
        target_col=args.target_col,
    )
    df = filter_existing_images(df=df, image_dir=args.image_dir, ext=args.ext, id_col=args.id_col)

    if len(df) == 0:
        raise RuntimeError("No valid images found after filtering existing files.")

    print(f"[INFO] Total usable samples: {len(df)}")
    print(f"[INFO] Positive samples: {int(df[args.target_col].sum())}")
    print(f"[INFO] Negative samples: {int((df[args.target_col] == 0).sum())}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df[args.target_col],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_transform = get_train_transforms(args.img_size, args.mean, args.std)
    val_transform = get_val_transforms(args.img_size, args.mean, args.std)

    train_dataset = PneumoClassificationDataset(
        df=train_df,
        image_dir=args.image_dir,
        roi_dir=args.roi_dir,
        id_col=args.id_col,
        target_col=args.target_col,
        transform=train_transform,
        use_roi=args.use_roi,
        ext=args.ext,
    )
    val_dataset = PneumoClassificationDataset(
        df=val_df,
        image_dir=args.image_dir,
        roi_dir=args.roi_dir,
        id_col=args.id_col,
        target_col=args.target_col,
        transform=val_transform,
        use_roi=args.use_roi,
        ext=args.ext,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(model_name=args.model_name, pretrained=args.pretrained, dropout=args.dropout).to(device)

    num_pos = int(train_df[args.target_col].sum())
    num_neg = int((train_df[args.target_col] == 0).sum())

    if args.loss_name == "bce":
        pos_weight_value = num_neg / max(num_pos, 1)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[INFO] BCE pos_weight: {pos_weight_value:.4f}")
    else:
        criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="mean")
        print(f"[INFO] Focal Loss alpha={args.focal_alpha}, gamma={args.focal_gamma}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_auc = -1.0
    log_rows = []

    best_ckpt_path = os.path.join(args.save_dir, "best.pt")
    last_ckpt_path = os.path.join(args.save_dir, "last.pt")
    log_csv_path = os.path.join(args.output_dir, "train_log.csv")
    best_pred_csv_path = os.path.join(args.output_dir, "val_predictions_best.csv")

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
        )

        val_metrics, val_pred_df = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_specificity": train_metrics["specificity"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_specificity": val_metrics["specificity"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tp": val_metrics["tp"],
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)

        print(
            f"[TRAIN] loss={train_metrics['loss']:.4f} "
            f"auc={train_metrics['auc']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1']:.4f}"
        )
        print(
            f"[VAL]   loss={val_metrics['loss']:.4f} "
            f"auc={val_metrics['auc']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"specificity={val_metrics['specificity']:.4f}"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_auc, args)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_auc, args)
            val_pred_df.to_csv(best_pred_csv_path, index=False)
            print(f"[INFO] Best model updated. New best val_auc={best_auc:.4f}")

    print("\n[INFO] Training completed.")
    print(f"[INFO] Best val AUC: {best_auc:.4f}")
    print(f"[INFO] Best checkpoint: {best_ckpt_path}")
    print(f"[INFO] Last checkpoint: {last_ckpt_path}")
    print(f"[INFO] Train log: {log_csv_path}")
    print(f"[INFO] Best val predictions: {best_pred_csv_path}")


if __name__ == "__main__":
    main()