import os
import json
import math
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    brier_score_loss,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================================
# Genel yardımcılar
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# Calibration: ECE
# =========================================================
def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    return float(ece)


# =========================================================
# Metrikler
# =========================================================
def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    if len(np.unique(y_true)) == 1:
        auc = np.nan
        pr_auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)   # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auc": float(auc) if not np.isnan(auc) else None,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(bal_acc),
        "brier_score": float(brier),
        "ece": float(ece),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# Dataset
# =========================================================
class RSNAClassifierDataset(Dataset):
    def __init__(self, csv_path, input_mode="plain", img_size=224, augment=False):
        self.df = pd.read_csv(csv_path).copy()
        self.input_mode = input_mode
        self.img_size = img_size
        self.augment = augment

        required_cols = ["label"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"{csv_path} içinde zorunlu sütun yok: {c}")

        if input_mode == "plain":
            if "image_path" not in self.df.columns:
                raise ValueError(f"{csv_path} içinde image_path sütunu yok")
            self.path_col = "image_path"

        elif input_mode == "masked":
            # önce masked_roi_path, yoksa roi_path
            if "masked_roi_path" in self.df.columns:
                self.path_col = "masked_roi_path"
            elif "roi_path" in self.df.columns:
                self.path_col = "roi_path"
            else:
                raise ValueError(f"{csv_path} içinde masked_roi_path ya da roi_path yok")

        elif input_mode == "roi":
            if "roi_path" not in self.df.columns:
                raise ValueError(f"{csv_path} içinde roi_path sütunu yok")
            self.path_col = "roi_path"

        else:
            raise ValueError("input_mode sadece plain, masked veya roi olabilir")

        self.df = self.df[self.df[self.path_col].notna()].reset_index(drop=True)

        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=7,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.4
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {path}")

        # grayscale ise 3 kanala çıkar
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Beklenmeyen image shape: {img.shape} - {path}")

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.path_col]
        label = float(row["label"])

        image = self._read_image(img_path)
        image = self.transform(image=image)["image"]

        sample = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": img_path,
            "image_id": row["image_id"] if "image_id" in row else str(idx),
        }
        return sample


# =========================================================
# Model
# =========================================================
class DenseNet121Binary(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


# =========================================================
# Train / Eval
# =========================================================
def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=False):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_paths = []
    all_ids = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()

        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        all_paths.extend(batch["path"])
        all_ids.extend(batch["image_id"])

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_probs, threshold=0.5)

    outputs_df = pd.DataFrame({
        "image_id": all_ids,
        "path": all_paths,
        "label": all_labels,
        "prob": all_probs,
        "pred": (np.array(all_probs) >= 0.5).astype(int),
    })

    return epoch_loss, metrics, outputs_df


# =========================================================
# Ana eğitim
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--input_mode", type=str, default="plain", choices=["plain", "masked", "roi"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_every_epoch", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    print(f"[INFO] train rows: {len(train_df)}")
    print(f"[INFO] val rows  : {len(val_df)}")
    print(f"[INFO] test rows : {len(test_df)}")

    train_dataset = RSNAClassifierDataset(
        csv_path=args.train_csv,
        input_mode=args.input_mode,
        img_size=args.img_size,
        augment=True,
    )
    val_dataset = RSNAClassifierDataset(
        csv_path=args.val_csv,
        input_mode=args.input_mode,
        img_size=args.img_size,
        augment=False,
    )
    test_dataset = RSNAClassifierDataset(
        csv_path=args.test_csv,
        input_mode=args.input_mode,
        img_size=args.img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # pos_weight
    train_labels = train_dataset.df["label"].values.astype(np.float32)
    pos_count = float((train_labels == 1).sum())
    neg_count = float((train_labels == 0).sum())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    print(f"[INFO] pos_count={pos_count}, neg_count={neg_count}, pos_weight={pos_weight.item():.4f}")

    model = DenseNet121Binary(pretrained=args.pretrained, dropout=args.dropout).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    history = []
    best_val_auc = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_loss, train_metrics, train_outputs = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
        )

        val_loss, val_metrics, val_outputs = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            train=False,
        )

        val_auc_for_sched = val_metrics["auc"] if val_metrics["auc"] is not None else 0.0
        scheduler.step(val_auc_for_sched)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_log)

        print(f"[TRAIN] loss={train_loss:.4f} auc={train_metrics['auc']} pr_auc={train_metrics['pr_auc']}")
        print(f"[VAL]   loss={val_loss:.4f} auc={val_metrics['auc']} pr_auc={val_metrics['pr_auc']}")

        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

        if args.save_every_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            )

        val_auc = val_metrics["auc"] if val_metrics["auc"] is not None else -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_model_path
            )
            val_outputs.to_csv(os.path.join(args.output_dir, "best_val_predictions.csv"), index=False)
            print(f"[INFO] best model updated @ epoch {epoch}")

    print("\n[INFO] training finished")
    print(f"[INFO] best_val_auc = {best_val_auc:.6f}")

    # best model yükle
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, test_outputs = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        scaler=None,
        train=False,
    )

    test_outputs.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))

    final_report = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val_metrics": ckpt["val_metrics"],
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    save_json(final_report, os.path.join(args.output_dir, "final_report.json"))

    print("\n========== TEST RESULTS ==========")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"\n[INFO] outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()