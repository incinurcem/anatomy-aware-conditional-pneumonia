import os
import json
import math
import copy
import random
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# Utilities
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_condition_vector(vec_path: str) -> np.ndarray:
    if not isinstance(vec_path, str) or not vec_path.strip():
        raise ValueError(f"Geçersiz vector yolu: {vec_path}")

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Condition vector dosyası bulunamadı: {vec_path}")

    ext = os.path.splitext(vec_path)[1].lower()

    if ext == ".npy":
        vec = np.load(vec_path)
    elif ext == ".npz":
        data = np.load(vec_path)
        if "array" in data:
            vec = data["array"]
        else:
            keys = list(data.keys())
            if len(keys) == 0:
                raise ValueError(f"Boş npz dosyası: {vec_path}")
            vec = data[keys[0]]
    else:
        raise ValueError(f"Desteklenmeyen condition vector formatı: {vec_path}")

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)

    if vec.ndim != 1:
        raise ValueError(f"Condition vector 1D olmalı. Dosya: {vec_path}, shape={vec.shape}")

    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        raise ValueError(f"Condition vector içinde NaN/Inf var: {vec_path}")

    return vec


def get_image_column(input_mode: str) -> str:
    if input_mode == "plain":
        return "image_path"
    elif input_mode == "roi":
        return "roi_path"
    elif input_mode == "masked_roi":
        return "masked_roi_path"
    else:
        raise ValueError(f"Geçersiz input_mode: {input_mode}")


def infer_condition_dim_from_csv(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)
    if "condition_vector_path" not in df.columns:
        raise ValueError(f"{csv_path} içinde condition_vector_path kolonu yok.")

    df = df.dropna(subset=["condition_vector_path"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{csv_path} içinde geçerli condition_vector_path yok.")

    vec_path = str(df.iloc[0]["condition_vector_path"])
    vec = load_condition_vector(vec_path)
    return int(vec.shape[0])


def read_image_as_rgb(img_path: str) -> Image.Image:
    if not isinstance(img_path, str) or not img_path.strip():
        raise ValueError(f"Geçersiz görüntü yolu: {img_path}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {img_path}")

    img = Image.open(img_path).convert("RGB")
    return img


# =========================================================
# Dataset
# =========================================================
class ConditionalRSNADataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        input_mode: str = "plain",
        img_size: int = 224,
        is_train: bool = False,
        expected_condition_dim: Optional[int] = None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.input_mode = input_mode
        self.path_col = get_image_column(input_mode)
        self.is_train = is_train
        self.expected_condition_dim = expected_condition_dim

        required_cols = ["label", self.path_col, "condition_vector_path"]
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"{csv_path} içinde eksik sütunlar: {missing_cols}")

        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"{csv_path} boş kaldı. Gerekli kolonlarda veri yok.")

        self.transform = self.build_transform(img_size=img_size, is_train=is_train)

        # Condition dim doğrulama
        first_vec_path = str(self.df.iloc[0]["condition_vector_path"])
        first_vec = load_condition_vector(first_vec_path)
        self.detected_condition_dim = int(first_vec.shape[0])

        if self.expected_condition_dim is not None and self.detected_condition_dim != self.expected_condition_dim:
            raise ValueError(
                f"Condition dim uyuşmuyor. "
                f"Beklenen={self.expected_condition_dim}, "
                f"Bulunan={self.detected_condition_dim}, "
                f"Dosya={first_vec_path}"
            )

    @staticmethod
    def build_transform(img_size: int, is_train: bool):
        if is_train:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(7),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.485, 0.485],
                    std=[0.229, 0.229, 0.229]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.485, 0.485],
                    std=[0.229, 0.229, 0.229]
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_id = str(row["image_id"]) if "image_id" in self.df.columns else str(idx)
        img_path = str(row[self.path_col])
        vec_path = str(row["condition_vector_path"])
        label = int(row["label"])

        image = read_image_as_rgb(img_path)
        image = self.transform(image)

        vec = load_condition_vector(vec_path)

        if self.expected_condition_dim is not None and vec.shape[0] != self.expected_condition_dim:
            raise ValueError(
                f"Condition vector boyutu hatalı. "
                f"image_id={image_id}, Beklenen={self.expected_condition_dim}, "
                f"Bulunan={vec.shape[0]}, Dosya={vec_path}"
            )

        return {
            "image": image,
            "condition": torch.tensor(vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "image_id": image_id,
        }


# =========================================================
# Model
# =========================================================
def build_backbone(model_name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int, str]:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        feat_dim = net.fc.in_features
        backbone = nn.Sequential(*list(net.children())[:-1])

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        net = models.densenet121(weights=weights)
        feat_dim = net.classifier.in_features
        backbone = net.features

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features
        backbone = net.features

    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    return backbone, feat_dim, model_name


class ConditionalClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        condition_dim: int = 27,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone, self.feat_dim, self.backbone_name = build_backbone(model_name, pretrained)

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "resnet50":
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            return x

        elif self.backbone_name in ["densenet121", "efficientnet_b0"]:
            x = self.backbone(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
            return x

        else:
            raise ValueError(f"Bilinmeyen backbone: {self.backbone_name}")

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        img_feat = self.extract_image_features(image)
        cond_feat = self.condition_mlp(condition)
        feat = torch.cat([img_feat, cond_feat], dim=1)
        logits = self.classifier(feat)
        return logits


# =========================================================
# Metrics
# =========================================================
def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if not math.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not math.isnan(pr_auc) else None,
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(bal_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# Train / Eval
# =========================================================
def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_ids = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        conditions = batch["condition"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)
        image_ids = batch["image_id"]

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            use_amp = bool(train and scaler is not None and device.type == "cuda")

            if use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits = model(images, conditions)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, conditions)
                loss = criterion(logits, labels)

                if train and optimizer is not None:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labs = labels.detach().cpu().numpy().reshape(-1)

        total_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        all_ids.extend(list(image_ids))

    epoch_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = float(epoch_loss)

    pred_df = pd.DataFrame({
        "image_id": all_ids,
        "y_true": np.asarray(all_labels).astype(int),
        "y_prob": np.asarray(all_probs),
        "y_pred": (np.asarray(all_probs) >= 0.5).astype(int)
    })

    return metrics, pred_df


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device):
    use_cuda = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--input_mode", type=str, default="plain", choices=["plain", "roi", "masked_roi"])
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "densenet121", "efficientnet_b0"])

    # 0 veya negatif verirsen otomatik infer eder
    parser.add_argument("--condition_dim", type=int, default=0)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--save_last", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.condition_dim <= 0:
        args.condition_dim = infer_condition_dim_from_csv(args.train_csv)
        print(f"[INFO] condition_dim otomatik bulundu: {args.condition_dim}")

    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    train_ds = ConditionalRSNADataset(
        csv_path=args.train_csv,
        input_mode=args.input_mode,
        img_size=args.img_size,
        is_train=True,
        expected_condition_dim=args.condition_dim,
    )
    val_ds = ConditionalRSNADataset(
        csv_path=args.val_csv,
        input_mode=args.input_mode,
        img_size=args.img_size,
        is_train=False,
        expected_condition_dim=args.condition_dim,
    )
    test_ds = ConditionalRSNADataset(
        csv_path=args.test_csv,
        input_mode=args.input_mode,
        img_size=args.img_size,
        is_train=False,
        expected_condition_dim=args.condition_dim,
    )

    print(
        f"[INFO] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)} | "
        f"Condition dim={train_ds.detected_condition_dim} | input_mode={args.input_mode}"
    )

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, device)

    model = ConditionalClassifier(
        model_name=args.model_name,
        condition_dim=args.condition_dim,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    train_df = pd.read_csv(args.train_csv)
    pos_count = int((train_df["label"] == 1).sum())
    neg_count = int((train_df["label"] == 0).sum())
    pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    print(f"[INFO] pos_count={pos_count}, neg_count={neg_count}, pos_weight={pos_weight_value:.6f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, train_pred = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
        )

        val_metrics, val_pred = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None,
            train=False,
        )

        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        scheduler.step(val_auc)

        epoch_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_row)

        print("=" * 100)
        print(f"Epoch {epoch}/{args.epochs}")
        print("Train:", train_metrics)
        print("Val  :", val_metrics)

        if args.save_every_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_val_auc,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pth")
            )

            train_pred.to_csv(os.path.join(args.output_dir, "best_train_predictions.csv"), index=False)
            val_pred.to_csv(os.path.join(args.output_dir, "best_val_predictions.csv"), index=False)

            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_val_auc": best_val_auc,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "best_metrics.json")
            )

            print(f"[INFO] Yeni en iyi model kaydedildi. val_auc={best_val_auc:.6f}")

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

    if args.save_last:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            os.path.join(args.output_dir, "last_model.pth")
        )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    test_metrics, test_pred = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        scaler=None,
        train=False,
    )

    test_pred.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))

    final_summary = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": test_metrics,
        "condition_dim": args.condition_dim,
        "input_mode": args.input_mode,
        "model_name": args.model_name,
    }
    save_json(final_summary, os.path.join(args.output_dir, "final_summary.json"))

    print("\n" + "=" * 100)
    print("FINAL TEST METRICS")
    print(test_metrics)
    print("=" * 100)


if __name__ == "__main__":
    main()