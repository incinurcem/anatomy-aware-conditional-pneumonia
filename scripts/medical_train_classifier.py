import os
import math
import json
import copy
import random
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

import matplotlib.pyplot as plt


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_gray_or_rgb(path, force_rgb=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    if force_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def get_best_input_path(row):
    for col in ["masked_roi_path", "roi_path", "image_path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return str(row[col]).strip(), col
    raise ValueError("Geçerli giriş görüntü yolu bulunamadı.")


# =========================================================
# Dataset
# =========================================================

class PneumoniaClassifierDataset(Dataset):
    def __init__(self, csv_path, img_size=224, train=False, use_tta=False, tta_index=0):
        self.df = pd.read_csv(csv_path).copy()
        self.img_size = img_size
        self.train = train
        self.use_tta = use_tta
        self.tta_index = tta_index

        if "label" not in self.df.columns:
            raise ValueError(f"'label' sütunu yok: {csv_path}")

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.08, contrast=0.08)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                 std=[0.229, 0.229, 0.229]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                 std=[0.229, 0.229, 0.229]),
        ])

        self.tta_transforms = [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                     std=[0.229, 0.229, 0.229]),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                     std=[0.229, 0.229, 0.229]),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=(5, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                     std=[0.229, 0.229, 0.229]),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=(-5, -5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                     std=[0.229, 0.229, 0.229]),
            ]),
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        img_path, used_col = get_best_input_path(row)

        img = read_gray_or_rgb(img_path, force_rgb=True)

        if self.train:
            x = self.train_transform(img)
        elif self.use_tta:
            x = self.tta_transforms[self.tta_index % len(self.tta_transforms)](img)
        else:
            x = self.eval_transform(img)

        y = torch.tensor(float(row["label"]), dtype=torch.float32)

        sample = {
            "image": x,
            "label": y,
            "image_id": str(row.get("image_id", idx)),
            "input_path": img_path,
            "input_source": used_col,
        }

        if "mask_path" in row and pd.notna(row["mask_path"]):
            sample["mask_path"] = str(row["mask_path"])

        if "gt_mask_path" in row and pd.notna(row["gt_mask_path"]):
            sample["gt_mask_path"] = str(row["gt_mask_path"])

        return sample


# =========================================================
# Model
# =========================================================

class MedicalClassifier(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, dropout=0.3):
        super().__init__()

        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            features = list(model.children())[:-1]
        elif backbone == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            features = list(model.children())[:-1]
        elif backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            features = list(model.children())[:-1]
        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            in_features = model.classifier[1].in_features
            self.backbone = model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1)
            )
            self.model_name = backbone
            return
        else:
            raise ValueError(f"Desteklenmeyen backbone: {backbone}")

        self.backbone = nn.Sequential(*features)
        self.pool = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
        self.model_name = backbone

    def forward(self, x):
        feat = self.backbone(x)
        if self.model_name.startswith("efficientnet"):
            feat = self.pool(feat).flatten(1)
        else:
            feat = feat.flatten(1)
        logit = self.classifier(feat)
        return logit


# =========================================================
# Metrics
# =========================================================

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)   # sensitivity
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

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

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


def binary_nll(y_true, y_prob, eps=1e-8):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_bins(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc = []
    bin_conf = []
    bin_count = []

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_count.append(0)
        else:
            bin_acc.append(float(y_true[mask].mean()))
            bin_conf.append(float(y_prob[mask].mean()))
            bin_count.append(int(mask.sum()))

    return bins, np.array(bin_acc), np.array(bin_conf), np.array(bin_count)


def compute_calibration_metrics(y_true, y_prob, n_bins=15):
    bins, bin_acc, bin_conf, bin_count = calibration_bins(y_true, y_prob, n_bins=n_bins)
    total = max(1, bin_count.sum())

    gaps = np.abs(bin_acc - bin_conf)
    ece = float(np.sum((bin_count / total) * gaps))
    mce = float(np.max(gaps)) if len(gaps) > 0 else 0.0

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score(y_true, y_prob),
        "nll": binary_nll(y_true, y_prob),
        "bins": bins.tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
        "bin_count": bin_count.tolist(),
    }


def plot_reliability_diagram(calib_dict, save_path):
    bin_acc = np.array(calib_dict["bin_acc"])
    bin_conf = np.array(calib_dict["bin_conf"])
    bin_count = np.array(calib_dict["bin_count"])
    n_bins = len(bin_acc)

    xs = np.linspace(0, 1, n_bins, endpoint=False) + (1 / n_bins) / 2
    width = 1 / n_bins

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.bar(xs, bin_acc, width=width, alpha=0.7, edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def compute_entropy_and_variance(prob_samples):
    prob_samples = np.asarray(prob_samples, dtype=np.float32)  # [T, N]
    mean_prob = prob_samples.mean(axis=0)
    var_prob = prob_samples.var(axis=0)
    entropy = -(mean_prob * np.log(mean_prob + 1e-8) +
                (1 - mean_prob) * np.log(1 - mean_prob + 1e-8))
    return mean_prob, var_prob, entropy


def dice_iou_binary(pred_mask, gt_mask, eps=1e-8):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    dice = (2 * inter + eps) / (union + eps)

    iou_den = pred.sum() + gt.sum() - inter
    iou = (inter + eps) / (iou_den + eps)

    return float(dice), float(iou)


def evaluate_segmentation_if_available(df):
    if "mask_path" not in df.columns or "gt_mask_path" not in df.columns:
        return None

    dices = []
    ious = []

    for _, row in df.iterrows():
        pred_path = row["mask_path"]
        gt_path = row["gt_mask_path"]

        if pd.isna(pred_path) or pd.isna(gt_path):
            continue
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        d, i = dice_iou_binary(pred, gt)
        dices.append(d)
        ious.append(i)

    if len(dices) == 0:
        return None

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "num_samples": len(dices),
    }


# =========================================================
# Train / Eval
# =========================================================

def build_loader(csv_path, batch_size, img_size, train=False, num_workers=4, use_tta=False, tta_index=0):
    ds = PneumoniaClassifierDataset(
        csv_path=csv_path,
        img_size=img_size,
        train=train,
        use_tta=use_tta,
        tta_index=tta_index,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp=False):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        lab = y.detach().cpu().numpy().reshape(-1)

        total_loss += loss.item() * x.size(0)
        all_labels.extend(lab.tolist())
        all_probs.extend(prob.tolist())

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_probs)
    metrics["loss"] = float(epoch_loss)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    rows = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        labs = y.cpu().numpy().reshape(-1)

        total_loss += loss.item() * x.size(0)
        all_labels.extend(labs.tolist())
        all_probs.extend(probs.tolist())

        for i in range(len(probs)):
            rows.append({
                "image_id": batch["image_id"][i],
                "input_path": batch["input_path"][i],
                "input_source": batch["input_source"][i],
                "y_true": int(labs[i]),
                "y_prob": float(probs[i]),
                "y_pred": int(probs[i] >= 0.5),
            })

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_probs)
    metrics["loss"] = float(epoch_loss)

    calib = compute_calibration_metrics(all_labels, all_probs, n_bins=15)

    return metrics, calib, pd.DataFrame(rows), np.array(all_labels), np.array(all_probs)


def enable_dropout_in_eval(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_with_mc_dropout(model, loader, device, mc_runs=10):
    model.eval()
    enable_dropout_in_eval(model)

    prob_runs = []
    y_true_ref = None
    meta_rows = None

    for _ in tqdm(range(mc_runs), desc="mc_dropout"):
        all_probs = []
        all_labels = []
        rows = []

        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy().reshape(-1)

            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            all_probs.extend(probs.tolist())
            all_labels.extend(y.tolist())

            for i in range(len(probs)):
                rows.append({
                    "image_id": batch["image_id"][i],
                    "input_path": batch["input_path"][i],
                    "input_source": batch["input_source"][i],
                })

        prob_runs.append(np.array(all_probs))
        if y_true_ref is None:
            y_true_ref = np.array(all_labels)
            meta_rows = pd.DataFrame(rows)

    prob_runs = np.stack(prob_runs, axis=0)  # [T, N]
    mean_prob, var_prob, entropy = compute_entropy_and_variance(prob_runs)

    out_df = meta_rows.copy()
    out_df["y_true"] = y_true_ref
    out_df["mean_prob"] = mean_prob
    out_df["var_prob"] = var_prob
    out_df["entropy"] = entropy
    out_df["y_pred"] = (mean_prob >= 0.5).astype(int)
    out_df["correct"] = (out_df["y_true"] == out_df["y_pred"]).astype(int)

    return out_df


def predict_with_tta(model, csv_path, batch_size, img_size, num_workers, device, tta_runs=4):
    prob_runs = []
    y_true_ref = None
    meta_rows = None

    for tta_idx in tqdm(range(tta_runs), desc="tta"):
        _, loader = build_loader(
            csv_path=csv_path,
            batch_size=batch_size,
            img_size=img_size,
            train=False,
            num_workers=num_workers,
            use_tta=True,
            tta_index=tta_idx,
        )

        model.eval()
        all_probs = []
        all_labels = []
        rows = []

        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].cpu().numpy().reshape(-1)

                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

                all_probs.extend(probs.tolist())
                all_labels.extend(y.tolist())

                for i in range(len(probs)):
                    rows.append({
                        "image_id": batch["image_id"][i],
                        "input_path": batch["input_path"][i],
                        "input_source": batch["input_source"][i],
                    })

        prob_runs.append(np.array(all_probs))
        if y_true_ref is None:
            y_true_ref = np.array(all_labels)
            meta_rows = pd.DataFrame(rows)

    prob_runs = np.stack(prob_runs, axis=0)
    mean_prob, var_prob, entropy = compute_entropy_and_variance(prob_runs)

    out_df = meta_rows.copy()
    out_df["y_true"] = y_true_ref
    out_df["mean_prob"] = mean_prob
    out_df["var_prob"] = var_prob
    out_df["entropy"] = entropy
    out_df["y_pred"] = (mean_prob >= 0.5).astype(int)
    out_df["correct"] = (out_df["y_true"] == out_df["y_pred"]).astype(int)

    return out_df


# =========================================================
# Grad-CAM
# =========================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            score = logits[:, 0].sum()
        else:
            score = logits[:, class_idx].sum()

        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def save_gradcam_visualizations(model, csv_path, out_dir, img_size, device, num_samples=20):
    ensure_dir(out_dir)

    ds = PneumoniaClassifierDataset(csv_path=csv_path, img_size=img_size, train=False)
    indices = list(range(min(num_samples, len(ds))))

    if hasattr(model.backbone, "__getitem__"):
        target_layer = model.backbone[-1][-1].conv3 if isinstance(model.backbone[-1], nn.Sequential) else model.backbone[-1]
    else:
        raise ValueError("Grad-CAM için target layer bulunamadı.")

    gradcam = GradCAM(model, target_layer)

    model.eval()

    for idx in tqdm(indices, desc="gradcam"):
        sample = ds[idx]
        x = sample["image"].unsqueeze(0).to(device)
        image_id = sample["image_id"]
        raw_path = sample["input_path"]

        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        raw_img = cv2.resize(raw_img, (img_size, img_size))
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)

        cam = gradcam.generate(x)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img_rgb, 0.6, heatmap, 0.4, 0)

        save_path = os.path.join(out_dir, f"{image_id}_gradcam.png")
        cv2.imwrite(save_path, overlay)

    gradcam.remove()


# =========================================================
# Main
# =========================================================

def main(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    train_ds, train_loader = build_loader(
        args.train_csv, args.batch_size, args.img_size, train=True, num_workers=args.num_workers
    )
    val_ds, val_loader = build_loader(
        args.val_csv, args.batch_size, args.img_size, train=False, num_workers=args.num_workers
    )
    test_ds, test_loader = build_loader(
        args.test_csv, args.batch_size, args.img_size, train=False, num_workers=args.num_workers
    )

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = MedicalClassifier(
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout
    ).to(device)

    pos_count = int((train_ds.df["label"] == 1).sum())
    neg_count = int((train_ds.df["label"] == 0).sum())
    pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val_auc = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, amp=(args.amp and device.type == "cuda")
        )

        val_metrics, val_calib, val_pred_df, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "val_recall_sensitivity": val_metrics["recall_sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_ece": val_calib["ece"],
            "val_nll": val_calib["nll"],
        }
        history.append(row)

        print(json.dumps(row, indent=2, ensure_ascii=False))

        current_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else -1.0
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
            }, best_path)
            print("[INFO] Best model saved:", best_path)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.output_dir, "train_history.csv"), index=False)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\n[INFO] Loaded best model from:", best_path)

    test_metrics, test_calib, test_pred_df, y_true, y_prob = evaluate(model, test_loader, criterion, device)

    plot_reliability_diagram(test_calib, os.path.join(args.output_dir, "reliability_diagram.png"))
    test_pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    results = {
        "classification_metrics": test_metrics,
        "calibration_metrics": {
            k: v for k, v in test_calib.items()
            if k not in ["bins", "bin_acc", "bin_conf", "bin_count"]
        }
    }

    seg_metrics = evaluate_segmentation_if_available(test_ds.df)
    if seg_metrics is not None:
        results["segmentation_metrics"] = seg_metrics

    if args.mc_dropout_runs > 0:
        mc_df = predict_with_mc_dropout(model, test_loader, device, mc_runs=args.mc_dropout_runs)
        mc_df.to_csv(os.path.join(args.output_dir, "uncertainty_mc_dropout.csv"), index=False)

        results["uncertainty_mc_dropout"] = {
            "mean_entropy": float(mc_df["entropy"].mean()),
            "mean_variance": float(mc_df["var_prob"].mean()),
            "correct_entropy_mean": float(mc_df.loc[mc_df["correct"] == 1, "entropy"].mean()),
            "incorrect_entropy_mean": float(mc_df.loc[mc_df["correct"] == 0, "entropy"].mean()) if (mc_df["correct"] == 0).sum() > 0 else None,
            "correct_variance_mean": float(mc_df.loc[mc_df["correct"] == 1, "var_prob"].mean()),
            "incorrect_variance_mean": float(mc_df.loc[mc_df["correct"] == 0, "var_prob"].mean()) if (mc_df["correct"] == 0).sum() > 0 else None,
        }

    if args.tta_runs > 0:
        tta_df = predict_with_tta(
            model=model,
            csv_path=args.test_csv,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            device=device,
            tta_runs=args.tta_runs,
        )
        tta_df.to_csv(os.path.join(args.output_dir, "uncertainty_tta.csv"), index=False)

        results["uncertainty_tta"] = {
            "mean_entropy": float(tta_df["entropy"].mean()),
            "mean_variance": float(tta_df["var_prob"].mean()),
            "correct_entropy_mean": float(tta_df.loc[tta_df["correct"] == 1, "entropy"].mean()),
            "incorrect_entropy_mean": float(tta_df.loc[tta_df["correct"] == 0, "entropy"].mean()) if (tta_df["correct"] == 0).sum() > 0 else None,
            "correct_variance_mean": float(tta_df.loc[tta_df["correct"] == 1, "var_prob"].mean()),
            "incorrect_variance_mean": float(tta_df.loc[tta_df["correct"] == 0, "var_prob"].mean()) if (tta_df["correct"] == 0).sum() > 0 else None,
        }

    if args.gradcam_samples > 0:
        gradcam_dir = os.path.join(args.output_dir, "gradcam")
        save_gradcam_visualizations(
            model=model,
            csv_path=args.test_csv,
            out_dir=gradcam_dir,
            img_size=args.img_size,
            device=device,
            num_samples=args.gradcam_samples,
        )

    save_json(results, os.path.join(args.output_dir, "final_results.json"))
    print("\n[INFO] Final results saved to:", os.path.join(args.output_dir, "final_results.json"))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--mc_dropout_runs", type=int, default=10)
    parser.add_argument("--tta_runs", type=int, default=4)
    parser.add_argument("--gradcam_samples", type=int, default=20)

    args = parser.parse_args()
    main(args)