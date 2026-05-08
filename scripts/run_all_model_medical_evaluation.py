#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve


# ======================================================================================
# BASIC UTILITIES
# ======================================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ======================================================================================
# MODEL DEFINITIONS
# ======================================================================================

class PlainResNet50Binary(nn.Module):
    """
    Plain image classifier:
    image -> ResNet50 backbone -> dropout -> 1 logit
    """

    def __init__(self, pretrained=False, dropout=0.3):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None

        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x, cond=None):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        logit = self.fc(feat)
        return logit


class ConditionalResNet50Binary(nn.Module):
    """
    Conditional image classifier:
    image -> ResNet50 backbone
    condition vector -> MLP
    concat(image_feat, cond_feat) -> dropout -> 1 logit
    """

    def __init__(
        self,
        condition_dim: int,
        pretrained=False,
        dropout=0.3,
        cond_hidden_dim=128
    ):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None

        backbone = models.resnet50(weights=weights)
        img_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.condition_dim = condition_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, cond_hidden_dim),
            nn.BatchNorm1d(cond_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(img_dim + cond_hidden_dim, 1)

    def forward(self, x, cond=None):
        if cond is None:
            raise ValueError("Conditional model requires cond tensor, but cond=None was given.")

        img_feat = self.backbone(x)
        cond_feat = self.cond_mlp(cond)
        feat = torch.cat([img_feat, cond_feat], dim=1)
        feat = self.dropout(feat)
        logit = self.fc(feat)
        return logit


def build_model(
    model_name: str,
    conditional: bool,
    condition_dim: int = 0,
    pretrained: bool = False,
    dropout: float = 0.3,
):
    model_name = model_name.lower()
    if model_name != "resnet50":
        raise ValueError(f"Only resnet50 is supported in this evaluation script. Got: {model_name}")

    if conditional:
        if condition_dim <= 0:
            raise ValueError("Conditional model requested but condition_dim <= 0.")
        return ConditionalResNet50Binary(
            condition_dim=condition_dim,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        return PlainResNet50Binary(
            pretrained=pretrained,
            dropout=dropout
        )


def load_checkpoint_flexible(model: nn.Module, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            # maybe the dict itself is the state dict
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                raise RuntimeError(f"Could not load checkpoint: {checkpoint_path}\nError: {e}")
    else:
        model.load_state_dict(ckpt, strict=False)

    return model


# ======================================================================================
# DATASET
# ======================================================================================

NON_CONDITION_COLS = {
    "patientId",
    "label",
    "target",
    "class",
    "prob",
    "pred",
    "prediction",
    "fold",
    "split",
    "image_path",
    "plain_path",
    "roi_path",
    "masked_roi_path",
    "mask_path",
    "bbox_path",
    "lung_mask_path",
    "x",
    "y",
    "width",
    "height",
}


def resolve_image_col(df: pd.DataFrame, input_mode: str, explicit_image_col: Optional[str] = None) -> str:
    if explicit_image_col is not None and explicit_image_col in df.columns:
        return explicit_image_col

    candidates = []
    if input_mode == "plain":
        candidates = ["image_path", "plain_path"]
    elif input_mode == "roi":
        candidates = ["roi_path", "image_path"]
    elif input_mode == "masked_roi":
        candidates = ["masked_roi_path", "image_path"]
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not resolve image column for input_mode={input_mode}. "
        f"Available columns: {list(df.columns)}"
    )


def auto_detect_condition_cols(
    df: pd.DataFrame,
    label_col: str,
    image_col: str,
    explicit_condition_cols: Optional[List[str]] = None
) -> List[str]:
    if explicit_condition_cols is not None and len(explicit_condition_cols) > 0:
        missing = [c for c in explicit_condition_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Explicit condition columns not found in CSV: {missing}")
        return explicit_condition_cols

    excluded = set(NON_CONDITION_COLS)
    excluded.add(label_col)
    excluded.add(image_col)

    cond_cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cond_cols.append(c)

    return cond_cols


class RSNAMultiModeDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        input_mode: str,
        img_size: int,
        label_col: str = "label",
        image_col: Optional[str] = None,
        conditional: bool = False,
        condition_cols: Optional[List[str]] = None,
    ):
        self.df = pd.read_csv(csv_path).copy()

        self.input_mode = input_mode
        self.label_col = label_col
        self.image_col = resolve_image_col(self.df, input_mode=input_mode, explicit_image_col=image_col)

        if self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in {csv_path}")

        self.conditional = conditional
        self.condition_cols = []

        if self.conditional:
            self.condition_cols = auto_detect_condition_cols(
                self.df,
                label_col=self.label_col,
                image_col=self.image_col,
                explicit_condition_cols=condition_cols
            )
            if len(self.condition_cols) == 0:
                raise ValueError(
                    f"Conditional mode requested but no condition columns found in {csv_path}"
                )

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                 std=[0.229, 0.229, 0.229]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row[self.image_col]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.tf(img)

        y = int(row[self.label_col])

        if "patientId" in self.df.columns:
            patient_id = str(row["patientId"])
        else:
            patient_id = os.path.splitext(os.path.basename(img_path))[0]

        if self.conditional:
            cond = row[self.condition_cols].astype(np.float32).values
            cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
            cond_tensor = torch.tensor(cond, dtype=torch.float32)
        else:
            cond_tensor = torch.zeros(1, dtype=torch.float32)

        return {
            "image": img_tensor,
            "label": torch.tensor(y, dtype=torch.long),
            "patient_id": patient_id,
            "image_path": img_path,
            "cond": cond_tensor
        }


# ======================================================================================
# METRICS
# ======================================================================================

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    accuracy = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(precision),
        "npv": float(npv),
        "f1": float(f1),
    }


def find_best_threshold_by_f1(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None
    best_f1 = -1.0
    for thr in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best = m
    return best


def find_threshold_for_target_specificity(y_true, y_prob, target_spec=0.90):
    thresholds = np.linspace(0.0, 1.0, 1001)
    chosen = None
    best_gap = 1e9
    for thr in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=thr)
        gap = abs(m["specificity"] - target_spec)
        if m["specificity"] >= target_spec and gap < best_gap:
            best_gap = gap
            chosen = m
    return chosen


def find_threshold_for_target_sensitivity(y_true, y_prob, target_sens=0.90):
    thresholds = np.linspace(0.0, 1.0, 1001)
    chosen = None
    best_gap = 1e9
    for thr in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=thr)
        gap = abs(m["recall_sensitivity"] - target_sens)
        if m["recall_sensitivity"] >= target_sens and gap < best_gap:
            best_gap = gap
            chosen = m
    return chosen


def compute_ece_mce(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    rows = []

    for b in range(n_bins):
        mask = (bin_ids == b)
        if mask.sum() == 0:
            rows.append({
                "bin_id": b,
                "count": 0,
                "avg_confidence": np.nan,
                "empirical_accuracy": np.nan,
                "gap": np.nan
            })
            continue

        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        gap = abs(acc - conf)

        ece += (mask.sum() / len(y_true)) * gap
        mce = max(mce, gap)

        rows.append({
            "bin_id": b,
            "count": int(mask.sum()),
            "avg_confidence": conf,
            "empirical_accuracy": acc,
            "gap": float(gap)
        })

    return float(ece), float(mce), pd.DataFrame(rows)


# ======================================================================================
# PLOTS
# ======================================================================================

def plot_roc(y_true, y_prob, save_path, title="ROC Curve"):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr(y_true, y_prob, save_path, title="Precision-Recall Curve"):
    ap = average_precision_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion(y_true, y_prob, threshold, save_path, title_prefix="Confusion Matrix"):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{title_prefix} @ threshold={threshold:.3f}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_threshold_sweep(y_true, y_prob, save_path, title="Threshold Sweep"):
    thresholds = np.linspace(0.01, 0.99, 99)

    recalls = []
    specificities = []
    precisions = []
    f1s = []
    bals = []

    for thr in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=thr)
        recalls.append(m["recall_sensitivity"])
        specificities.append(m["specificity"])
        precisions.append(m["precision_ppv"])
        f1s.append(m["f1"])
        bals.append(m["balanced_accuracy"])

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, recalls, label="Recall/Sensitivity")
    plt.plot(thresholds, specificities, label="Specificity")
    plt.plot(thresholds, precisions, label="Precision/PPV")
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, bals, label="Balanced Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_reliability(y_true, y_prob, save_path, n_bins=10, title="Reliability Diagram"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_uncertainty_hist(df, save_path, title="Uncertainty Distribution (MC Dropout)"):
    plt.figure(figsize=(7, 5))
    plt.hist(df["std_prob"].values, bins=40)
    plt.xlabel("Predictive Std")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_entropy_hist(df, save_path, title="Predictive Entropy Distribution"):
    plt.figure(figsize=(7, 5))
    plt.hist(df["entropy"].values, bins=40)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_risk_coverage(df, save_path, title="Risk-Coverage Curve"):
    df = df.sort_values("std_prob", ascending=True).reset_index(drop=True)

    coverages = []
    risks = []

    n = len(df)
    for k in range(1, n + 1):
        subset = df.iloc[:k]
        coverage = k / n
        risk = subset["error"].mean()
        coverages.append(coverage)
        risks.append(risk)

    plt.figure(figsize=(7, 5))
    plt.plot(coverages, risks)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (Error Rate)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_uncertainty_vs_error(df, save_path, title="Uncertainty vs Error"):
    df = df.copy()
    correct = df[df["error"] == 0]["std_prob"].values
    wrong = df[df["error"] == 1]["std_prob"].values

    plt.figure(figsize=(7, 5))
    plt.hist(correct, bins=30, alpha=0.7, label="Correct")
    plt.hist(wrong, bins=30, alpha=0.7, label="Wrong")
    plt.xlabel("Predictive Std")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison_bar(df, metric_col, title, save_path, ascending=False):
    plot_df = df.sort_values(metric_col, ascending=ascending)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["experiment_name"], plot_df[metric_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================================================
# PREDICTION / MC DROPOUT
# ======================================================================================

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_plain_or_conditional(model, loader, device, conditional: bool):
    model.eval()

    rows = []
    all_probs = []
    all_labels = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()
        cond = batch["cond"].to(device)

        if conditional:
            logits = model(x, cond)
        else:
            logits = model(x)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        for i in range(len(y)):
            rows.append({
                "patientId": batch["patient_id"][i],
                "image_path": batch["image_path"][i],
                "label": int(y[i]),
                "prob": float(probs[i]),
                "pred": int(probs[i] >= 0.5),
            })

        all_probs.append(probs)
        all_labels.append(y)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return pd.DataFrame(rows), all_labels, all_probs


@torch.no_grad()
def mc_dropout_predict(model, loader, device, conditional: bool, mc_runs=20):
    model.eval()
    enable_dropout(model)

    all_rows = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()
        cond = batch["cond"].to(device)

        mc_probs = []
        for _ in range(mc_runs):
            if conditional:
                logits = model(x, cond)
            else:
                logits = model(x)

            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            mc_probs.append(probs)

        mc_probs = np.stack(mc_probs, axis=0)  # [T, B]
        mean_prob = mc_probs.mean(axis=0)
        std_prob = mc_probs.std(axis=0)
        entropy = -(mean_prob * np.log(mean_prob + 1e-8) + (1 - mean_prob) * np.log(1 - mean_prob + 1e-8))

        for i in range(len(y)):
            pred = 1 if mean_prob[i] >= 0.5 else 0
            error = int(pred != y[i])
            all_rows.append({
                "patientId": batch["patient_id"][i],
                "image_path": batch["image_path"][i],
                "label": int(y[i]),
                "mean_prob": float(mean_prob[i]),
                "std_prob": float(std_prob[i]),
                "entropy": float(entropy[i]),
                "pred": int(pred),
                "error": error
            })

    return pd.DataFrame(all_rows)


# ======================================================================================
# GRADCAM
# ======================================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input_, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, cond=None, conditional=False):
        self.model.zero_grad()

        if conditional:
            logits = self.model(x, cond)
        else:
            logits = self.model(x)

        score = logits[:, 0].sum()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        prob = torch.sigmoid(logits)[0, 0].item()
        return cam, prob


def overlay_cam_on_image(img_rgb, cam):
    h, w, _ = img_rgb.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return cam_resized, overlay


def run_gradcam_for_loader(
    model,
    loader,
    device,
    conditional,
    output_dir,
    max_samples=300
):
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "cams"))
    ensure_dir(os.path.join(output_dir, "overlays"))

    model.eval()
    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    rows = []
    count = 0

    for batch in loader:
        bs = batch["image"].shape[0]
        for i in range(bs):
            if count >= max_samples:
                break

            img_tensor = batch["image"][i:i+1].to(device)
            cond = batch["cond"][i:i+1].to(device)
            label = int(batch["label"][i].item())
            patient_id = batch["patient_id"][i]
            image_path = batch["image_path"][i]

            img_rgb = np.array(Image.open(image_path).convert("RGB"))

            cam, prob = gradcam.generate(
                img_tensor,
                cond=cond,
                conditional=conditional
            )
            cam_resized, overlay = overlay_cam_on_image(img_rgb, cam)

            cam_path = os.path.join(output_dir, "cams", f"{patient_id}_cam.npy")
            overlay_path = os.path.join(output_dir, "overlays", f"{patient_id}_overlay.png")

            np.save(cam_path, cam_resized)
            cv2.imwrite(overlay_path, overlay)

            rows.append({
                "patientId": patient_id,
                "label": label,
                "prob": float(prob),
                "cam_path": cam_path,
                "overlay_path": overlay_path,
                "image_path": image_path
            })

            count += 1

        if count >= max_samples:
            break

    gradcam_index_csv = os.path.join(output_dir, "gradcam_index.csv")
    pd.DataFrame(rows).to_csv(gradcam_index_csv, index=False)
    return gradcam_index_csv


# ======================================================================================
# GRADCAM-BBOX EXPLAINABILITY METRICS
# ======================================================================================

def binary_mask_from_bbox(img_h, img_w, x, y, w, h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))
    mask[y1:y2, x1:x2] = 1
    return mask


def combine_bboxes_to_mask(group, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for _, r in group.iterrows():
        if pd.isna(r["x"]) or pd.isna(r["y"]) or pd.isna(r["width"]) or pd.isna(r["height"]):
            continue
        mask |= binary_mask_from_bbox(img_h, img_w, r["x"], r["y"], r["width"], r["height"])
    return mask.astype(np.uint8)


def threshold_cam(cam, thr_mode="percentile", thr_value=85):
    cam = cam.astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    if thr_mode == "percentile":
        thr = np.percentile(cam, thr_value)
    else:
        thr = float(thr_value)

    return (cam >= thr).astype(np.uint8)


def dice_score(a, b):
    inter = (a & b).sum()
    return 2.0 * inter / (a.sum() + b.sum() + 1e-8)


def iou_score(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / (union + 1e-8)


def surface_points(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in contours:
        for p in c[:, 0, :]:
            pts.append(p[::-1])  # y, x
    return np.array(pts) if len(pts) > 0 else np.zeros((0, 2))


def hd95(a, b):
    pts_a = surface_points(a)
    pts_b = surface_points(b)

    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.nan

    d_ab = cdist(pts_a, pts_b)
    mins_a = d_ab.min(axis=1)
    mins_b = d_ab.min(axis=0)
    all_mins = np.concatenate([mins_a, mins_b], axis=0)
    return float(np.percentile(all_mins, 95))


def pointing_game(cam, gt_mask):
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    return int(gt_mask[peak] > 0)


def inside_lung_ratio(cam_bin, lung_mask):
    cam_area = cam_bin.sum()
    if cam_area == 0:
        return 0.0
    inside = (cam_bin & lung_mask).sum()
    return inside / cam_area


def outside_lung_ratio(cam_bin, lung_mask):
    cam_area = cam_bin.sum()
    if cam_area == 0:
        return 0.0
    outside = (cam_bin & (1 - lung_mask)).sum()
    return outside / cam_area


def evaluate_gradcam_bbox_overlap(
    gradcam_index_csv: str,
    annotations_csv: str,
    output_dir: str,
    lung_mask_dir: Optional[str] = None,
    thr_mode: str = "percentile",
    thr_value: float = 85.0
):
    ensure_dir(output_dir)

    grad_df = pd.read_csv(gradcam_index_csv)
    ann_df = pd.read_csv(annotations_csv)

    results = []

    for _, row in grad_df.iterrows():
        pid = row["patientId"]
        label = int(row["label"])
        cam = np.load(row["cam_path"])

        h, w = cam.shape[:2]

        gt_rows = ann_df[ann_df["patientId"].astype(str) == str(pid)]
        if label == 1 and len(gt_rows) > 0:
            gt_mask = combine_bboxes_to_mask(gt_rows, h, w)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        cam_bin = threshold_cam(cam, thr_mode=thr_mode, thr_value=thr_value)

        d = dice_score(cam_bin, gt_mask)
        i = iou_score(cam_bin, gt_mask)
        h95 = hd95(cam_bin, gt_mask)
        pg = pointing_game(cam, gt_mask)

        lung_ratio = np.nan
        outside_ratio = np.nan

        if lung_mask_dir is not None:
            lung_mask_path = os.path.join(lung_mask_dir, f"{pid}.png")
            if os.path.exists(lung_mask_path):
                lung_mask = cv2.imread(lung_mask_path, cv2.IMREAD_GRAYSCALE)
                lung_mask = cv2.resize(lung_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                lung_mask = (lung_mask > 0).astype(np.uint8)
                lung_ratio = inside_lung_ratio(cam_bin, lung_mask)
                outside_ratio = outside_lung_ratio(cam_bin, lung_mask)

        results.append({
            "patientId": pid,
            "label": label,
            "dice": float(d),
            "iou": float(i),
            "hd95": None if np.isnan(h95) else float(h95),
            "pointing_game_hit": int(pg),
            "inside_lung_ratio": None if pd.isna(lung_ratio) else float(lung_ratio),
            "outside_lung_ratio": None if pd.isna(outside_ratio) else float(outside_ratio)
        })

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, "gradcam_bbox_overlap_metrics.csv")
    out_df.to_csv(out_csv, index=False)

    summary = {
        "n_samples": int(len(out_df)),
        "mean_dice": float(out_df["dice"].fillna(0).mean()),
        "mean_iou": float(out_df["iou"].fillna(0).mean()),
        "mean_hd95": float(out_df["hd95"].dropna().mean()) if out_df["hd95"].notna().any() else None,
        "pointing_game_accuracy": float(out_df["pointing_game_hit"].mean()),
        "mean_inside_lung_ratio": float(out_df["inside_lung_ratio"].dropna().mean()) if out_df["inside_lung_ratio"].notna().any() else None,
        "mean_outside_lung_ratio": float(out_df["outside_lung_ratio"].dropna().mean()) if out_df["outside_lung_ratio"].notna().any() else None
    }

    write_json(summary, os.path.join(output_dir, "gradcam_bbox_overlap_summary.json"))
    return summary, out_csv


# ======================================================================================
# EXPERIMENT CONFIG
# ======================================================================================

@dataclass
class ExperimentConfig:
    experiment_name: str
    input_mode: str
    conditional: bool
    test_csv: str
    checkpoint: str
    output_dir: str
    model_name: str = "resnet50"
    label_col: str = "label"
    image_col: Optional[str] = None
    condition_cols: Optional[List[str]] = None
    condition_dim: Optional[int] = None
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    dropout: float = 0.3


def load_experiment_configs(config_json_path: str) -> List[ExperimentConfig]:
    with open(config_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    experiments = []
    for item in raw["experiments"]:
        experiments.append(ExperimentConfig(
            experiment_name=item["experiment_name"],
            input_mode=item["input_mode"],
            conditional=item["conditional"],
            test_csv=item["test_csv"],
            checkpoint=item["checkpoint"],
            output_dir=item["output_dir"],
            model_name=item.get("model_name", "resnet50"),
            label_col=item.get("label_col", "label"),
            image_col=item.get("image_col", None),
            condition_cols=item.get("condition_cols", None),
            condition_dim=item.get("condition_dim", None),
            img_size=item.get("img_size", 224),
            batch_size=item.get("batch_size", 16),
            num_workers=item.get("num_workers", 4),
            dropout=item.get("dropout", 0.3),
        ))
    return experiments


# ======================================================================================
# MAIN PER-EXPERIMENT PIPELINE
# ======================================================================================

def run_single_experiment(
    exp: ExperimentConfig,
    annotations_csv: str,
    lung_mask_dir: Optional[str],
    mc_runs: int,
    gradcam_max_samples: int,
    gradcam_thr_mode: str,
    gradcam_thr_value: float,
    device: str,
):
    print("=" * 120)
    print(f"[INFO] Running experiment: {exp.experiment_name}")
    print("=" * 120)

    ensure_dir(exp.output_dir)
    pred_dir = os.path.join(exp.output_dir, "predictions")
    med_dir = os.path.join(exp.output_dir, "medical_metrics")
    cal_dir = os.path.join(exp.output_dir, "calibration")
    unc_dir = os.path.join(exp.output_dir, "uncertainty")
    cam_dir = os.path.join(exp.output_dir, "gradcam")
    cam_eval_dir = os.path.join(exp.output_dir, "gradcam_bbox_eval")

    for d in [pred_dir, med_dir, cal_dir, unc_dir, cam_dir, cam_eval_dir]:
        ensure_dir(d)

    # dataset
    dataset = RSNAMultiModeDataset(
        csv_path=exp.test_csv,
        input_mode=exp.input_mode,
        img_size=exp.img_size,
        label_col=exp.label_col,
        image_col=exp.image_col,
        conditional=exp.conditional,
        condition_cols=exp.condition_cols,
    )

    loader = DataLoader(
        dataset,
        batch_size=exp.batch_size,
        shuffle=False,
        num_workers=exp.num_workers,
        pin_memory=True
    )

    resolved_condition_dim = 0
    if exp.conditional:
        if exp.condition_dim is not None:
            resolved_condition_dim = exp.condition_dim
        else:
            resolved_condition_dim = len(dataset.condition_cols)

    model = build_model(
        model_name=exp.model_name,
        conditional=exp.conditional,
        condition_dim=resolved_condition_dim,
        pretrained=False,
        dropout=exp.dropout
    )
    model = load_checkpoint_flexible(model, exp.checkpoint, device=device)
    model.to(device)

    # -----------------------------------------------------------------------------
    # 1) STANDARD PREDICTIONS
    # -----------------------------------------------------------------------------
    pred_df, y_true, y_prob = predict_plain_or_conditional(
        model=model,
        loader=loader,
        device=device,
        conditional=exp.conditional
    )
    pred_csv = os.path.join(pred_dir, "test_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    default_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    best_f1_metrics = find_best_threshold_by_f1(y_true, y_prob)
    sens_at_spec90 = find_threshold_for_target_specificity(y_true, y_prob, target_spec=0.90)
    spec_at_sens90 = find_threshold_for_target_sensitivity(y_true, y_prob, target_sens=0.90)

    med_summary = {
        "experiment_name": exp.experiment_name,
        "input_mode": exp.input_mode,
        "conditional": exp.conditional,
        "n_samples": int(len(pred_df)),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "default_threshold_0.5": default_metrics,
        "best_f1_threshold": best_f1_metrics,
        "sensitivity_at_specificity_0.90": sens_at_spec90,
        "specificity_at_sensitivity_0.90": spec_at_sens90,
    }

    write_json(med_summary, os.path.join(med_dir, "medical_metrics_summary.json"))

    key_rows = []
    row_05 = {"setting": "threshold_0.5", **default_metrics}
    key_rows.append(row_05)
    if best_f1_metrics is not None:
        key_rows.append({"setting": "best_f1", **best_f1_metrics})
    if sens_at_spec90 is not None:
        key_rows.append({"setting": "sens_at_spec90", **sens_at_spec90})
    if spec_at_sens90 is not None:
        key_rows.append({"setting": "spec_at_sens90", **spec_at_sens90})

    pd.DataFrame(key_rows).to_csv(os.path.join(med_dir, "key_threshold_metrics.csv"), index=False)

    plot_roc(y_true, y_prob, os.path.join(med_dir, "roc_curve.png"), title=f"{exp.experiment_name} ROC Curve")
    plot_pr(y_true, y_prob, os.path.join(med_dir, "pr_curve.png"), title=f"{exp.experiment_name} PR Curve")
    plot_confusion(y_true, y_prob, 0.5, os.path.join(med_dir, "confusion_matrix_thr_0_5.png"),
                   title_prefix=f"{exp.experiment_name} Confusion Matrix")
    plot_confusion(y_true, y_prob, best_f1_metrics["threshold"],
                   os.path.join(med_dir, "confusion_matrix_best_f1.png"),
                   title_prefix=f"{exp.experiment_name} Confusion Matrix")
    plot_threshold_sweep(y_true, y_prob, os.path.join(med_dir, "threshold_sweep.png"),
                         title=f"{exp.experiment_name} Threshold Sweep")

    # -----------------------------------------------------------------------------
    # 2) CALIBRATION
    # -----------------------------------------------------------------------------
    y_prob_clip = np.clip(y_prob, 1e-7, 1 - 1e-7)
    ece, mce, bins_df = compute_ece_mce(y_true, y_prob_clip, n_bins=10)
    brier = brier_score_loss(y_true, y_prob_clip)
    nll = log_loss(y_true, y_prob_clip)

    cal_summary = {
        "experiment_name": exp.experiment_name,
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "nll": float(nll)
    }

    write_json(cal_summary, os.path.join(cal_dir, "calibration_summary.json"))
    bins_df.to_csv(os.path.join(cal_dir, "calibration_bins.csv"), index=False)
    plot_reliability(y_true, y_prob_clip, os.path.join(cal_dir, "reliability_diagram.png"),
                     n_bins=10, title=f"{exp.experiment_name} Reliability Diagram")

    # -----------------------------------------------------------------------------
    # 3) UNCERTAINTY
    # -----------------------------------------------------------------------------
    unc_df = mc_dropout_predict(
        model=model,
        loader=loader,
        device=device,
        conditional=exp.conditional,
        mc_runs=mc_runs
    )
    unc_csv = os.path.join(unc_dir, "uncertainty_predictions.csv")
    unc_df.to_csv(unc_csv, index=False)

    corr = unc_df[["std_prob", "error"]].corr().iloc[0, 1]
    if pd.isna(corr):
        corr = 0.0

    unc_summary = {
        "experiment_name": exp.experiment_name,
        "mc_runs": int(mc_runs),
        "n_samples": int(len(unc_df)),
        "mean_std_prob": float(unc_df["std_prob"].mean()),
        "mean_entropy": float(unc_df["entropy"].mean()),
        "error_rate": float(unc_df["error"].mean()),
        "uncertainty_error_correlation": float(corr)
    }

    write_json(unc_summary, os.path.join(unc_dir, "uncertainty_summary.json"))
    plot_uncertainty_hist(unc_df, os.path.join(unc_dir, "uncertainty_hist.png"),
                          title=f"{exp.experiment_name} Uncertainty Distribution")
    plot_entropy_hist(unc_df, os.path.join(unc_dir, "entropy_hist.png"),
                      title=f"{exp.experiment_name} Entropy Distribution")
    plot_risk_coverage(unc_df, os.path.join(unc_dir, "risk_coverage_curve.png"),
                       title=f"{exp.experiment_name} Risk-Coverage Curve")
    plot_uncertainty_vs_error(unc_df, os.path.join(unc_dir, "uncertainty_vs_error.png"),
                              title=f"{exp.experiment_name} Uncertainty vs Error")

    # -----------------------------------------------------------------------------
    # 4) GRADCAM
    # -----------------------------------------------------------------------------
    gradcam_index_csv = run_gradcam_for_loader(
        model=model,
        loader=loader,
        device=device,
        conditional=exp.conditional,
        output_dir=cam_dir,
        max_samples=gradcam_max_samples
    )

    # -----------------------------------------------------------------------------
    # 5) GRADCAM vs BBOX EXPLAINABILITY
    # -----------------------------------------------------------------------------
    cam_summary, cam_metrics_csv = evaluate_gradcam_bbox_overlap(
        gradcam_index_csv=gradcam_index_csv,
        annotations_csv=annotations_csv,
        output_dir=cam_eval_dir,
        lung_mask_dir=lung_mask_dir,
        thr_mode=gradcam_thr_mode,
        thr_value=gradcam_thr_value
    )

    # -----------------------------------------------------------------------------
    # 6) MERGED SUMMARY ROW
    # -----------------------------------------------------------------------------
    summary_row = {
        "experiment_name": exp.experiment_name,
        "input_mode": exp.input_mode,
        "conditional": exp.conditional,
        "roc_auc": med_summary["roc_auc"],
        "pr_auc": med_summary["pr_auc"],
        "test_accuracy": med_summary["default_threshold_0.5"]["accuracy"],
        "test_balanced_accuracy": med_summary["default_threshold_0.5"]["balanced_accuracy"],
        "test_recall": med_summary["default_threshold_0.5"]["recall_sensitivity"],
        "test_specificity": med_summary["default_threshold_0.5"]["specificity"],
        "test_precision": med_summary["default_threshold_0.5"]["precision_ppv"],
        "test_npv": med_summary["default_threshold_0.5"]["npv"],
        "test_f1": med_summary["default_threshold_0.5"]["f1"],
        "best_f1_threshold": med_summary["best_f1_threshold"]["threshold"],
        "ece": cal_summary["ece"],
        "mce": cal_summary["mce"],
        "brier_score": cal_summary["brier_score"],
        "nll": cal_summary["nll"],
        "mean_uncertainty_std": unc_summary["mean_std_prob"],
        "mean_uncertainty_entropy": unc_summary["mean_entropy"],
        "uncertainty_error_correlation": unc_summary["uncertainty_error_correlation"],
        "gradcam_mean_dice": cam_summary["mean_dice"],
        "gradcam_mean_iou": cam_summary["mean_iou"],
        "gradcam_mean_hd95": cam_summary["mean_hd95"],
        "gradcam_pointing_game_accuracy": cam_summary["pointing_game_accuracy"],
        "gradcam_mean_inside_lung_ratio": cam_summary["mean_inside_lung_ratio"],
        "gradcam_mean_outside_lung_ratio": cam_summary["mean_outside_lung_ratio"],
        "pred_csv": pred_csv,
        "uncertainty_csv": unc_csv,
        "gradcam_index_csv": gradcam_index_csv,
        "gradcam_metrics_csv": cam_metrics_csv,
    }

    write_json(summary_row, os.path.join(exp.output_dir, "final_experiment_summary.json"))
    return summary_row


# ======================================================================================
# GLOBAL COMPARISON
# ======================================================================================

def build_global_comparison(all_rows: List[Dict], global_output_dir: str):
    ensure_dir(global_output_dir)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(global_output_dir, "all_models_comparison.csv"), index=False)

    # ranking tables
    metrics_desc = [
        "roc_auc",
        "pr_auc",
        "test_balanced_accuracy",
        "test_recall",
        "test_specificity",
        "test_precision",
        "test_f1",
        "gradcam_mean_dice",
        "gradcam_mean_iou",
        "gradcam_pointing_game_accuracy",
        "gradcam_mean_inside_lung_ratio",
    ]
    metrics_asc = [
        "ece",
        "mce",
        "brier_score",
        "nll",
        "mean_uncertainty_std",
        "mean_uncertainty_entropy",
        "gradcam_mean_hd95",
        "gradcam_mean_outside_lung_ratio",
    ]

    rank_rows = []
    for m in metrics_desc:
        temp = df.sort_values(m, ascending=False).reset_index(drop=True)
        for i, r in temp.iterrows():
            rank_rows.append({
                "metric": m,
                "rank": i + 1,
                "experiment_name": r["experiment_name"],
                "value": r[m]
            })

    for m in metrics_asc:
        temp = df.copy()
        temp[m] = pd.to_numeric(temp[m], errors="coerce")
        temp = temp.sort_values(m, ascending=True, na_position="last").reset_index(drop=True)
        for i, r in temp.iterrows():
            rank_rows.append({
                "metric": m,
                "rank": i + 1,
                "experiment_name": r["experiment_name"],
                "value": r[m]
            })

    pd.DataFrame(rank_rows).to_csv(os.path.join(global_output_dir, "all_models_ranking.csv"), index=False)

    # comparison plots
    plot_model_comparison_bar(df, "roc_auc", "Test ROC-AUC", os.path.join(global_output_dir, "test_roc_auc.png"))
    plot_model_comparison_bar(df, "pr_auc", "Test PR-AUC", os.path.join(global_output_dir, "test_pr_auc.png"))
    plot_model_comparison_bar(df, "test_balanced_accuracy", "Test Balanced Accuracy",
                              os.path.join(global_output_dir, "test_balanced_accuracy.png"))
    plot_model_comparison_bar(df, "test_recall", "Test Recall", os.path.join(global_output_dir, "test_recall.png"))
    plot_model_comparison_bar(df, "test_specificity", "Test Specificity",
                              os.path.join(global_output_dir, "test_specificity.png"))
    plot_model_comparison_bar(df, "test_f1", "Test F1", os.path.join(global_output_dir, "test_f1.png"))
    plot_model_comparison_bar(df, "ece", "ECE", os.path.join(global_output_dir, "ece.png"), ascending=True)
    plot_model_comparison_bar(df, "brier_score", "Brier Score",
                              os.path.join(global_output_dir, "brier_score.png"), ascending=True)
    plot_model_comparison_bar(df, "mean_uncertainty_std", "Mean Predictive Std",
                              os.path.join(global_output_dir, "mean_uncertainty_std.png"), ascending=True)
    plot_model_comparison_bar(df, "gradcam_mean_dice", "Grad-CAM vs BBox Dice",
                              os.path.join(global_output_dir, "gradcam_mean_dice.png"))
    plot_model_comparison_bar(df, "gradcam_mean_iou", "Grad-CAM vs BBox IoU",
                              os.path.join(global_output_dir, "gradcam_mean_iou.png"))
    plot_model_comparison_bar(df, "gradcam_pointing_game_accuracy", "Grad-CAM Pointing Game Accuracy",
                              os.path.join(global_output_dir, "gradcam_pointing_game_accuracy.png"))
    plot_model_comparison_bar(df, "gradcam_mean_inside_lung_ratio", "Grad-CAM Inside-Lung Ratio",
                              os.path.join(global_output_dir, "gradcam_inside_lung_ratio.png"))
    plot_model_comparison_bar(df, "gradcam_mean_hd95", "Grad-CAM HD95",
                              os.path.join(global_output_dir, "gradcam_mean_hd95.png"), ascending=True)

    # markdown summary
    lines = []
    lines.append("# 6-Model Medical Evaluation Summary\n")
    lines.append("## Classification\n")
    lines.append(df[[
        "experiment_name", "roc_auc", "pr_auc", "test_balanced_accuracy",
        "test_recall", "test_specificity", "test_precision", "test_f1"
    ]].sort_values("roc_auc", ascending=False).to_markdown(index=False))
    lines.append("\n## Calibration\n")
    lines.append(df[[
        "experiment_name", "ece", "mce", "brier_score", "nll"
    ]].sort_values("ece", ascending=True).to_markdown(index=False))
    lines.append("\n## Uncertainty\n")
    lines.append(df[[
        "experiment_name", "mean_uncertainty_std", "mean_uncertainty_entropy",
        "uncertainty_error_correlation"
    ]].sort_values("mean_uncertainty_std", ascending=True).to_markdown(index=False))
    lines.append("\n## Grad-CAM Explainability\n")
    lines.append(df[[
        "experiment_name", "gradcam_mean_dice", "gradcam_mean_iou",
        "gradcam_mean_hd95", "gradcam_pointing_game_accuracy",
        "gradcam_mean_inside_lung_ratio", "gradcam_mean_outside_lung_ratio"
    ]].sort_values("gradcam_mean_iou", ascending=False).to_markdown(index=False))

    with open(os.path.join(global_output_dir, "comparison_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[INFO] Global comparison written to:", global_output_dir)
    return df


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--annotations_csv", type=str, required=True)
    parser.add_argument("--lung_mask_dir", type=str, default=None)
    parser.add_argument("--global_output_dir", type=str, required=True)
    parser.add_argument("--mc_runs", type=int, default=20)
    parser.add_argument("--gradcam_max_samples", type=int, default=300)
    parser.add_argument("--gradcam_thr_mode", type=str, default="percentile", choices=["percentile", "absolute"])
    parser.add_argument("--gradcam_thr_value", type=float, default=85.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ensure_dir(args.global_output_dir)
    experiments = load_experiment_configs(args.config_json)

    all_rows = []
    for exp in experiments:
        row = run_single_experiment(
            exp=exp,
            annotations_csv=args.annotations_csv,
            lung_mask_dir=args.lung_mask_dir,
            mc_runs=args.mc_runs,
            gradcam_max_samples=args.gradcam_max_samples,
            gradcam_thr_mode=args.gradcam_thr_mode,
            gradcam_thr_value=args.gradcam_thr_value,
            device=device,
        )
        all_rows.append(row)

    build_global_comparison(all_rows, args.global_output_dir)
    print("[INFO] All evaluations completed successfully.")


if __name__ == "__main__":
    main()