import os
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class RSNAClassifierDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_col: str = "image_path",
        label_col: str = "label",
        image_size: int = 224,
        mean: float = 0.485,
        std: float = 0.229,
    ):
        self.df = pd.read_csv(csv_path).copy()
        if image_col not in self.df.columns:
            raise ValueError(f"Column '{image_col}' not found in {csv_path}")
        if label_col not in self.df.columns:
            raise ValueError(f"Column '{label_col}' not found in {csv_path}")

        self.image_col = image_col
        self.label_col = label_col
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        label = int(row[self.label_col])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_path": image_path,
        }


def build_model(model_name: str = "resnet50", in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    model_name = model_name.lower()
    weights = None
    if pretrained:
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
        elif model_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_name == "resnet34":
        model = models.resnet34(weights=weights)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_name == "densenet121":
        model = models.densenet121(weights=weights)
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_weights(model: nn.Module, model_path: str, device: torch.device) -> nn.Module:
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    return model


def create_loader(
    csv_path: str,
    image_col: str,
    label_col: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: float,
    std: float,
) -> DataLoader:
    dataset = RSNAClassifierDataset(
        csv_path=csv_path,
        image_col=image_col,
        label_col=label_col,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs_all, labels_all, paths_all = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(np.uint8)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.tolist())
        paths_all.extend(list(batch["image_path"]))
    return np.asarray(labels_all), np.asarray(probs_all), paths_all


def bootstrap_metric(y_true: np.ndarray, y_prob: np.ndarray, fn, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        try:
            vals.append(float(fn(yt, yp)))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp + 1e-12))


def npv_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fn + 1e-12))


def ppv_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def positive_likelihood_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = specificity_score(y_true, y_pred)
    denom = max(1.0 - spec, 1e-12)
    return float(sens / denom)


def negative_likelihood_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = specificity_score(y_true, y_pred)
    denom = max(spec, 1e-12)
    return float((1.0 - sens) / denom)


def diagnostic_odds_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lr_plus = positive_likelihood_ratio(y_true, y_pred)
    lr_minus = negative_likelihood_ratio(y_true, y_pred)
    return float(lr_plus / max(lr_minus, 1e-12))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = specificity_score(y_true, y_pred)
    ppv = ppv_score(y_true, y_pred)
    npv = npv_score(y_true, y_pred)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    prevalence = np.mean(y_true)
    pred_pos_rate = np.mean(y_pred)
    youden_j = sens + spec - 1.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "precision": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "jaccard": float(jaccard_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "youden_j": float(youden_j),
        "lr_plus": float(positive_likelihood_ratio(y_true, y_pred)),
        "lr_minus": float(negative_likelihood_ratio(y_true, y_pred)),
        "diagnostic_odds_ratio": float(diagnostic_odds_ratio(y_true, y_pred)),
        "prevalence": float(prevalence),
        "predicted_positive_rate": float(pred_pos_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_prob_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    yp = np.clip(y_prob, 1e-8, 1 - 1e-8)
    return {
        "auroc": float(roc_auc_score(y_true, yp)),
        "auprc": float(average_precision_score(y_true, yp)),
        "brier": float(np.mean((yp - y_true) ** 2)),
        "nll": float(log_loss(y_true, yp, labels=[0, 1])),
        "prevalence": float(np.mean(y_true)),
    }


def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    best_thr, best_j = 0.5, -1e9
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.uint8)
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = specificity_score(y_true, y_pred)
        youden_j = sens + spec - 1.0
        rows.append({"threshold": float(thr), "sensitivity": float(sens), "specificity": float(spec), "youden_j": float(youden_j)})
        if youden_j > best_j:
            best_j, best_thr = youden_j, thr
    return float(best_thr), pd.DataFrame(rows)


def find_threshold_for_target(y_true: np.ndarray, y_prob: np.ndarray, target_metric: str, target_value: float) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    best_thr, best_gap = 0.5, float("inf")
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.uint8)
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = specificity_score(y_true, y_pred)
        value = sens if target_metric == "sensitivity" else spec
        gap = abs(value - target_value)
        rows.append({"threshold": float(thr), "sensitivity": float(sens), "specificity": float(spec), "target_metric_value": float(value), "gap": float(gap)})
        if gap < best_gap:
            best_gap, best_thr = gap, thr
    return float(best_thr), pd.DataFrame(rows)


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(y_prob, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == len(bins) - 2 else (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_id": i, "bin_lower": float(lo), "bin_upper": float(hi), "count": 0, "mean_confidence": np.nan, "fraction_positive": np.nan, "bin_accuracy": np.nan, "gap": np.nan})
            continue
        conf = float(np.mean(y_prob[mask]))
        frac_pos = float(np.mean(y_true[mask]))
        pred_label = (y_prob[mask] >= 0.5).astype(np.uint8)
        acc = float(np.mean(pred_label == y_true[mask]))
        gap = abs(conf - frac_pos)
        rows.append({"bin_id": i, "bin_lower": float(lo), "bin_upper": float(hi), "count": count, "mean_confidence": conf, "fraction_positive": frac_pos, "bin_accuracy": acc, "gap": float(gap)})
    return pd.DataFrame(rows)


def compute_ece_mce_ace(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> Dict[str, float]:
    bins_df = calibration_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    valid = bins_df["count"] > 0
    total = bins_df.loc[valid, "count"].sum()
    if total == 0:
        return {"ece": float("nan"), "mce": float("nan"), "ace": float("nan")}
    ece = float((bins_df.loc[valid, "gap"] * bins_df.loc[valid, "count"] / total).sum())
    mce = float(bins_df.loc[valid, "gap"].max())
    ace = float(bins_df.loc[valid, "gap"].mean())
    return {"ece": ece, "mce": mce, "ace": ace}


def compute_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    yp = np.clip(y_prob, 1e-8, 1 - 1e-8)
    logits = np.log(yp / (1 - yp)).reshape(-1, 1)
    try:
        clf = LogisticRegression(fit_intercept=True, solver="lbfgs", max_iter=1000)
        clf.fit(logits, y_true)
        slope = float(clf.coef_[0][0])
        intercept = float(clf.intercept_[0])
    except Exception:
        slope, intercept = float("nan"), float("nan")
    return {"calibration_slope": slope, "calibration_intercept": intercept}


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, n_groups: int = 10) -> Dict[str, float]:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p").reset_index(drop=True)
    groups = np.array_split(df, n_groups)
    hl = 0.0
    valid_groups = 0
    for g in groups:
        if len(g) == 0:
            continue
        obs = g["y"].sum()
        exp = g["p"].sum()
        n = len(g)
        if exp <= 0 or exp >= n:
            continue
        hl += ((obs - exp) ** 2) / (exp + 1e-12)
        hl += (((n - obs) - (n - exp)) ** 2) / ((n - exp) + 1e-12)
        valid_groups += 1
    dof = max(valid_groups - 2, 1)
    return {"hosmer_lemeshow_stat": float(hl), "hosmer_lemeshow_dof": int(dof)}


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(np.uint8)
    n = len(y_true)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    w = threshold / max(1.0 - threshold, 1e-12)
    return float((tp / n) - (fp / n) * w)


def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[Sequence[float]] = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    prevalence = np.mean(y_true)
    rows = []
    for thr in thresholds:
        nb_model = net_benefit(y_true, y_prob, thr)
        nb_all = prevalence - (1.0 - prevalence) * (thr / max(1.0 - thr, 1e-12))
        rows.append({
            "threshold": float(thr),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPRC = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_reliability_diagram(bins_df: pd.DataFrame, save_path: str):
    valid = bins_df["count"] > 0
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(bins_df.loc[valid, "mean_confidence"], bins_df.loc[valid, "fraction_positive"], marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event frequency")
    plt.title("Reliability Diagram")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_histogram(y_prob: np.ndarray, save_path: str):
    plt.figure(figsize=(6, 5))
    plt.hist(y_prob, bins=20)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Histogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(2), ["Negative", "Positive"])
    plt.yticks(np.arange(2), ["Negative", "Positive"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_decision_curve(dca_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(7, 5))
    plt.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], linestyle="--", label="Treat all")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_treat_none"], linestyle=":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
