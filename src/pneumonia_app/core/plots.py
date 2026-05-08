"""Matplotlib plot helpers."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
)

from .config import EXPERIMENTS, EXP_LABELS, COLORS, SEEDS

plt.rcParams["font.family"] = "DejaVu Sans"


# ============================================================
# Column detection helpers — robust against varying CSV schemas
# ============================================================
def _label_col(df):
    for c in ["label", "y_true", "true_label", "target", "y", "true"]:
        if c in df.columns:
            return c
    return None


def _prob_col(df):
    for c in ["prob", "y_prob", "probability", "score", "pred_prob"]:
        if c in df.columns:
            return c
    return None


def _yp(preds):
    """Return (y_true, y_prob) numpy arrays or (None, None) if columns missing."""
    if preds is None:
        return None, None
    yc = _label_col(preds)
    pc = _prob_col(preds)
    if yc is None or pc is None:
        return None, None
    return preds[yc].values, preds[pc].values


# ============================================================
# ROC OVERLAY
# ============================================================
def plot_roc_overlay(runs, threshold_mode="youden"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, pl in zip(axes, ["nnUNet", "Otsu"]):
        for exp in EXPERIMENTS:
            tprs, base_fpr, aucs = [], np.linspace(0, 1, 101), []
            for s in SEEDS:
                run = runs.get((pl, exp, s))
                if run is None:
                    continue
                y_true, y_prob = _yp(run.get("preds"))
                if y_true is None:
                    continue
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                aucs.append(auc(fpr, tpr))
                tprs.append(np.interp(base_fpr, fpr, tpr))
            if not tprs:
                continue
            ax.plot(base_fpr, np.mean(tprs, axis=0),
                    label=f"{EXP_LABELS[exp]} (AUC={np.mean(aucs):.3f})", lw=1.6)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_title(f"{pl} pipeline (3-seed mean)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("True Positive Rate")
    plt.tight_layout()
    return fig


# ============================================================
# PR OVERLAY
# ============================================================
def plot_pr_overlay(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, pl in zip(axes, ["nnUNet", "Otsu"]):
        for exp in EXPERIMENTS:
            precs, base_rec, aps = [], np.linspace(0, 1, 101), []
            for s in SEEDS:
                run = runs.get((pl, exp, s))
                if run is None:
                    continue
                y_true, y_prob = _yp(run.get("preds"))
                if y_true is None:
                    continue
                pr, rc, _ = precision_recall_curve(y_true, y_prob)
                aps.append(average_precision_score(y_true, y_prob))
                precs.append(np.interp(base_rec, rc[::-1], pr[::-1]))
            if not precs:
                continue
            ax.plot(base_rec, np.mean(precs, axis=0),
                    label=f"{EXP_LABELS[exp]} (AP={np.mean(aps):.3f})", lw=1.6)
        ax.set_xlabel("Recall")
        ax.set_title(f"{pl} pipeline (3-seed mean)")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Precision")
    plt.tight_layout()
    return fig


# ============================================================
# DELTA BARS
# ============================================================
def plot_delta_bars(p_auc):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "delta" not in p_auc.columns:
        ax.text(0.5, 0.5, "Need both pipelines", ha="center", va="center",
                transform=ax.transAxes)
        return fig
    df = p_auc.reset_index()
    labels = [EXP_LABELS.get(e, e) for e in df["experiment"]]
    deltas = df["delta"].values
    colors = [COLORS["good"] if d > 0 else COLORS["bad"] for d in deltas]
    bars = ax.bar(labels, deltas, color=colors, alpha=0.85, edgecolor="black", lw=0.6)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("ΔAUC (nnU-Net − Otsu)", fontsize=11)
    ax.set_title("Segmentation Contribution by Input Mode",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for b, d in zip(bars, deltas):
        offset = 0.0015 if d >= 0 else -0.002
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + offset,
                f"{d:+.4f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    return fig


# ============================================================
# CONFUSION GRID
# ============================================================
def plot_confusion_grid(runs, threshold_mode="youden"):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), sharex=True, sharey=True)
    for r, pl in enumerate(["nnUNet", "Otsu"]):
        for c, exp in enumerate(EXPERIMENTS):
            ax = axes[r, c]
            cms = []
            for s in SEEDS:
                run = runs.get((pl, exp, s))
                if run is None:
                    continue
                y_true, y_prob = _yp(run.get("preds"))
                if y_true is None:
                    continue
                thr = run["best_thr"] if threshold_mode == "youden" else 0.5
                yhat = (y_prob >= thr).astype(int)
                cms.append(confusion_matrix(y_true, yhat))
            if not cms:
                ax.set_axis_off()
                continue
            cm = np.mean(cms, axis=0).round().astype(int)
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Neg", "Pos"], fontsize=8)
            ax.set_yticklabels(["Neg", "Pos"], fontsize=8)
            if r == 0:
                ax.set_title(EXP_LABELS[exp], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{pl}\nTrue", fontsize=10)
            if r == 1:
                ax.set_xlabel("Predicted", fontsize=8)
    plt.tight_layout()
    return fig


# ============================================================
# METRIC BARS (uses agg DataFrame, not preds — no bug here)
# ============================================================
def plot_metric_bars(agg):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(EXPERIMENTS))
    w = 0.38
    for i, pl in enumerate(["nnUNet", "Otsu"]):
        sub = agg[agg.pipeline == pl].set_index("experiment").reindex(EXPERIMENTS)
        ax.bar(x + (i - 0.5) * w, sub["auc_mean"].values, w,
               yerr=sub["auc_std"].values, capsize=3,
               label=pl, color=COLORS[pl], alpha=0.85,
               edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([EXP_LABELS[e] for e in EXPERIMENTS],
                       rotation=15, ha="right")
    ax.set_ylim(0.78, 0.92)
    ax.set_ylabel("ROC-AUC (test set, 3-seed mean ± std)")
    ax.set_title("Test-Set AUC by Configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig