#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# BASIC
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def minmax_norm(series, higher_better=True):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min, s_max = s.min(), s.max()

    if pd.isna(s_min) or pd.isna(s_max):
        out = pd.Series(np.zeros(len(s)), index=s.index)
    elif abs(s_max - s_min) < 1e-12:
        out = pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    else:
        out = (s - s_min) / (s_max - s_min)

    if not higher_better:
        out = 1.0 - out

    return out


def zscore_norm(series, higher_better=True):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mean = s.mean()
    std = s.std()

    if pd.isna(std) or abs(std) < 1e-12:
        out = pd.Series(np.zeros(len(s)), index=s.index)
    else:
        out = (s - mean) / std

    if not higher_better:
        out = -out

    return out


def assign_family(model_name: str):
    m = model_name.lower()
    if "masked" in m:
        return "Masked ROI"
    elif "roi" in m:
        return "ROI"
    else:
        return "Plain"


def family_marker(family: str):
    if family == "Plain":
        return "o"
    elif family == "ROI":
        return "s"
    else:
        return "^"


def family_color(family: str):
    if family == "Plain":
        return "#1f77b4"
    elif family == "ROI":
        return "#2ca02c"
    else:
        return "#d62728"


def nice_name(name: str):
    return str(name).replace("_", " ").replace("+", " + ")


# =============================================================================
# FIGURE 1
# PERFORMANCE VS CALIBRATION BUBBLE PLOT
# =============================================================================

def figure1_performance_vs_calibration(df, save_path):
    fig, ax = plt.subplots(figsize=(8.2, 6.5))

    bubble = 200 + 1000 * minmax_norm(df["ece"], higher_better=False)

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["roc_auc"],
            row["pr_auc"],
            s=float(bubble.loc[row.name]),
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.80,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(
            row["roc_auc"] + 0.001,
            row["pr_auc"] + 0.001,
            nice_name(row["experiment_name"]),
            fontsize=9
        )

    ax.set_xlabel("ROC-AUC", fontsize=12)
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_title("Figure 1. Performance–Calibration Trade-off Across Models", fontsize=13)
    ax.grid(alpha=0.25)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Plain', markerfacecolor=family_color("Plain"),
               markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker='s', color='w', label='ROI', markerfacecolor=family_color("ROI"),
               markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Masked ROI', markerfacecolor=family_color("Masked ROI"),
               markeredgecolor="black", markersize=10),
    ]
    ax.legend(handles=legend_elements, title="Input family", loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIGURE 2
# CLINICAL OPERATING POINT PLOT
# =============================================================================

def figure2_clinical_operating_point(df, save_path):
    fig, ax = plt.subplots(figsize=(8.2, 6.5))

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["test_specificity"],
            row["test_recall"],
            s=220,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(
            row["test_specificity"] + 0.002,
            row["test_recall"] + 0.002,
            f"{nice_name(row['experiment_name'])}\nF1={row['test_f1']:.3f}",
            fontsize=8.5
        )

    ax.axhline(df["test_recall"].mean(), linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axvline(df["test_specificity"].mean(), linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Specificity", fontsize=12)
    ax.set_ylabel("Recall / Sensitivity", fontsize=12)
    ax.set_title("Figure 2. Clinical Operating Behavior of the Six Models", fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIGURE 3
# CALIBRATION SUMMARY (NOT RAW RELIABILITY CURVES, BUT PAPER-READY SUMMARY)
# =============================================================================

def figure3_calibration_summary(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # left: ECE vs Brier
    ax = axes[0]
    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["ece"],
            row["brier_score"],
            s=200,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(row["ece"] + 0.0005, row["brier_score"] + 0.0005, nice_name(row["experiment_name"]), fontsize=8.5)

    ax.set_xlabel("Expected Calibration Error (↓)", fontsize=11)
    ax.set_ylabel("Brier Score (↓)", fontsize=11)
    ax.set_title("(a) Calibration Error Space", fontsize=12)
    ax.grid(alpha=0.25)

    # right: NLL vs ROC-AUC
    ax = axes[1]
    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["nll"],
            row["roc_auc"],
            s=200,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(row["nll"] + 0.002, row["roc_auc"] + 0.001, nice_name(row["experiment_name"]), fontsize=8.5)

    ax.set_xlabel("Negative Log-Likelihood (↓)", fontsize=11)
    ax.set_ylabel("ROC-AUC (↑)", fontsize=11)
    ax.set_title("(b) Calibration vs Discrimination", fontsize=12)
    ax.grid(alpha=0.25)

    fig.suptitle("Figure 3. Calibration Characteristics of the Six Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIGURE 4
# UNCERTAINTY SUMMARY
# =============================================================================

def figure4_uncertainty_summary(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # left
    ax = axes[0]
    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["mean_uncertainty_std"],
            row["uncertainty_error_correlation"],
            s=220,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(
            row["mean_uncertainty_std"] + 0.0005,
            row["uncertainty_error_correlation"] + 0.002,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    ax.set_xlabel("Mean Predictive Std (↓ if confident)", fontsize=11)
    ax.set_ylabel("Uncertainty–Error Correlation (↑)", fontsize=11)
    ax.set_title("(a) Error Awareness of Uncertainty", fontsize=12)
    ax.grid(alpha=0.25)

    # right
    ax = axes[1]
    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["mean_uncertainty_entropy"],
            row["roc_auc"],
            s=220,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(
            row["mean_uncertainty_entropy"] + 0.0005,
            row["roc_auc"] + 0.001,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    ax.set_xlabel("Mean Predictive Entropy", fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("(b) Uncertainty vs Predictive Quality", fontsize=12)
    ax.grid(alpha=0.25)

    fig.suptitle("Figure 4. Uncertainty Behavior Across Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIGURE 5
# EXPLAINABILITY-LOCALIZATION PLOT
# =============================================================================

def figure5_explainability_localization(df, save_path):
    fig, ax = plt.subplots(figsize=(8.4, 6.6))

    bubble = 180 + 900 * minmax_norm(df["gradcam_mean_inside_lung_ratio"], higher_better=True)

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        ax.scatter(
            row["gradcam_mean_iou"],
            row["gradcam_pointing_game_accuracy"],
            s=float(bubble.loc[row.name]),
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.82,
            edgecolors="black",
            linewidths=0.8
        )
        ax.text(
            row["gradcam_mean_iou"] + 0.001,
            row["gradcam_pointing_game_accuracy"] + 0.002,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    ax.set_xlabel("Grad-CAM / BBox IoU", fontsize=12)
    ax.set_ylabel("Pointing Game Accuracy", fontsize=12)
    ax.set_title("Figure 5. Explainability Localization Performance", fontsize=13)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIGURE 6
# COMPACT SUMMARY HEATMAP
# =============================================================================

def figure6_compact_summary_heatmap(df, save_path):
    metrics = [
        "roc_auc",
        "pr_auc",
        "test_recall",
        "test_specificity",
        "test_f1",
        "ece",
        "mean_uncertainty_std",
        "gradcam_mean_iou",
        "gradcam_pointing_game_accuracy",
        "gradcam_mean_inside_lung_ratio"
    ]

    higher_better = {
        "roc_auc": True,
        "pr_auc": True,
        "test_recall": True,
        "test_specificity": True,
        "test_f1": True,
        "ece": False,
        "mean_uncertainty_std": False,
        "gradcam_mean_iou": True,
        "gradcam_pointing_game_accuracy": True,
        "gradcam_mean_inside_lung_ratio": True
    }

    heat_df = pd.DataFrame(index=df["experiment_name"])

    for m in metrics:
        heat_df[m] = zscore_norm(df[m], higher_better=higher_better[m]).values

    values = heat_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([
        "ROC-AUC",
        "PR-AUC",
        "Recall",
        "Specificity",
        "F1",
        "ECE",
        "Unc.Std",
        "CAM IoU",
        "Pointing",
        "Inside-Lung"
    ], rotation=35, ha="right", fontsize=10)

    ax.set_yticks(np.arange(len(heat_df.index)))
    ax.set_yticklabels([nice_name(i) for i in heat_df.index], fontsize=10)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Standardized Relative Performance", fontsize=10)

    ax.set_title("Figure 6. Compact Standardized Multi-Metric Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# TABLE 1-LIKE EXPORT
# =============================================================================

def export_publication_table(df, save_csv, save_md):
    out = df[[
        "experiment_name",
        "roc_auc",
        "pr_auc",
        "test_recall",
        "test_specificity",
        "test_f1",
        "ece",
        "brier_score",
        "mean_uncertainty_std",
        "gradcam_mean_iou",
        "gradcam_pointing_game_accuracy",
        "gradcam_mean_inside_lung_ratio"
    ]].copy()

    out = out.sort_values("roc_auc", ascending=False)
    out.to_csv(save_csv, index=False)

    md = out.to_markdown(index=False)
    with open(save_md, "w", encoding="utf-8") as f:
        f.write(md)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    df = pd.read_csv(args.comparison_csv).copy()

    figure1_performance_vs_calibration(
        df,
        save_path=os.path.join(args.output_dir, "figure1_performance_vs_calibration.png")
    )

    figure2_clinical_operating_point(
        df,
        save_path=os.path.join(args.output_dir, "figure2_clinical_operating_point.png")
    )

    figure3_calibration_summary(
        df,
        save_path=os.path.join(args.output_dir, "figure3_calibration_summary.png")
    )

    figure4_uncertainty_summary(
        df,
        save_path=os.path.join(args.output_dir, "figure4_uncertainty_summary.png")
    )

    figure5_explainability_localization(
        df,
        save_path=os.path.join(args.output_dir, "figure5_explainability_localization.png")
    )

    figure6_compact_summary_heatmap(
        df,
        save_path=os.path.join(args.output_dir, "figure6_compact_summary_heatmap.png")
    )

    export_publication_table(
        df,
        save_csv=os.path.join(args.output_dir, "publication_table.csv"),
        save_md=os.path.join(args.output_dir, "publication_table.md")
    )

    print("[INFO] Publication-ready figures created at:", args.output_dir)


if __name__ == "__main__":
    main()