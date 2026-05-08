#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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


def nice_name(name: str) -> str:
    return str(name).replace("_", " ").replace("+", " + ")


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


# =============================================================================
# COMPOSITE SCORES
# =============================================================================

def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    directions = {
        "roc_auc": True,
        "pr_auc": True,
        "test_recall": True,
        "test_specificity": True,
        "test_precision": True,
        "test_f1": True,
        "ece": False,
        "brier_score": False,
        "mean_uncertainty_std": False,
        "mean_uncertainty_entropy": False,
        "uncertainty_error_correlation": True,
        "gradcam_mean_iou": True,
        "gradcam_pointing_game_accuracy": True,
        "gradcam_mean_inside_lung_ratio": True,
    }

    classification = ["roc_auc", "pr_auc", "test_f1"]
    clinical = ["test_recall", "test_specificity", "test_precision"]
    calibration = ["ece", "brier_score"]
    uncertainty = ["mean_uncertainty_std", "uncertainty_error_correlation"]
    explainability = ["gradcam_mean_iou", "gradcam_pointing_game_accuracy", "gradcam_mean_inside_lung_ratio"]

    def avg_score(metrics):
        vals = []
        for m in metrics:
            vals.append(minmax_norm(out[m], higher_better=directions[m]))
        vals = pd.concat(vals, axis=1)
        return vals.mean(axis=1)

    out["classification_score"] = avg_score(classification)
    out["clinical_score"] = avg_score(clinical)
    out["calibration_score"] = avg_score(calibration)
    out["uncertainty_score"] = avg_score(uncertainty)
    out["explainability_score"] = avg_score(explainability)

    out["overall_score"] = (
        0.30 * out["classification_score"] +
        0.20 * out["clinical_score"] +
        0.20 * out["calibration_score"] +
        0.15 * out["uncertainty_score"] +
        0.15 * out["explainability_score"]
    )

    return out


# =============================================================================
# 1) LINE GRAPH - MULTI-METRIC PROFILE
# =============================================================================

def plot_line_metric_profile(df, save_path):
    metrics = [
        "roc_auc",
        "pr_auc",
        "test_recall",
        "test_specificity",
        "test_f1",
        "ece",
        "mean_uncertainty_std",
        "gradcam_mean_iou"
    ]

    directions = {
        "roc_auc": True,
        "pr_auc": True,
        "test_recall": True,
        "test_specificity": True,
        "test_f1": True,
        "ece": False,
        "mean_uncertainty_std": False,
        "gradcam_mean_iou": True
    }

    plot_df = pd.DataFrame(index=df["experiment_name"])
    for m in metrics:
        plot_df[m] = minmax_norm(df[m], higher_better=directions[m]).values

    x = np.arange(len(metrics))

    plt.figure(figsize=(12, 6.5))
    for _, row in plot_df.iterrows():
        family = assign_family(row.name)
        plt.plot(
            x,
            row.values.astype(float),
            marker=family_marker(family),
            linewidth=2.2,
            label=nice_name(row.name),
            alpha=0.9
        )

    plt.xticks(x, ["ROC-AUC", "PR-AUC", "Recall", "Specificity", "F1", "ECE", "Unc.Std", "CAM IoU"], rotation=25)
    plt.ylabel("Normalized Relative Performance")
    plt.title("Figure 1. Multi-Metric Performance Profile Across Models")
    plt.grid(alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 2) SCATTER GRAPH - ROC vs PR
# =============================================================================

def plot_scatter_roc_pr(df, save_path):
    plt.figure(figsize=(8.2, 6.2))

    bubble = 200 + 900 * minmax_norm(df["ece"], higher_better=False)

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.scatter(
            row["roc_auc"],
            row["pr_auc"],
            s=float(bubble.loc[row.name]),
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.82,
            edgecolors="black",
            linewidths=0.8
        )
        plt.text(row["roc_auc"] + 0.001, row["pr_auc"] + 0.001, nice_name(row["experiment_name"]), fontsize=8.5)

    plt.xlabel("ROC-AUC")
    plt.ylabel("PR-AUC")
    plt.title("Figure 2. Discrimination Performance Space")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 3) CLINICAL SCATTER
# =============================================================================

def plot_scatter_clinical(df, save_path):
    plt.figure(figsize=(8.2, 6.2))

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.scatter(
            row["test_specificity"],
            row["test_recall"],
            s=240,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        plt.text(
            row["test_specificity"] + 0.0015,
            row["test_recall"] + 0.0015,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    plt.xlabel("Specificity")
    plt.ylabel("Recall / Sensitivity")
    plt.title("Figure 3. Clinical Operating Point Comparison")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 4) CALIBRATION SCATTER
# =============================================================================

def plot_scatter_calibration(df, save_path):
    plt.figure(figsize=(8.2, 6.2))

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.scatter(
            row["ece"],
            row["brier_score"],
            s=240,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        plt.text(row["ece"] + 0.0005, row["brier_score"] + 0.0005, nice_name(row["experiment_name"]), fontsize=8.5)

    plt.xlabel("Expected Calibration Error (↓)")
    plt.ylabel("Brier Score (↓)")
    plt.title("Figure 4. Calibration Quality Comparison")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 5) UNCERTAINTY SCATTER
# =============================================================================

def plot_scatter_uncertainty(df, save_path):
    plt.figure(figsize=(8.2, 6.2))

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.scatter(
            row["mean_uncertainty_std"],
            row["uncertainty_error_correlation"],
            s=240,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        plt.text(
            row["mean_uncertainty_std"] + 0.0005,
            row["uncertainty_error_correlation"] + 0.002,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    plt.xlabel("Mean Predictive Std")
    plt.ylabel("Uncertainty–Error Correlation")
    plt.title("Figure 5. Uncertainty Behavior Comparison")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 6) EXPLAINABILITY SCATTER
# =============================================================================

def plot_scatter_explainability(df, save_path):
    plt.figure(figsize=(8.2, 6.2))

    bubble = 180 + 850 * minmax_norm(df["gradcam_mean_inside_lung_ratio"], higher_better=True)

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.scatter(
            row["gradcam_mean_iou"],
            row["gradcam_pointing_game_accuracy"],
            s=float(bubble.loc[row.name]),
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.82,
            edgecolors="black",
            linewidths=0.8
        )
        plt.text(
            row["gradcam_mean_iou"] + 0.001,
            row["gradcam_pointing_game_accuracy"] + 0.002,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    plt.xlabel("Grad-CAM / BBox IoU")
    plt.ylabel("Pointing Game Accuracy")
    plt.title("Figure 6. Explainability Localization Comparison")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 7) DONUT / PIE STYLE CHART
# =============================================================================

def plot_donut_overall_score(df, save_path):
    vals = df["overall_score"].values
    labels = [nice_name(x) for x in df["experiment_name"].tolist()]
    families = [assign_family(x) for x in df["experiment_name"].tolist()]
    colors = [family_color(f) for f in families]

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.82,
        colors=colors,
        wedgeprops=dict(width=0.38, edgecolor="white")
    )

    centre_circle = plt.Circle((0, 0), 0.45, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title("Figure 7. Relative Composite Contribution of Each Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 8) RADAR GRAPH
# =============================================================================

def plot_radar_graph(df, save_path):
    metrics = [
        "classification_score",
        "clinical_score",
        "calibration_score",
        "uncertainty_score",
        "explainability_score"
    ]

    labels = ["Classification", "Clinical", "Calibration", "Uncertainty", "Explainability"]
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.8, 8.8))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        vals = row[metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, label=nice_name(row["experiment_name"]))
        ax.fill(angles, vals, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Figure 8. Domain-Wise Model Profile", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.08), fontsize=8.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 9) LINE GRAPH - DOMAIN SCORES
# =============================================================================

def plot_line_domain_scores(df, save_path):
    metrics = [
        "classification_score",
        "clinical_score",
        "calibration_score",
        "uncertainty_score",
        "explainability_score"
    ]
    x = np.arange(len(metrics))

    plt.figure(figsize=(11.5, 6.2))

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.plot(
            x,
            row[metrics].values.astype(float),
            marker=family_marker(family),
            linewidth=2.2,
            label=nice_name(row["experiment_name"]),
            alpha=0.9
        )

    plt.xticks(x, ["Classification", "Clinical", "Calibration", "Uncertainty", "Explainability"], rotation=20)
    plt.ylabel("Composite Domain Score")
    plt.title("Figure 9. Domain-Level Comparative Profile")
    plt.grid(alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# 10) PARETO-LIKE FRONTIER
# =============================================================================

def plot_pareto_frontier(df, save_path):
    plt.figure(figsize=(8.2, 6.2))

    x = df["overall_score"]
    y = df["roc_auc"]

    for _, row in df.iterrows():
        family = assign_family(row["experiment_name"])
        plt.scatter(
            row["overall_score"],
            row["roc_auc"],
            s=240,
            marker=family_marker(family),
            color=family_color(family),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8
        )
        plt.text(
            row["overall_score"] + 0.003,
            row["roc_auc"] + 0.001,
            nice_name(row["experiment_name"]),
            fontsize=8.5
        )

    plt.xlabel("Overall Composite Score")
    plt.ylabel("ROC-AUC")
    plt.title("Figure 10. Overall Utility vs Discrimination Performance")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


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
    df = add_composite_scores(df)

    plot_line_metric_profile(df, os.path.join(args.output_dir, "figure1_line_metric_profile.png"))
    plot_scatter_roc_pr(df, os.path.join(args.output_dir, "figure2_scatter_roc_vs_pr.png"))
    plot_scatter_clinical(df, os.path.join(args.output_dir, "figure3_scatter_clinical.png"))
    plot_scatter_calibration(df, os.path.join(args.output_dir, "figure4_scatter_calibration.png"))
    plot_scatter_uncertainty(df, os.path.join(args.output_dir, "figure5_scatter_uncertainty.png"))
    plot_scatter_explainability(df, os.path.join(args.output_dir, "figure6_scatter_explainability.png"))
    plot_donut_overall_score(df, os.path.join(args.output_dir, "figure7_donut_overall_score.png"))
    plot_radar_graph(df, os.path.join(args.output_dir, "figure8_radar_domain_profile.png"))
    plot_line_domain_scores(df, os.path.join(args.output_dir, "figure9_line_domain_scores.png"))
    plot_pareto_frontier(df, os.path.join(args.output_dir, "figure10_pareto_frontier.png"))

    df.to_csv(os.path.join(args.output_dir, "comparison_with_composite_scores.csv"), index=False)

    print("[INFO] Publication-ready graphs created at:", args.output_dir)


if __name__ == "__main__":
    main()