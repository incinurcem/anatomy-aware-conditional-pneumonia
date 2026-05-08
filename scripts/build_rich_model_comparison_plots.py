#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# BASIC UTILS
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def minmax_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min = s.min()
    s_max = s.max()

    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.zeros(len(s)), index=s.index)

    if abs(s_max - s_min) < 1e-12:
        out = pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    else:
        out = (s - s_min) / (s_max - s_min)

    if not higher_is_better:
        out = 1.0 - out

    return out


def rank_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    ascending = not higher_is_better
    return s.rank(method="min", ascending=ascending)


def wrap_label(text: str, width: int = 16) -> str:
    words = str(text).split("_")
    lines = []
    current = []

    for w in words:
        candidate = "_".join(current + [w])
        if len(candidate) <= width:
            current.append(w)
        else:
            if current:
                lines.append("_".join(current))
            current = [w]

    if current:
        lines.append("_".join(current))

    return "\n".join(lines)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def save_grouped_bar_chart(df, metrics, title, save_path):
    models = df["experiment_name"].tolist()
    x = np.arange(len(models))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics

    plt.figure(figsize=(12, 7))

    for i, metric in enumerate(metrics):
        vals = df[metric].values
        plt.bar(x + i * width - (n_metrics - 1) * width / 2, vals, width=width, label=metric)

    plt.xticks(x, [wrap_label(m) for m in models], rotation=0)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_lollipop_chart(df, metric, title, save_path, ascending=False):
    plot_df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)

    y = np.arange(len(plot_df))
    x = plot_df[metric].values
    labels = plot_df["experiment_name"].tolist()

    plt.figure(figsize=(10, 6))
    plt.hlines(y=y, xmin=0, xmax=x)
    plt.plot(x, y, "o")
    plt.yticks(y, [wrap_label(l, 20) for l in labels])
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_heatmap(matrix_df, title, save_path, cmap="viridis", annotate=True, fmt=".2f"):
    values = matrix_df.values.astype(float)

    plt.figure(figsize=(max(8, 1.2 * len(matrix_df.columns)), max(5, 0.7 * len(matrix_df.index))))
    im = plt.imshow(values, aspect="auto", cmap=cmap)
    plt.colorbar(im)

    plt.xticks(np.arange(len(matrix_df.columns)), [wrap_label(c, 14) for c in matrix_df.columns], rotation=45, ha="right")
    plt.yticks(np.arange(len(matrix_df.index)), [wrap_label(i, 18) for i in matrix_df.index])

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_radar_chart(df, metrics, title, save_path, higher_better_map):
    norm_df = pd.DataFrame(index=df["experiment_name"])

    for metric in metrics:
        norm_df[metric] = minmax_normalize(df[metric], higher_is_better=higher_better_map[metric]).values

    labels = metrics
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for _, row in norm_df.iterrows():
        values = row[labels].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row.name)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_scatter_with_labels(df, x_metric, y_metric, title, save_path, size_metric=None):
    x = df[x_metric].values
    y = df[y_metric].values

    plt.figure(figsize=(8, 6))

    if size_metric is not None:
        s = df[size_metric].values.astype(float)
        s_min, s_max = np.nanmin(s), np.nanmax(s)
        if abs(s_max - s_min) < 1e-12:
            bubble = np.ones_like(s) * 200.0
        else:
            bubble = 100 + 700 * (s - s_min) / (s_max - s_min)
        plt.scatter(x, y, s=bubble, alpha=0.7)
    else:
        plt.scatter(x, y, s=120)

    for _, row in df.iterrows():
        plt.text(row[x_metric], row[y_metric], " " + row["experiment_name"], fontsize=9)

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_parallel_coordinates(df, metrics, title, save_path, higher_better_map):
    norm_df = pd.DataFrame()
    norm_df["experiment_name"] = df["experiment_name"]

    for metric in metrics:
        norm_df[metric] = minmax_normalize(df[metric], higher_is_better=higher_better_map[metric]).values

    xs = np.arange(len(metrics))

    plt.figure(figsize=(13, 7))
    for _, row in norm_df.iterrows():
        ys = row[metrics].values.astype(float)
        plt.plot(xs, ys, marker="o", linewidth=2, label=row["experiment_name"])

    plt.xticks(xs, metrics, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Normalized score")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_rank_bump_chart(df, metrics, title, save_path, higher_better_map):
    rank_df = pd.DataFrame(index=df["experiment_name"])

    for metric in metrics:
        rank_df[metric] = rank_series(df.set_index("experiment_name")[metric], higher_is_better=higher_better_map[metric])

    xs = np.arange(len(metrics))

    plt.figure(figsize=(12, 7))
    for model_name in rank_df.index:
        ys = rank_df.loc[model_name, metrics].values.astype(float)
        plt.plot(xs, ys, marker="o", linewidth=2, label=model_name)
        plt.text(xs[-1] + 0.05, ys[-1], model_name, fontsize=9, va="center")

    plt.gca().invert_yaxis()
    plt.xticks(xs, metrics, rotation=30, ha="right")
    plt.ylabel("Rank (1 = best)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_tradeoff_quadrant_plot(df, x_metric, y_metric, title, save_path, x_higher_better=True, y_higher_better=True):
    x = df[x_metric].values.astype(float)
    y = df[y_metric].values.astype(float)

    x_mid = np.nanmean(x)
    y_mid = np.nanmean(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=130)

    for _, row in df.iterrows():
        plt.text(row[x_metric], row[y_metric], " " + row["experiment_name"], fontsize=9)

    plt.axvline(x_mid, linestyle="--")
    plt.axhline(y_mid, linestyle="--")

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_topk_profile_heatmap(df, metric_groups, save_path):
    """
    metric_groups: ordered dict-like normal dict with group_name -> [metric1, metric2]
    """
    rows = []
    index_names = []

    higher_better = {
        "roc_auc": True,
        "pr_auc": True,
        "test_balanced_accuracy": True,
        "test_recall": True,
        "test_specificity": True,
        "test_precision": True,
        "test_f1": True,
        "ece": False,
        "mce": False,
        "brier_score": False,
        "nll": False,
        "mean_uncertainty_std": False,
        "mean_uncertainty_entropy": False,
        "uncertainty_error_correlation": True,
        "gradcam_mean_dice": True,
        "gradcam_mean_iou": True,
        "gradcam_mean_hd95": False,
        "gradcam_pointing_game_accuracy": True,
        "gradcam_mean_inside_lung_ratio": True,
        "gradcam_mean_outside_lung_ratio": False,
    }

    for _, row in df.iterrows():
        vals = []
        for group_name, metrics in metric_groups.items():
            group_vals = []
            for m in metrics:
                group_vals.append(minmax_normalize(df[m], higher_is_better=higher_better[m]).loc[row.name])
            vals.append(np.mean(group_vals))
        rows.append(vals)
        index_names.append(row["experiment_name"])

    out_df = pd.DataFrame(rows, index=index_names, columns=list(metric_groups.keys()))
    save_heatmap(out_df, "Group-wise Model Profile Heatmap", save_path, cmap="plasma", annotate=True, fmt=".2f")


# =============================================================================
# MAIN RICH COMPARISON BUILDER
# =============================================================================

def build_rich_comparison_plots(comparison_csv: str, output_dir: str):
    ensure_dir(output_dir)

    df = pd.read_csv(comparison_csv).copy()

    if "experiment_name" not in df.columns:
        raise ValueError("comparison CSV must include 'experiment_name' column.")

    # -------------------------------------------------------------------------
    # metric directions
    # -------------------------------------------------------------------------
    higher_better_map = {
        "roc_auc": True,
        "pr_auc": True,
        "test_balanced_accuracy": True,
        "test_recall": True,
        "test_specificity": True,
        "test_precision": True,
        "test_f1": True,
        "ece": False,
        "mce": False,
        "brier_score": False,
        "nll": False,
        "mean_uncertainty_std": False,
        "mean_uncertainty_entropy": False,
        "uncertainty_error_correlation": True,
        "gradcam_mean_dice": True,
        "gradcam_mean_iou": True,
        "gradcam_mean_hd95": False,
        "gradcam_pointing_game_accuracy": True,
        "gradcam_mean_inside_lung_ratio": True,
        "gradcam_mean_outside_lung_ratio": False,
    }

    # -------------------------------------------------------------------------
    # 1) grouped bar charts
    # -------------------------------------------------------------------------
    save_grouped_bar_chart(
        df,
        metrics=["roc_auc", "pr_auc", "test_f1"],
        title="Core Classification Performance",
        save_path=os.path.join(output_dir, "grouped_core_classification.png")
    )

    save_grouped_bar_chart(
        df,
        metrics=["test_recall", "test_specificity", "test_precision"],
        title="Clinical Operating Metrics",
        save_path=os.path.join(output_dir, "grouped_clinical_metrics.png")
    )

    save_grouped_bar_chart(
        df,
        metrics=["ece", "brier_score", "nll"],
        title="Calibration Metrics",
        save_path=os.path.join(output_dir, "grouped_calibration_metrics.png")
    )

    save_grouped_bar_chart(
        df,
        metrics=["gradcam_mean_dice", "gradcam_mean_iou", "gradcam_pointing_game_accuracy"],
        title="Explainability Localization Metrics",
        save_path=os.path.join(output_dir, "grouped_explainability_metrics.png")
    )

    # -------------------------------------------------------------------------
    # 2) lollipop charts
    # -------------------------------------------------------------------------
    save_lollipop_chart(
        df, "roc_auc", "Model Ranking by ROC-AUC",
        os.path.join(output_dir, "lollipop_roc_auc.png"), ascending=False
    )
    save_lollipop_chart(
        df, "ece", "Model Ranking by ECE (Lower is Better)",
        os.path.join(output_dir, "lollipop_ece.png"), ascending=True
    )
    save_lollipop_chart(
        df, "gradcam_mean_iou", "Model Ranking by GradCAM IoU",
        os.path.join(output_dir, "lollipop_gradcam_iou.png"), ascending=False
    )

    # -------------------------------------------------------------------------
    # 3) normalized score heatmap
    # -------------------------------------------------------------------------
    heatmap_metrics = [
        "roc_auc",
        "pr_auc",
        "test_balanced_accuracy",
        "test_recall",
        "test_specificity",
        "test_precision",
        "test_f1",
        "ece",
        "brier_score",
        "mean_uncertainty_std",
        "gradcam_mean_iou",
        "gradcam_pointing_game_accuracy",
        "gradcam_mean_inside_lung_ratio"
    ]

    norm_df = pd.DataFrame(index=df["experiment_name"])
    for m in heatmap_metrics:
        norm_df[m] = minmax_normalize(df[m], higher_is_better=higher_better_map[m]).values

    save_heatmap(
        norm_df,
        title="Normalized Multi-Metric Heatmap",
        save_path=os.path.join(output_dir, "normalized_multi_metric_heatmap.png"),
        cmap="viridis",
        annotate=True,
        fmt=".2f"
    )

    # -------------------------------------------------------------------------
    # 4) ranking heatmap
    # -------------------------------------------------------------------------
    rank_df = pd.DataFrame(index=df["experiment_name"])
    for m in heatmap_metrics:
        rank_df[m] = rank_series(df.set_index("experiment_name")[m], higher_is_better=higher_better_map[m])

    save_heatmap(
        rank_df,
        title="Ranking Heatmap (1 = Best)",
        save_path=os.path.join(output_dir, "ranking_heatmap.png"),
        cmap="coolwarm_r",
        annotate=True,
        fmt=".0f"
    )

    # -------------------------------------------------------------------------
    # 5) radar charts
    # -------------------------------------------------------------------------
    save_radar_chart(
        df,
        metrics=["roc_auc", "pr_auc", "test_recall", "test_specificity", "test_f1"],
        title="Radar Chart - Classification Profile",
        save_path=os.path.join(output_dir, "radar_classification_profile.png"),
        higher_better_map=higher_better_map
    )

    save_radar_chart(
        df,
        metrics=["ece", "brier_score", "mean_uncertainty_std", "gradcam_mean_iou", "gradcam_mean_inside_lung_ratio"],
        title="Radar Chart - Reliability + Explainability Profile",
        save_path=os.path.join(output_dir, "radar_reliability_explainability_profile.png"),
        higher_better_map=higher_better_map
    )

    # -------------------------------------------------------------------------
    # 6) scatter and trade-off plots
    # -------------------------------------------------------------------------
    save_scatter_with_labels(
        df,
        x_metric="test_recall",
        y_metric="test_specificity",
        title="Clinical Trade-off: Recall vs Specificity",
        save_path=os.path.join(output_dir, "scatter_recall_vs_specificity.png"),
        size_metric="roc_auc"
    )

    save_scatter_with_labels(
        df,
        x_metric="roc_auc",
        y_metric="ece",
        title="Calibration-Quality Trade-off: ROC-AUC vs ECE",
        save_path=os.path.join(output_dir, "scatter_roc_auc_vs_ece.png"),
        size_metric="pr_auc"
    )

    save_scatter_with_labels(
        df,
        x_metric="roc_auc",
        y_metric="mean_uncertainty_std",
        title="Quality vs Uncertainty: ROC-AUC vs Mean Predictive Std",
        save_path=os.path.join(output_dir, "scatter_roc_auc_vs_uncertainty.png"),
        size_metric="test_f1"
    )

    save_scatter_with_labels(
        df,
        x_metric="roc_auc",
        y_metric="gradcam_mean_iou",
        title="Quality vs Explainability: ROC-AUC vs GradCAM IoU",
        save_path=os.path.join(output_dir, "scatter_roc_auc_vs_gradcam_iou.png"),
        size_metric="gradcam_pointing_game_accuracy"
    )

    save_scatter_with_labels(
        df,
        x_metric="gradcam_mean_inside_lung_ratio",
        y_metric="gradcam_mean_outside_lung_ratio",
        title="Inside-Lung vs Outside-Lung CAM Activation",
        save_path=os.path.join(output_dir, "scatter_inside_vs_outside_lung_ratio.png"),
        size_metric="gradcam_mean_iou"
    )

    save_tradeoff_quadrant_plot(
        df,
        x_metric="test_recall",
        y_metric="test_specificity",
        title="Quadrant Plot: Recall vs Specificity",
        save_path=os.path.join(output_dir, "quadrant_recall_vs_specificity.png")
    )

    save_tradeoff_quadrant_plot(
        df,
        x_metric="roc_auc",
        y_metric="ece",
        title="Quadrant Plot: ROC-AUC vs ECE",
        save_path=os.path.join(output_dir, "quadrant_roc_auc_vs_ece.png")
    )

    # -------------------------------------------------------------------------
    # 7) parallel coordinates
    # -------------------------------------------------------------------------
    save_parallel_coordinates(
        df,
        metrics=[
            "roc_auc",
            "pr_auc",
            "test_recall",
            "test_specificity",
            "test_f1",
            "ece",
            "mean_uncertainty_std",
            "gradcam_mean_iou"
        ],
        title="Parallel Coordinates - Full Model Comparison",
        save_path=os.path.join(output_dir, "parallel_coordinates_full.png"),
        higher_better_map=higher_better_map
    )

    # -------------------------------------------------------------------------
    # 8) rank bump chart
    # -------------------------------------------------------------------------
    save_rank_bump_chart(
        df,
        metrics=[
            "roc_auc",
            "pr_auc",
            "test_recall",
            "test_specificity",
            "ece",
            "mean_uncertainty_std",
            "gradcam_mean_iou"
        ],
        title="Rank Flow Across Metrics",
        save_path=os.path.join(output_dir, "rank_bump_chart.png"),
        higher_better_map=higher_better_map
    )

    # -------------------------------------------------------------------------
    # 9) grouped profile heatmap
    # -------------------------------------------------------------------------
    metric_groups = {
        "classification": ["roc_auc", "pr_auc", "test_f1"],
        "clinical": ["test_recall", "test_specificity", "test_precision"],
        "calibration": ["ece", "brier_score", "nll"],
        "uncertainty": ["mean_uncertainty_std", "mean_uncertainty_entropy", "uncertainty_error_correlation"],
        "explainability": ["gradcam_mean_dice", "gradcam_mean_iou", "gradcam_pointing_game_accuracy"],
        "anatomical_focus": ["gradcam_mean_inside_lung_ratio", "gradcam_mean_outside_lung_ratio"]
    }
    save_topk_profile_heatmap(
        df.reset_index(drop=True),
        metric_groups=metric_groups,
        save_path=os.path.join(output_dir, "group_profile_heatmap.png")
    )

    # -------------------------------------------------------------------------
    # 10) overall composite scores
    # -------------------------------------------------------------------------
    composite_parts = {
        "classification_score": ["roc_auc", "pr_auc", "test_f1"],
        "clinical_score": ["test_recall", "test_specificity", "test_precision"],
        "reliability_score": ["ece", "brier_score", "mean_uncertainty_std"],
        "explainability_score": ["gradcam_mean_iou", "gradcam_pointing_game_accuracy", "gradcam_mean_inside_lung_ratio"]
    }

    composite_df = df.copy()

    for score_name, metrics in composite_parts.items():
        vals = []
        for _, row in df.iterrows():
            s = 0.0
            for m in metrics:
                s += minmax_normalize(df[m], higher_is_better=higher_better_map[m]).loc[row.name]
            vals.append(s / len(metrics))
        composite_df[score_name] = vals

    composite_df["overall_score"] = (
        0.35 * composite_df["classification_score"] +
        0.25 * composite_df["clinical_score"] +
        0.20 * composite_df["reliability_score"] +
        0.20 * composite_df["explainability_score"]
    )

    composite_df.to_csv(os.path.join(output_dir, "comparison_with_composite_scores.csv"), index=False)

    save_grouped_bar_chart(
        composite_df,
        metrics=["classification_score", "clinical_score", "reliability_score", "explainability_score"],
        title="Composite Subscores by Model",
        save_path=os.path.join(output_dir, "grouped_composite_subscores.png")
    )

    save_lollipop_chart(
        composite_df,
        "overall_score",
        "Overall Composite Score Ranking",
        os.path.join(output_dir, "lollipop_overall_score.png"),
        ascending=False
    )

    # -------------------------------------------------------------------------
    # 11) summary json
    # -------------------------------------------------------------------------
    best_overall = composite_df.sort_values("overall_score", ascending=False).iloc[0]
    best_classification = composite_df.sort_values("classification_score", ascending=False).iloc[0]
    best_reliability = composite_df.sort_values("reliability_score", ascending=False).iloc[0]
    best_explainability = composite_df.sort_values("explainability_score", ascending=False).iloc[0]

    summary = {
        "best_overall_model": str(best_overall["experiment_name"]),
        "best_classification_model": str(best_classification["experiment_name"]),
        "best_reliability_model": str(best_reliability["experiment_name"]),
        "best_explainability_model": str(best_explainability["experiment_name"])
    }

    with open(os.path.join(output_dir, "rich_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[INFO] Rich comparison plots saved to:", output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    build_rich_comparison_plots(
        comparison_csv=args.comparison_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()