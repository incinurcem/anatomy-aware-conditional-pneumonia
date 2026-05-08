#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# BASIC
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def nice_name(name: str) -> str:
    return str(name).replace("_", " ").replace("+", " + ")


def format_metric_value(val, decimals=3):
    if pd.isna(val):
        return "-"
    return f"{float(val):.{decimals}f}"


def get_best_and_second_indices(series: pd.Series, higher_is_better: bool):
    s = pd.to_numeric(series, errors="coerce")

    if not higher_is_better:
        s = -s

    valid = s.dropna()
    if len(valid) == 0:
        return None, None

    sorted_idx = valid.sort_values(ascending=False).index.tolist()
    best_idx = sorted_idx[0] if len(sorted_idx) >= 1 else None
    second_idx = sorted_idx[1] if len(sorted_idx) >= 2 else None
    return best_idx, second_idx


# =============================================================================
# TABLE DRAWER
# =============================================================================

def draw_publication_table_figure(
    df: pd.DataFrame,
    display_cols,
    metric_directions,
    title: str,
    save_path: str,
    decimals_map=None,
    figsize=(16, 3.8),
    font_size=10,
    title_size=13,
    model_col="experiment_name"
):
    """
    display_cols: ordered columns to show (including model_col)
    metric_directions: dict metric -> True if higher is better, False otherwise
    decimals_map: dict metric -> decimals
    """
    if decimals_map is None:
        decimals_map = {}

    table_df = df[display_cols].copy()
    table_df[model_col] = table_df[model_col].apply(nice_name)

    # format text
    cell_text = []
    for _, row in table_df.iterrows():
        row_list = []
        for col in display_cols:
            if col == model_col:
                row_list.append(str(row[col]))
            else:
                row_list.append(format_metric_value(row[col], decimals=decimals_map.get(col, 3)))
        cell_text.append(row_list)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    col_labels = []
    for col in display_cols:
        if col == model_col:
            col_labels.append("Model")
        else:
            col_labels.append(col)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.6)

    n_rows = len(table_df)
    n_cols = len(display_cols)

    # header style
    for c in range(n_cols):
        cell = tbl[(0, c)]
        cell.set_facecolor("#D9EAF7")
        cell.set_text_props(weight="bold", color="black")
        cell.set_edgecolor("#4F81BD")
        cell.set_linewidth(1.0)

    # zebra rows
    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            cell = tbl[(r, c)]
            if r % 2 == 1:
                cell.set_facecolor("#F8FBFD")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#B0B0B0")
            cell.set_linewidth(0.6)

    # first column bold
    for r in range(1, n_rows + 1):
        tbl[(r, 0)].set_text_props(weight="bold")

    # highlight best / second best for metric columns
    for c, col in enumerate(display_cols):
        if col == model_col:
            continue

        higher_is_better = metric_directions.get(col, True)
        best_idx, second_idx = get_best_and_second_indices(df[col], higher_is_better=higher_is_better)

        if best_idx is not None:
            table_row = list(df.index).index(best_idx) + 1
            tbl[(table_row, c)].set_facecolor("#C6EFCE")
            tbl[(table_row, c)].set_text_props(weight="bold", color="#006100")

        if second_idx is not None:
            table_row = list(df.index).index(second_idx) + 1
            tbl[(table_row, c)].set_facecolor("#FFEB9C")
            tbl[(table_row, c)].set_text_props(weight="bold", color="#9C6500")

    plt.title(title, fontsize=title_size, pad=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def export_table_versions(df, cols, csv_path, md_path, model_col="experiment_name", decimals_map=None):
    if decimals_map is None:
        decimals_map = {}

    out = df[cols].copy()
    out[model_col] = out[model_col].apply(nice_name)

    for c in cols:
        if c == model_col:
            continue
        out[c] = out[c].apply(lambda x: format_metric_value(x, decimals=decimals_map.get(c, 3)))

    out.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(out.to_markdown(index=False))


# =============================================================================
# MAIN TABLE GENERATOR
# =============================================================================

def build_publication_plot_tables(comparison_csv: str, output_dir: str):
    ensure_dir(output_dir)

    df = pd.read_csv(comparison_csv).copy()

    if "experiment_name" not in df.columns:
        raise ValueError("comparison CSV must include 'experiment_name' column.")

    # directions
    metric_directions = {
        "roc_auc": True,
        "pr_auc": True,
        "test_accuracy": True,
        "test_balanced_accuracy": True,
        "test_recall": True,
        "test_specificity": True,
        "test_precision": True,
        "test_npv": True,
        "test_f1": True,
        "best_f1_threshold": True,
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

    decimals_map = {
        "roc_auc": 4,
        "pr_auc": 4,
        "test_accuracy": 4,
        "test_balanced_accuracy": 4,
        "test_recall": 4,
        "test_specificity": 4,
        "test_precision": 4,
        "test_npv": 4,
        "test_f1": 4,
        "best_f1_threshold": 3,
        "ece": 4,
        "mce": 4,
        "brier_score": 4,
        "nll": 4,
        "mean_uncertainty_std": 4,
        "mean_uncertainty_entropy": 4,
        "uncertainty_error_correlation": 4,
        "gradcam_mean_dice": 4,
        "gradcam_mean_iou": 4,
        "gradcam_mean_hd95": 4,
        "gradcam_pointing_game_accuracy": 4,
        "gradcam_mean_inside_lung_ratio": 4,
        "gradcam_mean_outside_lung_ratio": 4,
    }

    # -------------------------------------------------------------------------
    # Table 1: classification
    # -------------------------------------------------------------------------
    cols1 = [
        "experiment_name",
        "roc_auc",
        "pr_auc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1"
    ]
    draw_publication_table_figure(
        df=df,
        display_cols=cols1,
        metric_directions=metric_directions,
        title="Table 1. Overall Classification Performance Comparison",
        save_path=os.path.join(output_dir, "table1_classification_performance.png"),
        decimals_map=decimals_map,
        figsize=(14, 3.6)
    )
    export_table_versions(
        df, cols1,
        os.path.join(output_dir, "table1_classification_performance.csv"),
        os.path.join(output_dir, "table1_classification_performance.md"),
        decimals_map=decimals_map
    )

    # -------------------------------------------------------------------------
    # Table 2: clinical metrics
    # -------------------------------------------------------------------------
    cols2 = [
        "experiment_name",
        "test_recall",
        "test_specificity",
        "test_precision",
        "test_npv",
        "best_f1_threshold"
    ]
    draw_publication_table_figure(
        df=df,
        display_cols=cols2,
        metric_directions=metric_directions,
        title="Table 2. Clinical Operating Metrics",
        save_path=os.path.join(output_dir, "table2_clinical_metrics.png"),
        decimals_map=decimals_map,
        figsize=(14, 3.6)
    )
    export_table_versions(
        df, cols2,
        os.path.join(output_dir, "table2_clinical_metrics.csv"),
        os.path.join(output_dir, "table2_clinical_metrics.md"),
        decimals_map=decimals_map
    )

    # -------------------------------------------------------------------------
    # Table 3: calibration
    # -------------------------------------------------------------------------
    cols3 = [
        "experiment_name",
        "ece",
        "mce",
        "brier_score",
        "nll"
    ]
    draw_publication_table_figure(
        df=df,
        display_cols=cols3,
        metric_directions=metric_directions,
        title="Table 3. Calibration Performance",
        save_path=os.path.join(output_dir, "table3_calibration_metrics.png"),
        decimals_map=decimals_map,
        figsize=(12.5, 3.5)
    )
    export_table_versions(
        df, cols3,
        os.path.join(output_dir, "table3_calibration_metrics.csv"),
        os.path.join(output_dir, "table3_calibration_metrics.md"),
        decimals_map=decimals_map
    )

    # -------------------------------------------------------------------------
    # Table 4: uncertainty
    # -------------------------------------------------------------------------
    cols4 = [
        "experiment_name",
        "mean_uncertainty_std",
        "mean_uncertainty_entropy",
        "uncertainty_error_correlation"
    ]
    draw_publication_table_figure(
        df=df,
        display_cols=cols4,
        metric_directions=metric_directions,
        title="Table 4. Uncertainty Characteristics",
        save_path=os.path.join(output_dir, "table4_uncertainty_metrics.png"),
        decimals_map=decimals_map,
        figsize=(12.5, 3.5)
    )
    export_table_versions(
        df, cols4,
        os.path.join(output_dir, "table4_uncertainty_metrics.csv"),
        os.path.join(output_dir, "table4_uncertainty_metrics.md"),
        decimals_map=decimals_map
    )

    # -------------------------------------------------------------------------
    # Table 5: explainability
    # -------------------------------------------------------------------------
    cols5 = [
        "experiment_name",
        "gradcam_mean_dice",
        "gradcam_mean_iou",
        "gradcam_mean_hd95",
        "gradcam_pointing_game_accuracy",
        "gradcam_mean_inside_lung_ratio",
        "gradcam_mean_outside_lung_ratio"
    ]
    draw_publication_table_figure(
        df=df,
        display_cols=cols5,
        metric_directions=metric_directions,
        title="Table 5. Explainability and Localization Metrics",
        save_path=os.path.join(output_dir, "table5_explainability_metrics.png"),
        decimals_map=decimals_map,
        figsize=(18, 3.8)
    )
    export_table_versions(
        df, cols5,
        os.path.join(output_dir, "table5_explainability_metrics.csv"),
        os.path.join(output_dir, "table5_explainability_metrics.md"),
        decimals_map=decimals_map
    )

    # -------------------------------------------------------------------------
    # Table 6: full summary
    # -------------------------------------------------------------------------
    cols6 = [
        "experiment_name",
        "roc_auc",
        "pr_auc",
        "test_recall",
        "test_specificity",
        "test_f1",
        "ece",
        "mean_uncertainty_std",
        "gradcam_mean_iou",
        "gradcam_pointing_game_accuracy"
    ]
    draw_publication_table_figure(
        df=df,
        display_cols=cols6,
        metric_directions=metric_directions,
        title="Table 6. Compact Multi-Domain Summary of All Models",
        save_path=os.path.join(output_dir, "table6_compact_full_summary.png"),
        decimals_map=decimals_map,
        figsize=(18, 4.2)
    )
    export_table_versions(
        df, cols6,
        os.path.join(output_dir, "table6_compact_full_summary.csv"),
        os.path.join(output_dir, "table6_compact_full_summary.md"),
        decimals_map=decimals_map
    )

    print("[INFO] Publication plot tables created in:", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    build_publication_plot_tables(
        comparison_csv=args.comparison_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()