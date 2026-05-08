import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Utils
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def pick_metric(metrics: dict, key: str, default=np.nan):
    if metrics is None:
        return default
    value = metrics.get(key, default)
    if value is None:
        return np.nan
    return value


# =========================================================
# Read one experiment
# =========================================================
def read_experiment(exp_name, exp_dir):
    test_metrics_path = os.path.join(exp_dir, "test_metrics.json")
    final_summary_path = os.path.join(exp_dir, "final_summary.json")
    history_path = os.path.join(exp_dir, "history.csv")
    config_path = os.path.join(exp_dir, "config.json")

    test_metrics = safe_read_json(test_metrics_path)
    final_summary = safe_read_json(final_summary_path)
    history = safe_read_csv(history_path)
    config = safe_read_json(config_path)

    row = {
        "experiment_name": exp_name,
        "experiment_dir": exp_dir,
        "exists": os.path.exists(exp_dir),

        # test metrics
        "test_accuracy": pick_metric(test_metrics, "accuracy"),
        "test_precision": pick_metric(test_metrics, "precision"),
        "test_recall": pick_metric(test_metrics, "recall_sensitivity"),
        "test_specificity": pick_metric(test_metrics, "specificity"),
        "test_f1": pick_metric(test_metrics, "f1"),
        "test_roc_auc": pick_metric(test_metrics, "roc_auc"),
        "test_pr_auc": pick_metric(test_metrics, "pr_auc"),
        "test_ppv": pick_metric(test_metrics, "ppv"),
        "test_npv": pick_metric(test_metrics, "npv"),
        "test_balanced_accuracy": pick_metric(test_metrics, "balanced_accuracy"),
        "test_tn": pick_metric(test_metrics, "tn"),
        "test_fp": pick_metric(test_metrics, "fp"),
        "test_fn": pick_metric(test_metrics, "fn"),
        "test_tp": pick_metric(test_metrics, "tp"),
        "test_loss": pick_metric(test_metrics, "loss"),

        # final summary
        "best_epoch": pick_metric(final_summary, "best_epoch"),
        "best_val_auc": pick_metric(final_summary, "best_val_auc"),

        # config
        "input_mode": config.get("input_mode") if config else None,
        "model_name": config.get("model_name") if config else None,
        "condition_dim": config.get("condition_dim") if config else None,
        "epochs": config.get("epochs") if config else None,
        "batch_size": config.get("batch_size") if config else None,
        "lr": config.get("lr") if config else None,
    }

    # history info
    if history is not None and len(history) > 0:
        row["num_history_rows"] = len(history)

        if "val_roc_auc" in history.columns:
            row["history_best_val_roc_auc"] = history["val_roc_auc"].max()
        else:
            row["history_best_val_roc_auc"] = np.nan

        if "test_roc_auc" in history.columns:
            row["history_best_test_roc_auc"] = history["test_roc_auc"].max()
        else:
            row["history_best_test_roc_auc"] = np.nan
    else:
        row["num_history_rows"] = 0
        row["history_best_val_roc_auc"] = np.nan
        row["history_best_test_roc_auc"] = np.nan

    return row


# =========================================================
# Plotting
# =========================================================
def plot_metric(df, metric_col, output_path, title=None, ylabel=None):
    plot_df = df.copy()
    plot_df = plot_df.sort_values(metric_col, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["experiment_name"], plot_df[metric_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel if ylabel else metric_col)
    plt.title(title if title else metric_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_metrics_grid(df, output_dir):
    metrics_to_plot = [
        ("test_roc_auc", "Test ROC-AUC"),
        ("test_pr_auc", "Test PR-AUC"),
        ("test_f1", "Test F1"),
        ("test_recall", "Test Recall"),
        ("test_specificity", "Test Specificity"),
        ("test_balanced_accuracy", "Test Balanced Accuracy"),
    ]

    for metric_col, title in metrics_to_plot:
        out_path = os.path.join(output_dir, f"{metric_col}.png")
        plot_metric(df, metric_col, out_path, title=title, ylabel=title)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # =====================================================
    # Burada kendi klasör isimlerini gerektiğinde değiştir
    # =====================================================
    experiments = {
        "plain": "/content/drive/MyDrive/Spring Semester/medical image analysis project/outputs2/classifier_plain",
        "roi": "/content/drive/MyDrive/Spring Semester/medical image analysis project/outputs2/classifier_roi",
        "masked_roi": "/content/drive/MyDrive/Spring Semester/medical image analysis project/outputs2/classifier_masked_roi",

        "plain+condition": "/content/drive/MyDrive/Spring Semester/medical image analysis project/outputs2/conditional_plain_safe",
        "roi+condition": "/content/drive/MyDrive/Spring Semester/medical image analysis project/outputs2/conditional_roi_safe",
        "masked_roi+condition": "/content/drive/MyDrive/Spring Semester/medical image analysis project/outputs2/conditional_masked_roi_safe",
    }

    rows = []
    for exp_name, exp_dir in experiments.items():
        print(f"[INFO] Reading: {exp_name} -> {exp_dir}")
        row = read_experiment(exp_name, exp_dir)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sıralama
    df_sorted_auc = df.sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

    # Pretty summary
    summary_cols = [
        "experiment_name",
        "input_mode",
        "model_name",
        "condition_dim",
        "best_epoch",
        "best_val_auc",
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_recall",
        "test_specificity",
        "test_precision",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_loss",
    ]
    summary_df = df_sorted_auc[summary_cols].copy()

    # Sıralı rank
    summary_df.insert(0, "rank_by_test_roc_auc", np.arange(1, len(summary_df) + 1))

    # Kaydet
    full_csv = os.path.join(args.output_dir, "all_experiments_full.csv")
    summary_csv = os.path.join(args.output_dir, "comparison_summary.csv")
    markdown_table_path = os.path.join(args.output_dir, "comparison_summary.md")

    df.to_csv(full_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(markdown_table_path, "w", encoding="utf-8") as f:
        f.write(summary_df.to_markdown(index=False))

    # Grafikler
    plot_top_metrics_grid(summary_df, args.output_dir)

    # En iyi model bilgisi
    best_row = summary_df.iloc[0].to_dict()

    best_json = os.path.join(args.output_dir, "best_model_summary.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("[INFO] COMPARISON FINISHED")
    print("=" * 100)
    print(f"[INFO] Full CSV        : {full_csv}")
    print(f"[INFO] Summary CSV     : {summary_csv}")
    print(f"[INFO] Markdown Table  : {markdown_table_path}")
    print(f"[INFO] Best Model JSON : {best_json}")
    print(f"[INFO] Plots Dir       : {args.output_dir}")
    print("=" * 100)

    print("\nTOP MODELS BY TEST ROC-AUC")
    print(summary_df[[
        "rank_by_test_roc_auc",
        "experiment_name",
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_recall",
        "test_specificity"
    ]])


if __name__ == "__main__":
    main()