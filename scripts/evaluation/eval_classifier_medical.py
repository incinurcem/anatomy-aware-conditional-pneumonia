import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common_eval_utils import (
    bootstrap_metric,
    build_model,
    compute_binary_metrics,
    compute_calibration_slope_intercept,
    compute_ece_mce_ace,
    compute_prob_metrics,
    create_loader,
    decision_curve_analysis,
    find_best_threshold_youden,
    find_threshold_for_target,
    hosmer_lemeshow_test,
    load_model_weights,
    plot_calibration_histogram,
    plot_confusion_matrix,
    plot_decision_curve,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_roc_curve,
    calibration_bins,
    run_inference,
    save_json,
)
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="image_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mean", type=float, default=0.485)
    parser.add_argument("--std", type=float, default=0.229)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--calibration_strategy", type=str, default="uniform", choices=["uniform", "quantile"])
    parser.add_argument("--threshold_mode", type=str, default="youden", choices=["youden", "fixed", "target_sensitivity", "target_specificity"])
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--target_value", type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader = create_loader(
        csv_path=args.test_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=args.mean,
        std=args.std,
    )
    model = build_model(model_name=args.model_name, in_channels=1, pretrained=False)
    model = load_model_weights(model, args.model_path, device)

    y_true, y_prob, image_paths = run_inference(model, loader, device)
    raw_df = pd.DataFrame({"image_path": image_paths, "y_true": y_true, "y_prob": y_prob})
    raw_df.to_csv(os.path.join(args.output_dir, "predictions_raw.csv"), index=False)

    threshold_search_df = None
    if args.threshold_mode == "youden":
        threshold, threshold_search_df = find_best_threshold_youden(y_true, y_prob)
        threshold_search_df.to_csv(os.path.join(args.output_dir, "threshold_search_youden.csv"), index=False)
    elif args.threshold_mode == "fixed":
        threshold = float(args.fixed_threshold)
    elif args.threshold_mode == "target_sensitivity":
        threshold, threshold_search_df = find_threshold_for_target(y_true, y_prob, "sensitivity", args.target_value)
        threshold_search_df.to_csv(os.path.join(args.output_dir, "threshold_search_target_sensitivity.csv"), index=False)
    else:
        threshold, threshold_search_df = find_threshold_for_target(y_true, y_prob, "specificity", args.target_value)
        threshold_search_df.to_csv(os.path.join(args.output_dir, "threshold_search_target_specificity.csv"), index=False)

    prob_metrics = compute_prob_metrics(y_true, y_prob)
    binary_metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    cal_uniform = compute_ece_mce_ace(y_true, y_prob, n_bins=args.n_bins, strategy="uniform")
    cal_quantile = compute_ece_mce_ace(y_true, y_prob, n_bins=args.n_bins, strategy="quantile")
    cal_reg = compute_calibration_slope_intercept(y_true, y_prob)
    hl = hosmer_lemeshow_test(y_true, y_prob, n_groups=args.n_bins)

    summary = {
        **prob_metrics,
        **binary_metrics,
        **{f"uniform_{k}": v for k, v in cal_uniform.items()},
        **{f"quantile_{k}": v for k, v in cal_quantile.items()},
        **cal_reg,
        **hl,
    }

    y_pred = (y_prob >= threshold).astype(np.uint8)
    pred_df = pd.DataFrame({
        "image_path": image_paths,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold_used": threshold,
        "correct": (y_pred == y_true).astype(np.uint8),
    })
    pred_df.to_csv(os.path.join(args.output_dir, "predictions_with_labels.csv"), index=False)

    bins_uniform = calibration_bins(y_true, y_prob, n_bins=args.n_bins, strategy="uniform")
    bins_quantile = calibration_bins(y_true, y_prob, n_bins=args.n_bins, strategy="quantile")
    bins_uniform.to_csv(os.path.join(args.output_dir, "calibration_bins_uniform.csv"), index=False)
    bins_quantile.to_csv(os.path.join(args.output_dir, "calibration_bins_quantile.csv"), index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    save_json(os.path.join(args.output_dir, "metrics_summary.json"), summary)

    def make_prob_metric(name):
        if name == "auroc":
            return lambda yt, yp: compute_prob_metrics(yt, yp)["auroc"]
        if name == "auprc":
            return lambda yt, yp: compute_prob_metrics(yt, yp)["auprc"]
        if name == "brier":
            return lambda yt, yp: compute_prob_metrics(yt, yp)["brier"]
        if name == "nll":
            return lambda yt, yp: compute_prob_metrics(yt, yp)["nll"]
        raise ValueError(name)

    def make_bin_metric(name):
        return lambda yt, yp: compute_binary_metrics(yt, yp, threshold=threshold)[name]

    ci_rows = []
    for metric_name in [
        "auroc", "auprc", "brier", "nll",
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "ppv", "npv", "f1", "mcc", "cohen_kappa", "jaccard",
        "lr_plus", "lr_minus", "diagnostic_odds_ratio",
    ]:
        fn = make_prob_metric(metric_name) if metric_name in {"auroc", "auprc", "brier", "nll"} else make_bin_metric(metric_name)
        mean_val, lo, hi = bootstrap_metric(y_true, y_prob, fn, n_boot=args.n_boot, seed=42)
        ci_rows.append({"metric": metric_name, "mean": mean_val, "ci_lower_95": lo, "ci_upper_95": hi})
    pd.DataFrame(ci_rows).to_csv(os.path.join(args.output_dir, "metrics_bootstrap_ci.csv"), index=False)

    dca_df = decision_curve_analysis(y_true, y_prob)
    dca_df.to_csv(os.path.join(args.output_dir, "decision_curve_analysis.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.savetxt(os.path.join(args.output_dir, "confusion_matrix.txt"), cm, fmt="%d")

    plot_roc_curve(y_true, y_prob, os.path.join(args.output_dir, "roc_curve.png"))
    plot_pr_curve(y_true, y_prob, os.path.join(args.output_dir, "pr_curve.png"))
    plot_reliability_diagram(bins_uniform, os.path.join(args.output_dir, "reliability_diagram_uniform.png"))
    plot_reliability_diagram(bins_quantile, os.path.join(args.output_dir, "reliability_diagram_quantile.png"))
    plot_calibration_histogram(y_prob, os.path.join(args.output_dir, "prediction_confidence_histogram.png"))
    plot_confusion_matrix(cm, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_decision_curve(dca_df, os.path.join(args.output_dir, "decision_curve_analysis.png"))

    print("===== MEDICAL CLASSIFICATION EVALUATION COMPLETE =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
