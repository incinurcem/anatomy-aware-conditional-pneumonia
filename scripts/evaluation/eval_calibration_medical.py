import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from common_eval_utils import (
    calibration_bins,
    compute_calibration_slope_intercept,
    compute_ece_mce_ace,
    decision_curve_analysis,
    hosmer_lemeshow_test,
    plot_calibration_histogram,
    plot_decision_curve,
    plot_reliability_diagram,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=str, required=True, help="CSV with columns y_true and y_prob")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.predictions_csv)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError("predictions_csv must contain y_true and y_prob columns")

    y_true = df["y_true"].values
    y_prob = df["y_prob"].values

    bins_uniform = calibration_bins(y_true, y_prob, n_bins=args.n_bins, strategy="uniform")
    bins_quantile = calibration_bins(y_true, y_prob, n_bins=args.n_bins, strategy="quantile")
    bins_uniform.to_csv(os.path.join(args.output_dir, "calibration_bins_uniform.csv"), index=False)
    bins_quantile.to_csv(os.path.join(args.output_dir, "calibration_bins_quantile.csv"), index=False)

    uniform = compute_ece_mce_ace(y_true, y_prob, n_bins=args.n_bins, strategy="uniform")
    quantile = compute_ece_mce_ace(y_true, y_prob, n_bins=args.n_bins, strategy="quantile")
    reg = compute_calibration_slope_intercept(y_true, y_prob)
    hl = hosmer_lemeshow_test(y_true, y_prob, n_groups=args.n_bins)
    dca = decision_curve_analysis(y_true, y_prob)
    dca.to_csv(os.path.join(args.output_dir, "decision_curve_analysis.csv"), index=False)

    summary = {
        **{f"uniform_{k}": v for k, v in uniform.items()},
        **{f"quantile_{k}": v for k, v in quantile.items()},
        **reg,
        **hl,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "calibration_summary.csv"), index=False)
    save_json(os.path.join(args.output_dir, "calibration_summary.json"), summary)

    plot_reliability_diagram(bins_uniform, os.path.join(args.output_dir, "reliability_diagram_uniform.png"))
    plot_reliability_diagram(bins_quantile, os.path.join(args.output_dir, "reliability_diagram_quantile.png"))
    plot_calibration_histogram(y_prob, os.path.join(args.output_dir, "prediction_confidence_histogram.png"))
    plot_decision_curve(dca, os.path.join(args.output_dir, "decision_curve_analysis.png"))

    plt.figure(figsize=(6, 5))
    plt.bar(bins_uniform["bin_id"].astype(str), bins_uniform["count"])
    plt.xlabel("Uniform calibration bin")
    plt.ylabel("Sample count")
    plt.title("Calibration Bin Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "calibration_bin_counts.png"), dpi=300)
    plt.close()

    print("===== CALIBRATION EVALUATION COMPLETE =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
