"""
run_calibration.py
#d
Purpose
-------
Evaluate and optionally improve probability calibration for
binary pneumonia classification outputs.

This script supports:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Negative Log Likelihood (NLL / Log Loss)
- Temperature scaling
- Reliability diagram data export

Expected Input
--------------
A CSV file containing at least:
    y_true
    y_prob

Optional columns:
    split
    patient_id
    model_name

Outputs
-------
outputs/
    calibration_metrics.csv
    calibration_curve_before.csv
    calibration_curve_after.csv
    calibration_summary.json

Usage Example
-------------
python run_calibration.py \
    --input_csv outputs/classifier_predictions_val.csv \
    --output_dir outputs/calibration \
    --fit_temperature
"""

import os
import json
import math
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def clip_probs(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.clip(probs.astype(np.float64), eps, 1.0 - eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# =========================================================
# Metrics
# =========================================================

def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = clip_probs(y_prob)
    y_true = y_true.astype(np.float64)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return float(np.mean(y_pred == y_true))


# =========================================================
# Reliability / calibration bins
# =========================================================

def calibration_curve_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Returns per-bin reliability info:
    bin_idx, bin_lower, bin_upper, count, avg_confidence, avg_accuracy, gap
    """
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(mask.sum())

        if count == 0:
            avg_conf = 0.0
            avg_acc = 0.0
            gap = 0.0
        else:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            gap = abs(avg_acc - avg_conf)

        rows.append({
            "bin_idx": i,
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "count": count,
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,
            "gap": float(gap),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    curve = calibration_curve_binary(y_true, y_prob, n_bins=n_bins)
    total = max(1, int(curve["count"].sum()))
    ece = float(np.sum((curve["count"] / total) * curve["gap"]))
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    curve = calibration_curve_binary(y_true, y_prob, n_bins=n_bins)
    return float(curve["gap"].max() if len(curve) > 0 else 0.0)


# =========================================================
# Temperature scaling
# =========================================================

def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Binary temperature scaling on probability outputs via logits:
        p' = sigmoid(logit(p) / T)
    """
    probs = clip_probs(probs)
    logits = logit(probs)
    scaled_logits = logits / max(temperature, 1e-8)
    scaled_probs = sigmoid(scaled_logits)
    return clip_probs(scaled_probs)


def fit_temperature_grid_search(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_min: float = 0.5,
    t_max: float = 5.0,
    num_steps: int = 200
) -> Tuple[float, float]:
    """
    Fit temperature by minimizing log loss on a validation set.
    """
    temps = np.linspace(t_min, t_max, num_steps)
    best_t = 1.0
    best_loss = float("inf")

    for t in temps:
        scaled = apply_temperature_scaling(y_prob, float(t))
        loss = binary_log_loss(y_true, scaled)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    return best_t, float(best_loss)


# =========================================================
# Input handling
# =========================================================

def load_predictions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df = df.copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    df["y_prob"] = np.clip(df["y_prob"].values, 1e-8, 1.0 - 1e-8)

    return df


def maybe_filter_split(df: pd.DataFrame, split: Optional[str]) -> pd.DataFrame:
    if split is None:
        return df
    if "split" not in df.columns:
        raise ValueError("Requested --split filter, but input CSV has no 'split' column.")
    return df[df["split"].astype(str) == str(split)].copy()


# =========================================================
# Evaluation core
# =========================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.5
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_from_probs(y_true, y_prob, threshold=threshold),
        "nll": binary_log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins=n_bins),
        "mean_probability": float(np.mean(y_prob)),
        "std_probability": float(np.std(y_prob)),
    }


def evaluate_before_after(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.5,
    fit_temperature: bool = False,
    fixed_temperature: Optional[float] = None
) -> Tuple[Dict[str, float], Dict[str, float], float, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns:
      metrics_before, metrics_after, temperature, curve_before, curve_after, scaled_probs
    """
    metrics_before = compute_calibration_metrics(y_true, y_prob, n_bins=n_bins, threshold=threshold)
    curve_before = calibration_curve_binary(y_true, y_prob, n_bins=n_bins)

    temperature = 1.0
    scaled_probs = y_prob.copy()

    if fixed_temperature is not None:
        temperature = float(fixed_temperature)
        scaled_probs = apply_temperature_scaling(y_prob, temperature)

    elif fit_temperature:
        temperature, _ = fit_temperature_grid_search(y_true, y_prob)
        scaled_probs = apply_temperature_scaling(y_prob, temperature)

    metrics_after = compute_calibration_metrics(y_true, scaled_probs, n_bins=n_bins, threshold=threshold)
    curve_after = calibration_curve_binary(y_true, scaled_probs, n_bins=n_bins)

    return metrics_before, metrics_after, temperature, curve_before, curve_after, scaled_probs


# =========================================================
# Report formatting
# =========================================================

def build_metrics_dataframe(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    temperature: float,
    n_samples: int
) -> pd.DataFrame:
    keys = [
        "accuracy",
        "nll",
        "brier",
        "ece",
        "mce",
        "mean_probability",
        "std_probability",
    ]

    rows = []
    for key in keys:
        rows.append({
            "metric": key,
            "before": float(metrics_before.get(key, 0.0)),
            "after": float(metrics_after.get(key, 0.0)),
            "delta_after_minus_before": float(metrics_after.get(key, 0.0) - metrics_before.get(key, 0.0)),
            "temperature": float(temperature),
            "n_samples": int(n_samples),
        })

    return pd.DataFrame(rows)


def print_summary(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    temperature: float,
    n_samples: int
) -> None:
    print("\n============= CALIBRATION SUMMARY =============")
    print(f"Number of samples        : {n_samples}")
    print(f"Applied temperature      : {temperature:.6f}")
    print("--- BEFORE ---")
    print(f"Accuracy                 : {metrics_before['accuracy']:.6f}")
    print(f"NLL                      : {metrics_before['nll']:.6f}")
    print(f"Brier                    : {metrics_before['brier']:.6f}")
    print(f"ECE                      : {metrics_before['ece']:.6f}")
    print(f"MCE                      : {metrics_before['mce']:.6f}")
    print("--- AFTER ---")
    print(f"Accuracy                 : {metrics_after['accuracy']:.6f}")
    print(f"NLL                      : {metrics_after['nll']:.6f}")
    print(f"Brier                    : {metrics_after['brier']:.6f}")
    print(f"ECE                      : {metrics_after['ece']:.6f}")
    print(f"MCE                      : {metrics_after['mce']:.6f}")
    print("===============================================\n")


# =========================================================
# Save outputs
# =========================================================

def save_outputs(
    output_dir: str,
    metrics_df: pd.DataFrame,
    curve_before: pd.DataFrame,
    curve_after: pd.DataFrame,
    summary_json: Dict,
    calibrated_predictions: Optional[pd.DataFrame] = None
) -> None:
    ensure_dir(output_dir)

    metrics_df.to_csv(os.path.join(output_dir, "calibration_metrics.csv"), index=False)
    curve_before.to_csv(os.path.join(output_dir, "calibration_curve_before.csv"), index=False)
    curve_after.to_csv(os.path.join(output_dir, "calibration_curve_after.csv"), index=False)

    with open(os.path.join(output_dir, "calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    if calibrated_predictions is not None:
        calibrated_predictions.to_csv(
            os.path.join(output_dir, "calibrated_predictions.csv"),
            index=False
        )


# =========================================================
# Main run
# =========================================================

def run_calibration_pipeline(
    input_csv: str,
    output_dir: str,
    split: Optional[str] = None,
    n_bins: int = 10,
    threshold: float = 0.5,
    fit_temperature: bool = False,
    fixed_temperature: Optional[float] = None,
    save_calibrated_predictions: bool = True
) -> Dict:
    df = load_predictions_csv(input_csv)
    df = maybe_filter_split(df, split=split)

    if len(df) == 0:
        raise ValueError("No rows left after filtering. Check your input CSV or --split argument.")

    y_true = df["y_true"].values.astype(np.int32)
    y_prob = df["y_prob"].values.astype(np.float64)

    metrics_before, metrics_after, temperature, curve_before, curve_after, scaled_probs = evaluate_before_after(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        threshold=threshold,
        fit_temperature=fit_temperature,
        fixed_temperature=fixed_temperature
    )

    metrics_df = build_metrics_dataframe(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        temperature=temperature,
        n_samples=len(df)
    )

    summary_json = {
        "input_csv": input_csv,
        "split": split,
        "n_samples": int(len(df)),
        "n_bins": int(n_bins),
        "threshold": float(threshold),
        "temperature": float(temperature),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }

    calibrated_predictions = None
    if save_calibrated_predictions:
        calibrated_predictions = df.copy()
        calibrated_predictions["y_prob_calibrated"] = scaled_probs
        calibrated_predictions["y_pred_before"] = (calibrated_predictions["y_prob"] >= threshold).astype(int)
        calibrated_predictions["y_pred_after"] = (calibrated_predictions["y_prob_calibrated"] >= threshold).astype(int)

    save_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        curve_before=curve_before,
        curve_after=curve_after,
        summary_json=summary_json,
        calibrated_predictions=calibrated_predictions
    )

    print_summary(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        temperature=temperature,
        n_samples=len(df)
    )

    return summary_json


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and improve classifier calibration.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV containing at least y_true and y_prob columns"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/calibration",
        help="Directory to save calibration outputs"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split filter if input CSV contains a split column"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for reliability analysis"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for accuracy reporting"
    )
    parser.add_argument(
        "--fit_temperature",
        action="store_true",
        help="Fit temperature scaling using input CSV"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Use a fixed temperature value instead of fitting"
    )
    parser.add_argument(
        "--no_save_calibrated_predictions",
        action="store_true",
        help="Do not save calibrated predictions CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.fit_temperature and args.temperature is not None:
        raise ValueError("Use either --fit_temperature or --temperature, not both.")

    run_calibration_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        split=args.split,
        n_bins=args.n_bins,
        threshold=args.threshold,
        fit_temperature=args.fit_temperature,
        fixed_temperature=args.temperature,
        save_calibrated_predictions=not args.no_save_calibrated_predictions
    )


if __name__ == "__main__":
    main()