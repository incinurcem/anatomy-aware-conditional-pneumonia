#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#d
"""
compare_plain_vs_conditional.py

Amaç:
    Plain classifier sonuçları ile conditional classifier sonuçlarını
    sistematik olarak karşılaştırmak.

Karşılaştırdığı başlıklar:
    - ROC-AUC
    - PR-AUC / AP
    - Accuracy
    - Sensitivity
    - Specificity
    - F1
    - Precision
    - Recall
    - Brier score
    - ECE
    - NLL / Log loss
    - subgroup performansları
    - bootstrap fark analizi
    - opsiyonel McNemar testi

Beklenen girişler:
    1) prediction csv
       Sütunlar:
         - patientId veya patient_id
         - y_true
         - y_prob
         - y_pred (opsiyonel)
         - split (opsiyonel)
         - model_name (opsiyonel)
         - variant (opsiyonel)
         - subgroup_* (opsiyonel)
    2) plain ve conditional için ayrı prediction dosyaları

Üretilen çıktılar:
    - comparison_summary.json
    - comparison_metrics.csv
    - subgroup_comparison.csv
    - bootstrap_deltas.csv
    - paired_case_analysis.csv
    - mcnemar_summary.json (opsiyonel)

Beklenen kullanım:
    python scripts/compare_plain_vs_conditional.py \
        --plain-pred outputs/classification/plain/test_predictions.csv \
        --conditional-pred outputs/classification/conditional/test_predictions.csv \
        --out-dir outputs/comparisons/plain_vs_conditional
"""

from __future__ import annotations

import os
import json
import math
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    log_loss
)


LOGGER = logging.getLogger("compare_plain_vs_conditional")


# =========================================================
# Logging
# =========================================================

def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# =========================================================
# IO utils
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_predictions(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prediction CSV bulunamadı: {path}")

    df = pd.read_csv(path)
    lower_map = {c.lower(): c for c in df.columns}

    required_any = {
        "patientId": ["patientid", "patient_id"],
        "y_true": ["y_true", "label", "target"],
        "y_prob": ["y_prob", "prob", "probability", "pred_prob"]
    }

    rename_map = {}
    for canonical, aliases in required_any.items():
        matched = None
        for a in aliases:
            if a in lower_map:
                matched = lower_map[a]
                break
        if matched is None:
            raise ValueError(f"{path} içinde gerekli sütun bulunamadı: {canonical}")
        rename_map[matched] = canonical

    df = df.rename(columns=rename_map).copy()

    if "y_pred" not in df.columns:
        df["y_pred"] = (df["y_prob"].astype(float) >= 0.5).astype(int)

    df["patientId"] = df["patientId"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce").astype(float)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)

    return df


# =========================================================
# Metrics
# =========================================================

def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return float("nan")
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else float("nan")


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def compute_metrics(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    y_true = df["y_true"].values.astype(int)
    y_prob = df["y_prob"].values.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["ap"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["ap"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["sensitivity"] = float(sensitivity_score(y_true, y_pred))
    metrics["specificity"] = float(specificity_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics["brier"] = float("nan")

    try:
        metrics["nll"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        metrics["nll"] = float("nan")

    metrics["ece"] = float(expected_calibration_error(y_true, y_prob))
    metrics["num_cases"] = int(len(df))
    metrics["positive_ratio"] = float(y_true.mean()) if len(y_true) > 0 else float("nan")

    return metrics


# =========================================================
# Alignment
# =========================================================

def align_plain_conditional(
    plain_df: pd.DataFrame,
    conditional_df: pd.DataFrame
) -> pd.DataFrame:
    plain_cols = set(plain_df.columns)
    cond_cols = set(conditional_df.columns)

    common_extra = sorted(list((plain_cols & cond_cols) - {"patientId", "y_true", "y_prob", "y_pred"}))

    plain_keep = ["patientId", "y_true", "y_prob", "y_pred"] + common_extra
    cond_keep = ["patientId", "y_true", "y_prob", "y_pred"] + common_extra

    merged = plain_df[plain_keep].merge(
        conditional_df[cond_keep],
        on="patientId",
        how="inner",
        suffixes=("_plain", "_conditional")
    )

    if len(merged) == 0:
        raise RuntimeError("Plain ve conditional prediction dosyaları eşleşmedi.")

    mismatched = (merged["y_true_plain"].astype(int) != merged["y_true_conditional"].astype(int)).sum()
    if mismatched > 0:
        raise ValueError(f"Eşleşen kayıtlarda y_true farklı: {mismatched} örnek")

    merged["y_true"] = merged["y_true_plain"].astype(int)
    return merged


# =========================================================
# Bootstrap
# =========================================================

def metric_from_arrays(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_pred = (y_prob >= 0.5).astype(int)

    if metric_name == "auc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "ap":
        return float(average_precision_score(y_true, y_prob))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric_name == "sensitivity":
        return float(sensitivity_score(y_true, y_pred))
    if metric_name == "specificity":
        return float(specificity_score(y_true, y_pred))
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "brier":
        return float(brier_score_loss(y_true, y_prob))
    if metric_name == "ece":
        return float(expected_calibration_error(y_true, y_prob))
    if metric_name == "nll":
        return float(log_loss(y_true, y_prob, labels=[0, 1]))

    raise ValueError(f"Desteklenmeyen metrik: {metric_name}")


def bootstrap_metric_deltas(
    merged_df: pd.DataFrame,
    metrics: List[str],
    n_boot: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(merged_df)

    y_true = merged_df["y_true"].values.astype(int)
    p_plain = merged_df["y_prob_plain"].values.astype(float)
    p_cond = merged_df["y_prob_conditional"].values.astype(float)

    rows = []

    for metric_name in metrics:
        deltas = []

        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            yt = y_true[idx]
            pp = p_plain[idx]
            pc = p_cond[idx]

            try:
                m_plain = metric_from_arrays(metric_name, yt, pp)
                m_cond = metric_from_arrays(metric_name, yt, pc)
                deltas.append(m_cond - m_plain)
            except Exception:
                continue

        if len(deltas) == 0:
            rows.append({
                "metric": metric_name,
                "mean_delta": np.nan,
                "std_delta": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "prob_conditional_better": np.nan,
                "n_boot_valid": 0
            })
            continue

        deltas = np.array(deltas, dtype=float)
        rows.append({
            "metric": metric_name,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "prob_conditional_better": float((deltas > 0).mean()),
            "n_boot_valid": int(len(deltas))
        })

    return pd.DataFrame(rows)


# =========================================================
# McNemar
# =========================================================

def mcnemar_test_from_predictions(
    y_true: np.ndarray,
    pred_plain: np.ndarray,
    pred_cond: np.ndarray
) -> Dict[str, float]:
    correct_plain = (pred_plain == y_true).astype(int)
    correct_cond = (pred_cond == y_true).astype(int)

    b = int(((correct_plain == 1) & (correct_cond == 0)).sum())
    c = int(((correct_plain == 0) & (correct_cond == 1)).sum())

    if (b + c) == 0:
        return {
            "b": b,
            "c": c,
            "chi2_cc": 0.0,
            "approx_p_value": 1.0
        }

    chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
    # scipy kullanmadan yaklaşık p-value:
    approx_p = math.erfc(math.sqrt(chi2_cc / 2.0))

    return {
        "b": b,
        "c": c,
        "chi2_cc": float(chi2_cc),
        "approx_p_value": float(approx_p)
    }


# =========================================================
# Subgroups
# =========================================================

def find_subgroup_columns(df: pd.DataFrame) -> List[str]:
    subgroup_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("subgroup_"):
            subgroup_cols.append(c)
    return subgroup_cols


def subgroup_comparison(merged_df: pd.DataFrame) -> pd.DataFrame:
    subgroup_cols = find_subgroup_columns(merged_df)
    rows = []

    for col in subgroup_cols:
        values = merged_df[col].dropna().unique().tolist()
        for v in values:
            sub = merged_df[merged_df[col] == v].copy()
            if len(sub) < 5:
                continue

            plain_df = pd.DataFrame({
                "y_true": sub["y_true"].values,
                "y_prob": sub["y_prob_plain"].values
            })
            cond_df = pd.DataFrame({
                "y_true": sub["y_true"].values,
                "y_prob": sub["y_prob_conditional"].values
            })

            plain_metrics = compute_metrics(plain_df)
            cond_metrics = compute_metrics(cond_df)

            row = {
                "subgroup_column": col,
                "subgroup_value": v,
                "num_cases": int(len(sub))
            }

            for k, val in plain_metrics.items():
                row[f"{k}_plain"] = val
            for k, val in cond_metrics.items():
                row[f"{k}_conditional"] = val

            for metric_name in ["auc", "ap", "accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"]:
                pv = row.get(f"{metric_name}_plain", np.nan)
                cv = row.get(f"{metric_name}_conditional", np.nan)
                row[f"delta_{metric_name}"] = (
                    float(cv - pv) if pd.notna(pv) and pd.notna(cv) else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Paired case analysis
# =========================================================

def build_paired_case_analysis(merged_df: pd.DataFrame) -> pd.DataFrame:
    out = merged_df.copy()

    out["error_plain"] = np.abs(out["y_true"] - out["y_prob_plain"])
    out["error_conditional"] = np.abs(out["y_true"] - out["y_prob_conditional"])
    out["delta_error"] = out["error_conditional"] - out["error_plain"]

    out["pred_plain"] = (out["y_prob_plain"] >= 0.5).astype(int)
    out["pred_conditional"] = (out["y_prob_conditional"] >= 0.5).astype(int)

    out["correct_plain"] = (out["pred_plain"] == out["y_true"]).astype(int)
    out["correct_conditional"] = (out["pred_conditional"] == out["y_true"]).astype(int)

    def case_status(row):
        cp = row["correct_plain"]
        cc = row["correct_conditional"]
        if cp == 1 and cc == 1:
            return "both_correct"
        if cp == 0 and cc == 0:
            return "both_wrong"
        if cp == 1 and cc == 0:
            return "plain_only_correct"
        return "conditional_only_correct"

    out["case_status"] = out.apply(case_status, axis=1)

    keep_cols = ["patientId", "y_true", "y_prob_plain", "y_prob_conditional", "pred_plain", "pred_conditional",
                 "correct_plain", "correct_conditional", "error_plain", "error_conditional", "delta_error", "case_status"]

    for c in merged_df.columns:
        if c.startswith("subgroup_") and c not in keep_cols:
            keep_cols.append(c)

    return out[keep_cols].copy()


# =========================================================
# Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain vs Conditional classifier karşılaştırması")
    parser.add_argument("--plain-pred", type=str, required=True)
    parser.add_argument("--conditional-pred", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dir(args.out_dir)

    LOGGER.info("Plain prediction: %s", args.plain_pred)
    LOGGER.info("Conditional prediction: %s", args.conditional_pred)

    plain_df = load_predictions(args.plain_pred)
    conditional_df = load_predictions(args.conditional_pred)
    merged_df = align_plain_conditional(plain_df, conditional_df)

    LOGGER.info("Eşleşen örnek sayısı: %d", len(merged_df))

    plain_eval_df = pd.DataFrame({
        "y_true": merged_df["y_true"].values,
        "y_prob": merged_df["y_prob_plain"].values
    })
    cond_eval_df = pd.DataFrame({
        "y_true": merged_df["y_true"].values,
        "y_prob": merged_df["y_prob_conditional"].values
    })

    plain_metrics = compute_metrics(plain_eval_df)
    conditional_metrics = compute_metrics(cond_eval_df)

    comp_rows = []
    metric_order = ["auc", "ap", "accuracy", "sensitivity", "specificity", "f1", "precision", "recall", "brier", "ece", "nll"]
    for m in metric_order:
        pv = plain_metrics.get(m, np.nan)
        cv = conditional_metrics.get(m, np.nan)
        comp_rows.append({
            "metric": m,
            "plain": pv,
            "conditional": cv,
            "delta_conditional_minus_plain": (
                float(cv - pv) if pd.notna(pv) and pd.notna(cv) else np.nan
            )
        })

    comparison_df = pd.DataFrame(comp_rows)
    comparison_csv = os.path.join(args.out_dir, "comparison_metrics.csv")
    comparison_df.to_csv(comparison_csv, index=False)

    subgroup_df = subgroup_comparison(merged_df)
    subgroup_csv = os.path.join(args.out_dir, "subgroup_comparison.csv")
    subgroup_df.to_csv(subgroup_csv, index=False)

    bootstrap_df = bootstrap_metric_deltas(
        merged_df=merged_df,
        metrics=metric_order,
        n_boot=args.bootstrap_iters,
        seed=args.seed
    )
    bootstrap_csv = os.path.join(args.out_dir, "bootstrap_deltas.csv")
    bootstrap_df.to_csv(bootstrap_csv, index=False)

    paired_df = build_paired_case_analysis(merged_df)
    paired_csv = os.path.join(args.out_dir, "paired_case_analysis.csv")
    paired_df.to_csv(paired_csv, index=False)

    mcnemar = mcnemar_test_from_predictions(
        y_true=merged_df["y_true"].values.astype(int),
        pred_plain=(merged_df["y_prob_plain"].values.astype(float) >= 0.5).astype(int),
        pred_cond=(merged_df["y_prob_conditional"].values.astype(float) >= 0.5).astype(int)
    )
    mcnemar_json = os.path.join(args.out_dir, "mcnemar_summary.json")
    with open(mcnemar_json, "w", encoding="utf-8") as f:
        json.dump(mcnemar, f, indent=2, ensure_ascii=False)

    summary = {
        "num_cases": int(len(merged_df)),
        "plain_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in plain_metrics.items()},
        "conditional_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in conditional_metrics.items()},
        "best_variant_by_auc": (
            "conditional" if conditional_metrics.get("auc", np.nan) > plain_metrics.get("auc", np.nan) else "plain"
        ),
        "mcnemar": mcnemar
    }

    summary_json = os.path.join(args.out_dir, "comparison_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    LOGGER.info("Tamamlandı.")
    LOGGER.info(" - %s", comparison_csv)
    LOGGER.info(" - %s", subgroup_csv)
    LOGGER.info(" - %s", bootstrap_csv)
    LOGGER.info(" - %s", paired_csv)
    LOGGER.info(" - %s", mcnemar_json)
    LOGGER.info(" - %s", summary_json)


if __name__ == "__main__":
    main()