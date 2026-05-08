#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#d
"""
compare_roi_sources.py

Amaç:
    Farklı ROI kaynakları ile eğitilmiş / test edilmiş classifier sonuçlarını
    karşılaştırmak.

Desteklenen ROI örnekleri:
    - full
    - lung_mask
    - lung_crop_pad
    - lung_only
    - rsna_bbox
    - bbox_only
    - union
    - intersection

Beklenen girişler:
    1) Tek bir klasör altında birden fazla prediction / metrics dosyası
    2) veya doğrudan birden fazla dosya yolu

Desteklenen dosya tipleri:
    - prediction csv
    - metrics csv
    - metrics json

Prediction CSV beklenen sütunlar:
    - patientId veya patient_id
    - y_true
    - y_prob
    - roi_source (opsiyonel; yoksa dosya adından çıkarılmaya çalışılır)
    - model_name (opsiyonel)
    - variant (opsiyonel)
    - split (opsiyonel)

Üretilen çıktılar:
    - roi_comparison_metrics.csv
    - roi_best_by_metric.csv
    - roi_pairwise_bootstrap.csv
    - roi_summary.json
    - roi_casewise_matrix.csv
    - roi_rankings.csv

Beklenen kullanım:
    python scripts/compare_roi_sources.py \
        --pred-dir outputs/classification/roi_experiments \
        --out-dir outputs/comparisons/roi_sources

veya

    python scripts/compare_roi_sources.py \
        --pred-files a.csv b.csv c.csv \
        --out-dir outputs/comparisons/roi_sources
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
import logging
from itertools import combinations
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


LOGGER = logging.getLogger("compare_roi_sources")


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
# Constants
# =========================================================

KNOWN_ROI_SOURCES = [
    "full",
    "lung_mask",
    "lung_crop_pad",
    "lung_only",
    "rsna_bbox",
    "bbox_only",
    "union",
    "intersection",
]


# =========================================================
# IO helpers
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    found = []
    if not root or not os.path.isdir(root):
        return found
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("CSV okunamadı: %s | %s", path, exc)
        return None


def safe_read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        LOGGER.warning("JSON okunamadı: %s | %s", path, exc)
        return None


# =========================================================
# Parsing helpers
# =========================================================

def infer_roi_source_from_path(path: str) -> str:
    lower = path.lower()
    for roi in sorted(KNOWN_ROI_SOURCES, key=len, reverse=True):
        if roi in lower:
            return roi
    return "unknown"


def infer_variant_from_path(path: str) -> str:
    lower = path.lower()
    if "conditional" in lower:
        return "conditional"
    if "plain" in lower:
        return "plain"
    return "default"


def infer_model_name_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    known = [
        "unet", "attention_unet", "unetpp", "transunet", "deeplabv3plus", "nnunet",
        "resnet18", "resnet34", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"
    ]
    for k in known:
        if k in stem:
            return k
    return stem


def normalize_prediction_df(df: pd.DataFrame, source_path: str) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    required_aliases = {
        "patientId": ["patientid", "patient_id"],
        "y_true": ["y_true", "target", "label"],
        "y_prob": ["y_prob", "prob", "probability", "pred_prob"]
    }

    rename_map = {}
    for canonical, aliases in required_aliases.items():
        found = None
        for alias in aliases:
            if alias in lower_map:
                found = lower_map[alias]
                break
        if found is None:
            raise ValueError(f"Gerekli sütun bulunamadı ({canonical}): {source_path}")
        rename_map[found] = canonical

    out = df.rename(columns=rename_map).copy()

    if "y_pred" not in out.columns:
        out["y_pred"] = (pd.to_numeric(out["y_prob"], errors="coerce").fillna(0.0) >= 0.5).astype(int)

    if "roi_source" not in out.columns:
        out["roi_source"] = infer_roi_source_from_path(source_path)
    else:
        out["roi_source"] = out["roi_source"].fillna(infer_roi_source_from_path(source_path)).astype(str)

    if "model_name" not in out.columns:
        out["model_name"] = infer_model_name_from_path(source_path)
    if "variant" not in out.columns:
        out["variant"] = infer_variant_from_path(source_path)
    if "split" not in out.columns:
        out["split"] = "unknown"

    out["patientId"] = out["patientId"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce").fillna(0).astype(int)
    out["y_prob"] = pd.to_numeric(out["y_prob"], errors="coerce").astype(float)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce").fillna(0).astype(int)
    out["roi_source"] = out["roi_source"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    out["variant"] = out["variant"].astype(str)
    out["source_file"] = source_path

    return out


def parse_metrics_json(path: str) -> Optional[dict]:
    data = safe_read_json(path)
    if data is None:
        return None

    roi_source = data.get("roi_source", infer_roi_source_from_path(path))
    model_name = data.get("model_name", infer_model_name_from_path(path))
    variant = data.get("variant", infer_variant_from_path(path))
    split = data.get("split", "unknown")

    return {
        "roi_source": roi_source,
        "model_name": model_name,
        "variant": variant,
        "split": split,
        "auc": data.get("auc"),
        "ap": data.get("ap", data.get("average_precision")),
        "accuracy": data.get("accuracy", data.get("acc")),
        "sensitivity": data.get("sensitivity", data.get("recall")),
        "specificity": data.get("specificity"),
        "f1": data.get("f1", data.get("f1_score")),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "brier": data.get("brier", data.get("brier_score")),
        "ece": data.get("ece"),
        "nll": data.get("nll", data.get("log_loss")),
        "num_cases": data.get("num_cases"),
        "source_file": path,
        "file_type": "json_metrics"
    }


def parse_metrics_csv(path: str) -> Optional[pd.DataFrame]:
    df = safe_read_csv(path)
    if df is None or len(df) == 0:
        return None

    out = df.copy()
    if "roi_source" not in out.columns:
        out["roi_source"] = infer_roi_source_from_path(path)
    if "model_name" not in out.columns:
        out["model_name"] = infer_model_name_from_path(path)
    if "variant" not in out.columns:
        out["variant"] = infer_variant_from_path(path)
    if "split" not in out.columns:
        out["split"] = "unknown"

    out["source_file"] = path
    out["file_type"] = "csv_metrics"
    return out


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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
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
    metrics["sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))
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
# Ranking and summaries
# =========================================================

def build_metrics_from_prediction_groups(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["roi_source", "model_name", "variant", "split"]

    for keys, sub in pred_df.groupby(group_cols):
        roi_source, model_name, variant, split = keys
        metrics = compute_metrics(sub)
        row = {
            "roi_source": roi_source,
            "model_name": model_name,
            "variant": variant,
            "split": split,
            "file_type": "predictions",
            "source_file": "grouped_predictions",
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def rank_roi_sources(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if len(metrics_df) == 0:
        return pd.DataFrame()

    df = metrics_df.copy()

    agg = (
        df.groupby("roi_source")[["auc", "ap", "accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    for col in ["auc", "ap", "accuracy", "sensitivity", "specificity", "f1"]:
        agg[f"rank_{col}"] = agg[col].rank(ascending=False, method="min")
    for col in ["brier", "ece", "nll"]:
        agg[f"rank_{col}"] = agg[col].rank(ascending=True, method="min")

    rank_cols = [c for c in agg.columns if c.startswith("rank_")]
    agg["mean_rank"] = agg[rank_cols].mean(axis=1)
    agg = agg.sort_values(["mean_rank", "auc"], ascending=[True, False]).reset_index(drop=True)
    return agg


def best_by_metric(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if len(metrics_df) == 0:
        return pd.DataFrame()

    metric_prefs = {
        "auc": False,
        "ap": False,
        "accuracy": False,
        "sensitivity": False,
        "specificity": False,
        "f1": False,
        "precision": False,
        "recall": False,
        "brier": True,
        "ece": True,
        "nll": True,
    }

    rows = []
    for metric, ascending in metric_prefs.items():
        if metric not in metrics_df.columns:
            continue
        sub = metrics_df[metrics_df[metric].notna()].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(metric, ascending=ascending)
        best = sub.iloc[0]
        rows.append({
            "metric": metric,
            "roi_source": best["roi_source"],
            "model_name": best["model_name"],
            "variant": best["variant"],
            "split": best["split"],
            "metric_value": float(best[metric]),
        })

    return pd.DataFrame(rows)


# =========================================================
# Pairwise bootstrap
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
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "specificity":
        return float(specificity_score(y_true, y_pred))
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric_name == "brier":
        return float(brier_score_loss(y_true, y_prob))
    if metric_name == "ece":
        return float(expected_calibration_error(y_true, y_prob))
    if metric_name == "nll":
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    raise ValueError(f"Desteklenmeyen metric: {metric_name}")


def build_pairwise_bootstrap(
    pred_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    metrics: List[str]
) -> pd.DataFrame:
    rows = []
    rng = np.random.RandomState(seed)

    roi_groups = {}
    for roi, sub in pred_df.groupby("roi_source"):
        roi_groups[roi] = sub.copy()

    for roi_a, roi_b in combinations(sorted(roi_groups.keys()), 2):
        a = roi_groups[roi_a]
        b = roi_groups[roi_b]

        merged = a[["patientId", "y_true", "y_prob"]].merge(
            b[["patientId", "y_true", "y_prob"]],
            on="patientId",
            how="inner",
            suffixes=("_a", "_b")
        )

        if len(merged) < 10:
            continue

        mismatch = (merged["y_true_a"].astype(int) != merged["y_true_b"].astype(int)).sum()
        if mismatch > 0:
            LOGGER.warning("y_true uyuşmazlığı var: %s vs %s | %d örnek", roi_a, roi_b, mismatch)
            continue

        y_true = merged["y_true_a"].values.astype(int)
        p_a = merged["y_prob_a"].values.astype(float)
        p_b = merged["y_prob_b"].values.astype(float)
        n = len(merged)

        for metric_name in metrics:
            deltas = []
            for _ in range(n_boot):
                idx = rng.randint(0, n, size=n)
                yt = y_true[idx]
                pa = p_a[idx]
                pb = p_b[idx]
                try:
                    m_a = metric_from_arrays(metric_name, yt, pa)
                    m_b = metric_from_arrays(metric_name, yt, pb)
                    deltas.append(m_b - m_a)
                except Exception:
                    continue

            if len(deltas) == 0:
                rows.append({
                    "roi_a": roi_a,
                    "roi_b": roi_b,
                    "metric": metric_name,
                    "mean_delta_b_minus_a": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "prob_b_better": np.nan,
                    "n_boot_valid": 0,
                    "n_cases": int(n),
                })
                continue

            deltas = np.array(deltas, dtype=float)
            rows.append({
                "roi_a": roi_a,
                "roi_b": roi_b,
                "metric": metric_name,
                "mean_delta_b_minus_a": float(np.mean(deltas)),
                "ci_low": float(np.percentile(deltas, 2.5)),
                "ci_high": float(np.percentile(deltas, 97.5)),
                "prob_b_better": float((deltas > 0).mean()),
                "n_boot_valid": int(len(deltas)),
                "n_cases": int(n),
            })

    return pd.DataFrame(rows)


# =========================================================
# Casewise matrix
# =========================================================

def build_casewise_matrix(pred_df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for roi, sub in pred_df.groupby("roi_source"):
        tmp = sub[["patientId", "y_true", "y_prob"]].copy()
        tmp = tmp.rename(columns={"y_prob": f"y_prob__{roi}"})
        pieces.append(tmp)

    if len(pieces) == 0:
        return pd.DataFrame()

    base = pieces[0]
    for p in pieces[1:]:
        y_prob_cols = [c for c in p.columns if c.startswith("y_prob__")]
        base = base.merge(
            p[["patientId"] + y_prob_cols],
            on="patientId",
            how="outer"
        )

    # y_true için tek kolon tut
    if "y_true" not in base.columns:
        y_true_cols = [c for c in base.columns if c.startswith("y_true")]
        if y_true_cols:
            base["y_true"] = base[y_true_cols[0]]
    return base


# =========================================================
# Collection
# =========================================================

def collect_prediction_files(pred_dir: Optional[str], pred_files: Optional[List[str]]) -> List[str]:
    files = []
    if pred_dir:
        for path in find_files(pred_dir, (".csv",)):
            name = os.path.basename(path).lower()
            if "pred" in name or "prediction" in name:
                files.append(path)
    if pred_files:
        files.extend(pred_files)

    uniq = []
    seen = set()
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def collect_metric_files(metrics_dir: Optional[str]) -> Tuple[List[str], List[str]]:
    if not metrics_dir or not os.path.isdir(metrics_dir):
        return [], []
    csvs = find_files(metrics_dir, (".csv",))
    jsons = find_files(metrics_dir, (".json",))
    return csvs, jsons


# =========================================================
# Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI kaynaklarını karşılaştırır.")
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--pred-files", type=str, nargs="*", default=None)
    parser.add_argument("--metrics-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dir(args.out_dir)

    pred_files = collect_prediction_files(args.pred_dir, args.pred_files)
    LOGGER.info("Prediction dosyası sayısı: %d", len(pred_files))

    pred_frames = []
    for path in pred_files:
        df = safe_read_csv(path)
        if df is None or len(df) == 0:
            continue
        try:
            norm = normalize_prediction_df(df, path)
            pred_frames.append(norm)
        except Exception as exc:
            LOGGER.warning("Prediction dosyası atlandı: %s | %s", path, exc)

    pred_df = pd.concat(pred_frames, axis=0, ignore_index=True) if len(pred_frames) > 0 else pd.DataFrame()

    metric_rows = []
    if len(pred_df) > 0:
        pred_metrics_df = build_metrics_from_prediction_groups(pred_df)
        metric_rows.append(pred_metrics_df)

    csv_metric_files, json_metric_files = collect_metric_files(args.metrics_dir)
    for path in json_metric_files:
        rec = parse_metrics_json(path)
        if rec is not None:
            metric_rows.append(pd.DataFrame([rec]))

    for path in csv_metric_files:
        df = parse_metrics_csv(path)
        if df is not None and len(df) > 0:
            metric_rows.append(df)

    if len(metric_rows) == 0:
        raise RuntimeError("Karşılaştırılacak hiçbir prediction/metric verisi bulunamadı.")

    metrics_df = pd.concat(metric_rows, axis=0, ignore_index=True)

    preferred_cols = [
        "roi_source", "model_name", "variant", "split",
        "auc", "ap", "accuracy", "sensitivity", "specificity", "f1",
        "precision", "recall", "brier", "ece", "nll", "num_cases",
        "file_type", "source_file"
    ]
    existing_cols = [c for c in preferred_cols if c in metrics_df.columns] + \
                    [c for c in metrics_df.columns if c not in preferred_cols]
    metrics_df = metrics_df[existing_cols].copy()

    metrics_csv = os.path.join(args.out_dir, "roi_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    best_df = best_by_metric(metrics_df)
    best_csv = os.path.join(args.out_dir, "roi_best_by_metric.csv")
    best_df.to_csv(best_csv, index=False)

    rankings_df = rank_roi_sources(metrics_df)
    rankings_csv = os.path.join(args.out_dir, "roi_rankings.csv")
    rankings_df.to_csv(rankings_csv, index=False)

    pairwise_df = pd.DataFrame()
    casewise_df = pd.DataFrame()

    if len(pred_df) > 0 and pred_df["roi_source"].nunique() >= 2:
        pairwise_df = build_pairwise_bootstrap(
            pred_df=pred_df,
            n_boot=args.bootstrap_iters,
            seed=args.seed,
            metrics=["auc", "ap", "accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"]
        )
        pairwise_csv = os.path.join(args.out_dir, "roi_pairwise_bootstrap.csv")
        pairwise_df.to_csv(pairwise_csv, index=False)

        casewise_df = build_casewise_matrix(pred_df)
        casewise_csv = os.path.join(args.out_dir, "roi_casewise_matrix.csv")
        casewise_df.to_csv(casewise_csv, index=False)
    else:
        pairwise_csv = os.path.join(args.out_dir, "roi_pairwise_bootstrap.csv")
        casewise_csv = os.path.join(args.out_dir, "roi_casewise_matrix.csv")
        pd.DataFrame().to_csv(pairwise_csv, index=False)
        pd.DataFrame().to_csv(casewise_csv, index=False)

    summary = {
        "num_metric_rows": int(len(metrics_df)),
        "num_unique_roi_sources": int(metrics_df["roi_source"].nunique()) if "roi_source" in metrics_df.columns else 0,
        "roi_sources": sorted(metrics_df["roi_source"].dropna().astype(str).unique().tolist()) if "roi_source" in metrics_df.columns else [],
        "best_by_auc": None,
        "best_by_ap": None,
        "best_by_accuracy": None,
        "best_by_f1": None,
    }

    for metric_name in ["auc", "ap", "accuracy", "f1"]:
        if metric_name in best_df["metric"].values:
            row = best_df[best_df["metric"] == metric_name].iloc[0]
            summary[f"best_by_{metric_name}"] = {
                "roi_source": row["roi_source"],
                "model_name": row["model_name"],
                "variant": row["variant"],
                "split": row["split"],
                "metric_value": float(row["metric_value"]),
            }

    summary_json = os.path.join(args.out_dir, "roi_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    LOGGER.info("Tamamlandı.")
    LOGGER.info(" - %s", metrics_csv)
    LOGGER.info(" - %s", best_csv)
    LOGGER.info(" - %s", rankings_csv)
    LOGGER.info(" - %s", pairwise_csv)
    LOGGER.info(" - %s", casewise_csv)
    LOGGER.info(" - %s", summary_json)


if __name__ == "__main__":
    main()