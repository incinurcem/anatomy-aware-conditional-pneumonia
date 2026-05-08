#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#d
"""
build_reports.py

Amaç:
    Projedeki segmentation, classification, calibration, uncertainty,
    radiomics, burden, QC ve karşılaştırma çıktılarından derli toplu
    rapor dosyaları üretmek.

Üretilen çıktılar:
    - summary_report.json
    - summary_report.csv
    - markdown_report.md
    - per_experiment_table.csv
    - best_models.csv
    - optional error analysis tables

Beklenen kullanım:
    python scripts/build_reports.py \
        --config configs/base.yaml \
        --paths configs/paths.yaml

Not:
    Bu script eğitim yapmaz.
    Sadece daha önce üretilmiş sonuçları okuyup raporlaştırır.
"""

from __future__ import annotations

import os
import json
import yaml
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("build_reports")


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
# Data classes
# =========================================================

@dataclass
class ExperimentRecord:
    task: str
    group: str
    model_name: str
    variant: str
    source_file: str
    split: Optional[str]
    primary_metric_name: Optional[str]
    primary_metric_value: Optional[float]
    dice: Optional[float] = None
    iou: Optional[float] = None
    hd95: Optional[float] = None
    assd: Optional[float] = None
    auc: Optional[float] = None
    ap: Optional[float] = None
    accuracy: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    nll: Optional[float] = None
    mean_uncertainty: Optional[float] = None
    num_cases: Optional[int] = None
    notes: Optional[str] = None


# =========================================================
# Config
# =========================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =========================================================
# File helpers
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("CSV okunamadı: %s | %s", path, exc)
        return None


def safe_read_json(path: str) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        LOGGER.warning("JSON okunamadı: %s | %s", path, exc)
        return None


def find_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    found = []
    if not root or not os.path.isdir(root):
        return found
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


# =========================================================
# Metric helpers
# =========================================================

HIGHER_IS_BETTER = {
    "dice": True,
    "iou": True,
    "auc": True,
    "ap": True,
    "accuracy": True,
    "sensitivity": True,
    "specificity": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "hd95": False,
    "assd": False,
    "brier": False,
    "ece": False,
    "nll": False,
    "mean_uncertainty": False,
}

PRIMARY_BY_TASK = {
    "segmentation": "dice",
    "classification": "auc",
    "calibration": "ece",
    "uncertainty": "mean_uncertainty",
    "radiomics": None,
    "burden": None,
    "qc": None,
    "comparison": None,
}


def to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def extract_metric_value(d: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            return to_float(d[k])
    return None


def choose_primary_metric(task: str, record_dict: dict) -> Tuple[Optional[str], Optional[float]]:
    metric_name = PRIMARY_BY_TASK.get(task)
    if metric_name is None:
        return None, None
    return metric_name, to_float(record_dict.get(metric_name))


# =========================================================
# Parsers
# =========================================================

def infer_model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    known = [
        "unet", "attention_unet", "unetpp", "transunet", "deeplabv3plus", "nnunet",
        "plain", "conditional", "resnet18", "resnet34", "resnet50", "densenet121",
        "efficientnet_b0", "convnext_tiny"
    ]

    lower = stem.lower()
    for k in known:
        if k in lower:
            return k
    return stem


def infer_variant_from_path(path: str) -> str:
    lower = path.lower()
    if "conditional" in lower:
        return "conditional"
    if "plain" in lower:
        return "plain"
    return "default"


def infer_group_from_path(path: str) -> str:
    lower = path.lower()
    if "seg" in lower or "segmentation" in lower:
        return "segmentation"
    if "classifier" in lower or "classification" in lower:
        return "classification"
    if "calibration" in lower:
        return "calibration"
    if "uncertainty" in lower:
        return "uncertainty"
    if "radiomics" in lower:
        return "radiomics"
    if "burden" in lower:
        return "burden"
    if "qc" in lower:
        return "qc"
    if "compare" in lower:
        return "comparison"
    return "misc"


def parse_segmentation_json(path: str) -> Optional[ExperimentRecord]:
    data = safe_read_json(path)
    if data is None:
        return None

    record = {
        "task": "segmentation",
        "group": infer_group_from_path(path),
        "model_name": data.get("model_name", infer_model_name_from_path(path)),
        "variant": data.get("variant", infer_variant_from_path(path)),
        "source_file": path,
        "split": data.get("split"),
        "dice": extract_metric_value(data, ["dice", "val_dice", "test_dice", "mean_dice"]),
        "iou": extract_metric_value(data, ["iou", "val_iou", "test_iou", "mean_iou"]),
        "hd95": extract_metric_value(data, ["hd95", "val_hd95", "test_hd95"]),
        "assd": extract_metric_value(data, ["assd", "val_assd", "test_assd"]),
        "num_cases": to_int(data.get("num_cases")),
        "notes": data.get("notes")
    }
    pm_name, pm_value = choose_primary_metric("segmentation", record)
    record["primary_metric_name"] = pm_name
    record["primary_metric_value"] = pm_value
    return ExperimentRecord(**record)


def parse_classification_json(path: str) -> Optional[ExperimentRecord]:
    data = safe_read_json(path)
    if data is None:
        return None

    record = {
        "task": "classification",
        "group": infer_group_from_path(path),
        "model_name": data.get("model_name", infer_model_name_from_path(path)),
        "variant": data.get("variant", infer_variant_from_path(path)),
        "source_file": path,
        "split": data.get("split"),
        "auc": extract_metric_value(data, ["auc", "roc_auc", "val_auc", "test_auc"]),
        "ap": extract_metric_value(data, ["ap", "average_precision", "val_ap", "test_ap"]),
        "accuracy": extract_metric_value(data, ["accuracy", "acc", "val_accuracy", "test_accuracy"]),
        "sensitivity": extract_metric_value(data, ["sensitivity", "recall_pos", "tpr"]),
        "specificity": extract_metric_value(data, ["specificity", "tnr"]),
        "f1": extract_metric_value(data, ["f1", "f1_score"]),
        "precision": extract_metric_value(data, ["precision"]),
        "recall": extract_metric_value(data, ["recall"]),
        "brier": extract_metric_value(data, ["brier", "brier_score"]),
        "ece": extract_metric_value(data, ["ece"]),
        "nll": extract_metric_value(data, ["nll", "log_loss"]),
        "num_cases": to_int(data.get("num_cases")),
        "notes": data.get("notes")
    }
    pm_name, pm_value = choose_primary_metric("classification", record)
    record["primary_metric_name"] = pm_name
    record["primary_metric_value"] = pm_value
    return ExperimentRecord(**record)


def parse_calibration_json(path: str) -> Optional[ExperimentRecord]:
    data = safe_read_json(path)
    if data is None:
        return None

    record = {
        "task": "calibration",
        "group": infer_group_from_path(path),
        "model_name": data.get("model_name", infer_model_name_from_path(path)),
        "variant": data.get("variant", infer_variant_from_path(path)),
        "source_file": path,
        "split": data.get("split"),
        "auc": extract_metric_value(data, ["auc", "roc_auc"]),
        "brier": extract_metric_value(data, ["brier", "brier_score"]),
        "ece": extract_metric_value(data, ["ece"]),
        "nll": extract_metric_value(data, ["nll", "log_loss"]),
        "accuracy": extract_metric_value(data, ["accuracy", "acc"]),
        "num_cases": to_int(data.get("num_cases")),
        "notes": data.get("notes")
    }
    pm_name, pm_value = choose_primary_metric("calibration", record)
    record["primary_metric_name"] = pm_name
    record["primary_metric_value"] = pm_value
    return ExperimentRecord(**record)


def parse_uncertainty_json(path: str) -> Optional[ExperimentRecord]:
    data = safe_read_json(path)
    if data is None:
        return None

    mean_unc = extract_metric_value(
        data,
        ["mean_uncertainty", "avg_uncertainty", "predictive_entropy_mean", "uncertainty_mean"]
    )

    record = {
        "task": "uncertainty",
        "group": infer_group_from_path(path),
        "model_name": data.get("model_name", infer_model_name_from_path(path)),
        "variant": data.get("variant", infer_variant_from_path(path)),
        "source_file": path,
        "split": data.get("split"),
        "auc": extract_metric_value(data, ["auc", "roc_auc"]),
        "accuracy": extract_metric_value(data, ["accuracy", "acc"]),
        "mean_uncertainty": mean_unc,
        "num_cases": to_int(data.get("num_cases")),
        "notes": data.get("notes")
    }
    pm_name, pm_value = choose_primary_metric("uncertainty", record)
    record["primary_metric_name"] = pm_name
    record["primary_metric_value"] = pm_value
    return ExperimentRecord(**record)


def parse_generic_csv(path: str, task: str) -> Optional[ExperimentRecord]:
    df = safe_read_csv(path)
    if df is None or len(df) == 0:
        return None

    row = df.iloc[0].to_dict()
    record = {
        "task": task,
        "group": infer_group_from_path(path),
        "model_name": infer_model_name_from_path(path),
        "variant": infer_variant_from_path(path),
        "source_file": path,
        "split": row.get("split"),
        "dice": to_float(row.get("dice")),
        "iou": to_float(row.get("iou")),
        "hd95": to_float(row.get("hd95")),
        "assd": to_float(row.get("assd")),
        "auc": to_float(row.get("auc")),
        "ap": to_float(row.get("ap")),
        "accuracy": to_float(row.get("accuracy")),
        "sensitivity": to_float(row.get("sensitivity")),
        "specificity": to_float(row.get("specificity")),
        "f1": to_float(row.get("f1")),
        "precision": to_float(row.get("precision")),
        "recall": to_float(row.get("recall")),
        "brier": to_float(row.get("brier")),
        "ece": to_float(row.get("ece")),
        "nll": to_float(row.get("nll")),
        "mean_uncertainty": to_float(row.get("mean_uncertainty")),
        "num_cases": to_int(row.get("num_cases")),
        "notes": None
    }
    pm_name, pm_value = choose_primary_metric(task, record)
    record["primary_metric_name"] = pm_name
    record["primary_metric_value"] = pm_value
    return ExperimentRecord(**record)


# =========================================================
# Collection
# =========================================================

def collect_records(results_root: str) -> List[ExperimentRecord]:
    records: List[ExperimentRecord] = []

    json_files = find_files(results_root, (".json",))
    csv_files = find_files(results_root, (".csv",))

    for path in json_files:
        lower = path.lower()

        rec = None
        if "seg" in lower or "segmentation" in lower:
            rec = parse_segmentation_json(path)
        elif "calibration" in lower:
            rec = parse_calibration_json(path)
        elif "uncertainty" in lower:
            rec = parse_uncertainty_json(path)
        elif "classifier" in lower or "classification" in lower or "metrics" in lower:
            rec = parse_classification_json(path)

        if rec is not None:
            records.append(rec)

    for path in csv_files:
        lower = path.lower()
        rec = None

        if "seg_metrics" in lower:
            rec = parse_generic_csv(path, "segmentation")
        elif "classification" in lower or "cls_metrics" in lower:
            rec = parse_generic_csv(path, "classification")
        elif "calibration" in lower:
            rec = parse_generic_csv(path, "calibration")
        elif "uncertainty" in lower:
            rec = parse_generic_csv(path, "uncertainty")
        elif "radiomics" in lower:
            rec = parse_generic_csv(path, "radiomics")
        elif "burden" in lower:
            rec = parse_generic_csv(path, "burden")
        elif "qc" in lower:
            rec = parse_generic_csv(path, "qc")
        elif "compare" in lower:
            rec = parse_generic_csv(path, "comparison")

        if rec is not None:
            records.append(rec)

    return records


# =========================================================
# Reporting
# =========================================================

def records_to_dataframe(records: List[ExperimentRecord]) -> pd.DataFrame:
    if len(records) == 0:
        return pd.DataFrame(columns=list(ExperimentRecord.__annotations__.keys()))
    return pd.DataFrame([asdict(r) for r in records])


def select_best_models(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    best_rows = []

    for task in sorted(df["task"].dropna().unique()):
        sub = df[df["task"] == task].copy()
        if len(sub) == 0:
            continue

        metric_name = PRIMARY_BY_TASK.get(task)
        if metric_name is None or metric_name not in sub.columns:
            continue

        sub = sub[sub[metric_name].notna()].copy()
        if len(sub) == 0:
            continue

        higher = HIGHER_IS_BETTER.get(metric_name, True)
        sub = sub.sort_values(metric_name, ascending=not higher)
        best_rows.append(sub.iloc[0])

    if len(best_rows) == 0:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(best_rows).reset_index(drop=True)


def build_task_summary(df: pd.DataFrame) -> dict:
    summary = {}

    if len(df) == 0:
        return summary

    for task in sorted(df["task"].dropna().unique()):
        sub = df[df["task"] == task].copy()
        if len(sub) == 0:
            continue

        metric_name = PRIMARY_BY_TASK.get(task)
        task_info = {
            "num_experiments": int(len(sub)),
            "variants": sorted([x for x in sub["variant"].dropna().unique().tolist()]),
            "models": sorted([x for x in sub["model_name"].dropna().unique().tolist()])
        }

        if metric_name and metric_name in sub.columns:
            valid = sub[sub[metric_name].notna()].copy()
            if len(valid) > 0:
                higher = HIGHER_IS_BETTER.get(metric_name, True)
                valid = valid.sort_values(metric_name, ascending=not higher)
                best_row = valid.iloc[0]
                task_info["primary_metric"] = metric_name
                task_info["best_model"] = best_row["model_name"]
                task_info["best_variant"] = best_row["variant"]
                task_info["best_value"] = float(best_row[metric_name])

        summary[task] = task_info

    return summary


def build_markdown_report(df: pd.DataFrame, best_df: pd.DataFrame, summary: dict) -> str:
    lines = []
    lines.append("# Pneumo Project Summary Report")
    lines.append("")
    lines.append("Bu rapor segmentation, classification, calibration, uncertainty ve ilgili alt modüllerden toplanan çıktıları özetler.")
    lines.append("")

    lines.append("## Genel Özet")
    lines.append("")

    if not summary:
        lines.append("Herhangi bir sonuç bulunamadı.")
        return "\n".join(lines)

    for task, info in summary.items():
        lines.append(f"### {task.capitalize()}")
        lines.append(f"- Deney sayısı: {info.get('num_experiments', 0)}")
        lines.append(f"- Modeller: {', '.join(info.get('models', [])) if info.get('models') else '-'}")
        lines.append(f"- Varyantlar: {', '.join(info.get('variants', [])) if info.get('variants') else '-'}")
        if "primary_metric" in info:
            lines.append(
                f"- En iyi sonuç: {info.get('best_model')} | "
                f"{info.get('best_variant')} | "
                f"{info.get('primary_metric')}={info.get('best_value'):.6f}"
            )
        lines.append("")

    lines.append("## En İyi Modeller")
    lines.append("")

    if len(best_df) == 0:
        lines.append("En iyi model bilgisi yok.")
    else:
        cols = ["task", "model_name", "variant", "primary_metric_name", "primary_metric_value", "source_file"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in best_df[cols].iterrows():
            vals = []
            for c in cols:
                val = row[c]
                if c == "primary_metric_value" and pd.notna(val):
                    vals.append(f"{float(val):.6f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Tüm Deneyler")
    lines.append("")

    if len(df) == 0:
        lines.append("Veri yok.")
    else:
        lines.append(f"Toplam kayıt: {len(df)}")
        lines.append("")

    return "\n".join(lines)


# =========================================================
# Error analysis
# =========================================================

def build_error_analysis(results_root: str, out_dir: str) -> None:
    pred_files = find_files(results_root, (".csv",))
    frames = []

    for path in pred_files:
        name = os.path.basename(path).lower()
        if "pred" not in name and "prediction" not in name:
            continue

        df = safe_read_csv(path)
        if df is None or len(df) == 0:
            continue

        lower_cols = {c.lower(): c for c in df.columns}
        needed = ["y_true", "y_prob"]
        if not all(c in lower_cols for c in needed):
            continue

        work = pd.DataFrame({
            "y_true": df[lower_cols["y_true"]],
            "y_prob": df[lower_cols["y_prob"]],
        })

        if "patientid" in lower_cols:
            work["patientId"] = df[lower_cols["patientid"]]
        elif "patient_id" in lower_cols:
            work["patientId"] = df[lower_cols["patient_id"]]
        else:
            work["patientId"] = np.arange(len(work))

        work["source_file"] = path
        work["error"] = np.abs(work["y_true"].astype(float) - work["y_prob"].astype(float))
        work["pred_label"] = (work["y_prob"].astype(float) >= 0.5).astype(int)
        work["is_wrong"] = (work["pred_label"].astype(int) != work["y_true"].astype(int)).astype(int)
        frames.append(work)

    if len(frames) == 0:
        return

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.sort_values(["is_wrong", "error"], ascending=[False, False]).reset_index(drop=True)

    merged.to_csv(os.path.join(out_dir, "error_analysis_all.csv"), index=False)
    merged.head(200).to_csv(os.path.join(out_dir, "top_200_errors.csv"), index=False)


# =========================================================
# Main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projedeki tüm sonuçlardan özet rapor üretir.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--paths", type=str, default="configs/paths.yaml")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def resolve_runtime(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config) if os.path.isfile(args.config) else {}
    paths = load_yaml(args.paths) if os.path.isfile(args.paths) else {}

    results_root = (
        args.results_root or
        deep_get(paths, ["outputs", "root"]) or
        deep_get(paths, ["results_root"]) or
        "outputs"
    )

    out_dir = (
        args.out_dir or
        deep_get(paths, ["reports", "root"]) or
        deep_get(paths, ["outputs", "reports"]) or
        os.path.join(results_root, "reports")
    )

    return {
        "results_root": results_root,
        "out_dir": out_dir,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    runtime = resolve_runtime(args)

    results_root = runtime["results_root"]
    out_dir = runtime["out_dir"]

    ensure_dir(out_dir)

    LOGGER.info("Sonuçlar okunuyor: %s", results_root)
    records = collect_records(results_root)
    df = records_to_dataframe(records)

    per_exp_csv = os.path.join(out_dir, "per_experiment_table.csv")
    df.to_csv(per_exp_csv, index=False)

    best_df = select_best_models(df)
    best_csv = os.path.join(out_dir, "best_models.csv")
    best_df.to_csv(best_csv, index=False)

    summary = build_task_summary(df)
    summary_json_path = os.path.join(out_dir, "summary_report.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_csv_rows = []
    for task, info in summary.items():
        summary_csv_rows.append({
            "task": task,
            "num_experiments": info.get("num_experiments"),
            "primary_metric": info.get("primary_metric"),
            "best_model": info.get("best_model"),
            "best_variant": info.get("best_variant"),
            "best_value": info.get("best_value"),
            "models": ", ".join(info.get("models", [])),
            "variants": ", ".join(info.get("variants", [])),
        })
    summary_csv = os.path.join(out_dir, "summary_report.csv")
    pd.DataFrame(summary_csv_rows).to_csv(summary_csv, index=False)

    md_text = build_markdown_report(df, best_df, summary)
    md_path = os.path.join(out_dir, "markdown_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    build_error_analysis(results_root, out_dir)

    LOGGER.info("Raporlar oluşturuldu:")
    LOGGER.info(" - %s", per_exp_csv)
    LOGGER.info(" - %s", best_csv)
    LOGGER.info(" - %s", summary_json_path)
    LOGGER.info(" - %s", summary_csv)
    LOGGER.info(" - %s", md_path)


if __name__ == "__main__":
    main()