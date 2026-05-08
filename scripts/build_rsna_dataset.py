#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_rsna_dataset.py

Amaç
-----
RSNA Pneumonia Detection Challenge eğitim verisini proje pipeline'ına uygun
patient-level ve bbox-level metadata tablolarına dönüştürmek.

Temel tasarım kararı
--------------------
Bu scriptte ana çalışma görüntü kaynağı PNG'dir.
DICOM klasörü yalnızca:
1) patientId doğrulaması
2) opsiyonel header metadata çıkarımı
için kullanılır.

Bu script artık eğitim / ROI / classifier tarafında kullanılacak merkezi veri tablosunu üretir.
Raw DICOM fallback mantığı YOKTUR.

Yaptıkları
----------
1) stage_2_train_labels.csv dosyasını okur
2) stage_2_detailed_class_info.csv varsa okur
3) metadata_clean.csv varsa okur
4) train PNG klasörünü ana görüntü kaynağı olarak indexler
5) opsiyonel train DICOM klasörünü indexler ve istenirse header metadata okur
6) patient bazında birleşik master metadata üretir
7) bbox-level metadata üretir
8) stratified train / val / test split oluşturur
9) CSV ve JSON özet çıktıları yazar

Notlar
------
- Split yalnızca label'lı train verisi içinden oluşturulur.
- stage_2_test_images bu scriptte kullanılmaz.
- Bounding box bilgisi stage_2_train_labels.csv içindeki x,y,width,height sütunlarından alınır.
- Aynı patientId birden fazla satıra sahip olabilir; patient-level tabloda tekilleştirilir.
"""

from __future__ import annotations

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import yaml
except ImportError:
    yaml = None


LOGGER = logging.getLogger("build_rsna_dataset")


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
# Path / YAML helpers
# =========================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(yaml_path: Optional[str]) -> Dict[str, Any]:
    if yaml_path is None:
        return {}

    if yaml is None:
        raise ImportError("PyYAML gerekli. Lütfen `pip install pyyaml` kur.")

    yaml_file = Path(yaml_path)
    if not yaml_file.is_absolute():
        yaml_file = get_project_root() / yaml_file

    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML bulunamadı: {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_output_dir(cfg: Dict[str, Any], cli_out_dir: Optional[str]) -> str:
    if cli_out_dir:
        return cli_out_dir

    return (
        deep_get(cfg, ["outputs", "rsna_dataset_dir"]) or
        deep_get(cfg, ["data", "rsna_dataset_dir"]) or
        str(get_project_root() / "outputs" / "rsna_dataset")
    )


# =========================================================
# Generic helpers
# =========================================================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def safe_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    return pd.read_csv(path)


def json_dumps_compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(value)
    except Exception:
        return default


# =========================================================
# DICOM header helpers
# =========================================================

def load_dicom_header(dicom_path: str) -> Dict[str, object]:
    if pydicom is None:
        raise ImportError("pydicom kurulu değil. `pip install pydicom` gerekli.")

    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

    pixel_spacing = getattr(ds, "PixelSpacing", None)
    if pixel_spacing is not None:
        try:
            pixel_spacing = [float(v) for v in pixel_spacing]
        except Exception:
            pixel_spacing = None

    return {
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "modality": str(getattr(ds, "Modality", "")),
        "view_position": str(getattr(ds, "ViewPosition", "")),
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "")),
        "pixel_spacing": pixel_spacing,
        "study_description": str(getattr(ds, "StudyDescription", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
    }


# =========================================================
# Validation / normalization
# =========================================================

def validate_labels_df(df: pd.DataFrame) -> None:
    required = {"patientId", "x", "y", "width", "height", "Target"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"stage_2_train_labels.csv eksik sütunlar: {missing}")


def normalize_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_labels_df(df)

    df["patientId"] = normalize_string_series(df["patientId"])

    for col in ["x", "y", "width", "height"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Target"] = pd.to_numeric(df["Target"], errors="coerce").fillna(0).astype(int)

    return df


def normalize_class_info_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    if "patientId" not in df.columns:
        raise ValueError("stage_2_detailed_class_info.csv içinde 'patientId' sütunu yok.")

    df = df.copy()
    df["patientId"] = normalize_string_series(df["patientId"])

    if "class" in df.columns:
        df["class"] = normalize_string_series(df["class"])
    else:
        df["class"] = "Unknown"

    return df[["patientId", "class"]].drop_duplicates(subset=["patientId"])


def normalize_metadata_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    id_col = None
    for cand in ["patient_id", "patientId", "id"]:
        if cand in df.columns:
            id_col = cand
            break

    if id_col is None:
        raise ValueError("metadata CSV içinde 'patient_id' / 'patientId' / 'id' sütunu yok.")

    df = df.copy()
    df[id_col] = normalize_string_series(df[id_col])
    df = df.rename(columns={id_col: "patientId"})

    if "patient_age" not in df.columns:
        if "age" in df.columns:
            df["patient_age"] = df["age"]
        else:
            df["patient_age"] = np.nan
    df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")

    if "patient_sex" not in df.columns:
        if "sex" in df.columns:
            df["patient_sex"] = df["sex"]
        elif "gender" in df.columns:
            df["patient_sex"] = df["gender"]
        else:
            df["patient_sex"] = ""
    df["patient_sex"] = normalize_string_series(df["patient_sex"])

    for col in ["study_description", "series_description", "filename"]:
        if col not in df.columns:
            df[col] = ""

    keep_cols = [
        "patientId",
        "patient_age",
        "patient_sex",
        "study_description",
        "series_description",
        "filename",
    ]
    return df[keep_cols].drop_duplicates(subset=["patientId"])


# =========================================================
# Index builders
# =========================================================

def build_png_index(png_dir: str) -> pd.DataFrame:
    if not png_dir:
        raise ValueError("PNG klasörü boş bırakılamaz.")
    if not os.path.isdir(png_dir):
        raise FileNotFoundError(f"PNG klasörü bulunamadı: {png_dir}")

    rows = []
    for name in sorted(os.listdir(png_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            patient_id = os.path.splitext(name)[0]
            rows.append({
                "patientId": str(patient_id).strip(),
                "image_path": os.path.join(png_dir, name),
                "image_name": name,
                "image_ext": os.path.splitext(name)[1].lower(),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"PNG klasöründe görsel bulunamadı: {png_dir}")

    df = df.drop_duplicates(subset=["patientId"]).reset_index(drop=True)
    return df


def build_dicom_index(dicom_dir: Optional[str]) -> Optional[pd.DataFrame]:
    if dicom_dir is None or str(dicom_dir).strip() == "":
        return None

    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM klasörü bulunamadı: {dicom_dir}")

    rows = []
    for name in sorted(os.listdir(dicom_dir)):
        if name.lower().endswith(".dcm"):
            patient_id = os.path.splitext(name)[0]
            rows.append({
                "patientId": str(patient_id).strip(),
                "dicom_path": os.path.join(dicom_dir, name),
                "dicom_name": name,
            })

    if len(rows) == 0:
        LOGGER.warning("DICOM klasöründe .dcm dosyası bulunamadı: %s", dicom_dir)
        return pd.DataFrame(columns=["patientId", "dicom_path", "dicom_name"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["patientId"]).reset_index(drop=True)
    return df


# =========================================================
# BBox helpers
# =========================================================

def build_bbox_summary_from_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for patient_id, g in labels_df.groupby("patientId", sort=True):
        g = g.copy()

        positive_rows = g[g["Target"] == 1].copy()
        bbox_count = int(len(positive_rows))
        has_bbox = int(bbox_count > 0)
        target = int(has_bbox)

        bbox_list = []
        for _, row in positive_rows.iterrows():
            bbox_list.append({
                "x": safe_float(row["x"]),
                "y": safe_float(row["y"]),
                "w": safe_float(row["width"]),
                "h": safe_float(row["height"]),
            })

        if bbox_count > 0:
            x1 = min(float(b["x"]) for b in bbox_list)
            y1 = min(float(b["y"]) for b in bbox_list)
            x2 = max(float(b["x"]) + float(b["w"]) for b in bbox_list)
            y2 = max(float(b["y"]) + float(b["h"]) for b in bbox_list)
            union_x = x1
            union_y = y1
            union_w = x2 - x1
            union_h = y2 - y1
            bbox_area_sum = float(sum(float(b["w"]) * float(b["h"]) for b in bbox_list))
        else:
            union_x = np.nan
            union_y = np.nan
            union_w = np.nan
            union_h = np.nan
            bbox_area_sum = 0.0

        rows.append({
            "patientId": patient_id,
            "target": target,
            "label": target,
            "has_bbox": has_bbox,
            "bbox_count": bbox_count,
            "num_boxes": bbox_count,
            "bbox_json": json_dumps_compact(bbox_list),
            "bbox_union_x": union_x,
            "bbox_union_y": union_y,
            "bbox_union_w": union_w,
            "bbox_union_h": union_h,
            "bbox_area_sum": bbox_area_sum,
        })

    return pd.DataFrame(rows)


def build_bbox_level_table(
    labels_df: pd.DataFrame,
    png_df: pd.DataFrame,
    class_info_df: Optional[pd.DataFrame] = None,
    dicom_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    bbox_df = labels_df.copy()

    bbox_df["has_box"] = (
        (bbox_df["Target"] == 1) &
        bbox_df["x"].notna() &
        bbox_df["y"].notna() &
        bbox_df["width"].notna() &
        bbox_df["height"].notna()
    ).astype(int)

    bbox_df["bbox_area"] = (
        bbox_df["width"].fillna(0.0).astype(float) *
        bbox_df["height"].fillna(0.0).astype(float)
    )

    bbox_df = bbox_df.merge(
        png_df[["patientId", "image_path", "image_name"]],
        on="patientId",
        how="left"
    )

    if dicom_df is not None:
        bbox_df = bbox_df.merge(
            dicom_df[["patientId", "dicom_path"]],
            on="patientId",
            how="left"
        )
    else:
        bbox_df["dicom_path"] = ""

    if class_info_df is not None:
        bbox_df = bbox_df.merge(class_info_df, on="patientId", how="left")
    else:
        bbox_df["class"] = "Unknown"

    bbox_df["image_exists"] = bbox_df["image_path"].apply(
        lambda p: int(isinstance(p, str) and os.path.exists(p))
    )

    bbox_df["class"] = bbox_df["class"].fillna("Unknown").astype(str)
    bbox_df["dicom_path"] = bbox_df["dicom_path"].fillna("").astype(str)

    return bbox_df


# =========================================================
# Patient-level table
# =========================================================

def maybe_attach_dicom_headers(
    patient_df: pd.DataFrame,
    dicom_df: Optional[pd.DataFrame],
    read_dicom_headers: bool,
) -> pd.DataFrame:
    df = patient_df.copy()

    if dicom_df is None:
        df["dicom_path"] = ""
        df["has_dicom"] = 0
        df["missing_dicom"] = 1
        df["image_height"] = np.nan
        df["image_width"] = np.nan
        df["modality"] = ""
        df["view_position"] = ""
        df["photometric_interpretation"] = ""
        df["pixel_spacing"] = ""
        df["dicom_study_description"] = ""
        df["dicom_series_description"] = ""
        return df

    df = df.merge(dicom_df[["patientId", "dicom_path"]], on="patientId", how="left")
    df["dicom_path"] = df["dicom_path"].fillna("").astype(str)
    df["has_dicom"] = df["dicom_path"].apply(lambda p: int(bool(p) and os.path.exists(p)))
    df["missing_dicom"] = (df["has_dicom"] == 0).astype(int)

    df["image_height"] = np.nan
    df["image_width"] = np.nan
    df["modality"] = ""
    df["view_position"] = ""
    df["photometric_interpretation"] = ""
    df["pixel_spacing"] = ""
    df["dicom_study_description"] = ""
    df["dicom_series_description"] = ""

    if not read_dicom_headers:
        return df

    if pydicom is None:
        LOGGER.warning("pydicom yok; DICOM header metadata atlanıyor.")
        return df

    header_rows = []
    for _, row in tqdm(
        df[["patientId", "dicom_path", "has_dicom"]].iterrows(),
        total=len(df),
        desc="Reading DICOM headers"
    ):
        patient_id = row["patientId"]
        dicom_path = row["dicom_path"]
        has_dicom = int(row["has_dicom"])

        out = {
            "patientId": patient_id,
            "image_height": np.nan,
            "image_width": np.nan,
            "modality": "",
            "view_position": "",
            "photometric_interpretation": "",
            "pixel_spacing": "",
            "dicom_study_description": "",
            "dicom_series_description": "",
        }

        if has_dicom == 1 and dicom_path:
            try:
                header = load_dicom_header(dicom_path)
                out["image_height"] = safe_int(header.get("rows", np.nan), default=0)
                out["image_width"] = safe_int(header.get("cols", np.nan), default=0)
                out["modality"] = str(header.get("modality", ""))
                out["view_position"] = str(header.get("view_position", ""))
                out["photometric_interpretation"] = str(header.get("photometric_interpretation", ""))
                out["pixel_spacing"] = json_dumps_compact(header.get("pixel_spacing", None))
                out["dicom_study_description"] = str(header.get("study_description", ""))
                out["dicom_series_description"] = str(header.get("series_description", ""))
            except Exception as exc:
                LOGGER.warning("DICOM header okunamadı | patientId=%s | error=%s", patient_id, str(exc))

        header_rows.append(out)

    header_df = pd.DataFrame(header_rows)
    df = df.drop(
        columns=[
            "image_height",
            "image_width",
            "modality",
            "view_position",
            "photometric_interpretation",
            "pixel_spacing",
            "dicom_study_description",
            "dicom_series_description",
        ],
        errors="ignore"
    ).merge(header_df, on="patientId", how="left")

    return df


def compute_bbox_derived_fields(patient_df: pd.DataFrame) -> pd.DataFrame:
    df = patient_df.copy()

    def _bbox_area_ratio(row):
        h = row.get("image_height", np.nan)
        w = row.get("image_width", np.nan)
        area_sum = safe_float(row.get("bbox_area_sum", 0.0), default=0.0)
        if pd.notna(h) and pd.notna(w) and float(h) > 0 and float(w) > 0:
            return float(area_sum / (float(h) * float(w)))
        return np.nan

    def _laterality(row):
        bbox_json = row.get("bbox_json", "[]")
        image_width = row.get("image_width", np.nan)

        try:
            bbox_list = json.loads(bbox_json) if isinstance(bbox_json, str) else []
        except Exception:
            bbox_list = []

        if not bbox_list:
            return "none"

        if pd.isna(image_width) or float(image_width) <= 0:
            return "unknown"

        mid_x = float(image_width) / 2.0
        left_count = 0
        right_count = 0

        for b in bbox_list:
            cx = safe_float(b.get("x", 0.0)) + safe_float(b.get("w", 0.0)) / 2.0
            if cx < mid_x:
                left_count += 1
            else:
                right_count += 1

        if left_count > 0 and right_count > 0:
            return "bilateral"
        if left_count > 0:
            return "left"
        if right_count > 0:
            return "right"
        return "none"

    df["bbox_area_ratio"] = df.apply(_bbox_area_ratio, axis=1)
    df["laterality"] = df.apply(_laterality, axis=1)

    return df


def build_patient_level_table(
    labels_df: pd.DataFrame,
    png_df: pd.DataFrame,
    class_info_df: Optional[pd.DataFrame] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    dicom_df: Optional[pd.DataFrame] = None,
    read_dicom_headers: bool = False,
) -> pd.DataFrame:
    bbox_summary_df = build_bbox_summary_from_labels(labels_df)

    patient_df = bbox_summary_df.merge(
        png_df[["patientId", "image_path", "image_name", "image_ext"]],
        on="patientId",
        how="left"
    )

    patient_df["image_exists"] = patient_df["image_path"].apply(
        lambda p: int(isinstance(p, str) and os.path.exists(p))
    )
    patient_df["has_png"] = patient_df["image_exists"].astype(int)
    patient_df["missing_png"] = (patient_df["has_png"] == 0).astype(int)

    if class_info_df is not None:
        patient_df = patient_df.merge(class_info_df, on="patientId", how="left")
    else:
        patient_df["class"] = "Unknown"
    patient_df["class"] = patient_df["class"].fillna("Unknown").astype(str)

    if metadata_df is not None:
        patient_df = patient_df.merge(metadata_df, on="patientId", how="left")
    else:
        patient_df["patient_age"] = np.nan
        patient_df["patient_sex"] = ""
        patient_df["study_description"] = ""
        patient_df["series_description"] = ""
        patient_df["filename"] = ""

    patient_df["patient_sex"] = patient_df["patient_sex"].fillna("").astype(str)
    patient_df["study_description"] = patient_df["study_description"].fillna("").astype(str)
    patient_df["series_description"] = patient_df["series_description"].fillna("").astype(str)
    patient_df["filename"] = patient_df["filename"].fillna("").astype(str)

    patient_df = maybe_attach_dicom_headers(
        patient_df=patient_df,
        dicom_df=dicom_df,
        read_dicom_headers=read_dicom_headers,
    )

    patient_df = compute_bbox_derived_fields(patient_df)

    patient_df["source_dataset"] = "rsna_stage2_train"
    patient_df["source_split"] = "train_source"
    patient_df["preprocess_ok"] = patient_df["has_png"].astype(int)

    ordered_cols = [
        "patientId",
        "image_path",
        "image_name",
        "image_ext",
        "image_exists",
        "has_png",
        "missing_png",
        "target",
        "label",
        "has_bbox",
        "bbox_count",
        "num_boxes",
        "bbox_json",
        "bbox_union_x",
        "bbox_union_y",
        "bbox_union_w",
        "bbox_union_h",
        "bbox_area_sum",
        "bbox_area_ratio",
        "laterality",
        "class",
        "patient_age",
        "patient_sex",
        "study_description",
        "series_description",
        "filename",
        "dicom_path",
        "has_dicom",
        "missing_dicom",
        "image_height",
        "image_width",
        "modality",
        "view_position",
        "photometric_interpretation",
        "pixel_spacing",
        "dicom_study_description",
        "dicom_series_description",
        "source_dataset",
        "source_split",
        "preprocess_ok",
    ]

    for c in ordered_cols:
        if c not in patient_df.columns:
            patient_df[c] = np.nan

    patient_df = patient_df[ordered_cols].sort_values("patientId").reset_index(drop=True)
    return patient_df


# =========================================================
# Split helpers
# =========================================================

def _stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_size: float,
    test_size: float,
    seed: int
) -> pd.DataFrame:
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size + test_size toplamı 1'den küçük olmalı.")

    if len(df) == 0:
        raise ValueError("Split yapılacak veri boş.")

    if label_col not in df.columns:
        raise ValueError(f"Split için label sütunu bulunamadı: {label_col}")

    work_df = df.copy().reset_index(drop=True)
    y = work_df[label_col].astype(int)

    train_df, temp_df = train_test_split(
        work_df,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=seed
    )

    temp_ratio = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=temp_ratio,
        stratify=temp_df[label_col].astype(int),
        random_state=seed
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    return out


def build_split_csv(
    patient_df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int
) -> pd.DataFrame:
    split_df = _stratified_split(
        df=patient_df[["patientId", "label"]].copy(),
        label_col="label",
        val_size=val_size,
        test_size=test_size,
        seed=seed
    )
    return split_df[["patientId", "split"]].sort_values(["split", "patientId"]).reset_index(drop=True)


# =========================================================
# Summary
# =========================================================

def split_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name].copy()
        n = len(sub)
        n_pos = int(sub["label"].sum()) if n > 0 else 0
        n_neg = int((sub["label"] == 0).sum()) if n > 0 else 0

        rows.append({
            "split": split_name,
            "num_cases": int(n),
            "num_positive": int(n_pos),
            "num_negative": int(n_neg),
            "positive_ratio": float(n_pos / n) if n > 0 else 0.0
        })

    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "class" not in df.columns:
        return df.groupby(["split", "label"]).size().reset_index(name="count")
    return df.groupby(["split", "class", "label"]).size().reset_index(name="count")


def build_summary_json(
    patient_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    split_df: pd.DataFrame,
    split_dist_df: pd.DataFrame,
) -> dict:
    total_patients = len(patient_df)
    total_positive = int(patient_df["label"].sum()) if total_patients > 0 else 0
    total_negative = int((patient_df["label"] == 0).sum()) if total_patients > 0 else 0
    total_bbox_rows = len(bbox_df)
    total_positive_bbox_rows = int((bbox_df["has_box"] == 1).sum()) if total_bbox_rows > 0 else 0

    payload = {
        "num_patients": int(total_patients),
        "num_positive_patients": int(total_positive),
        "num_negative_patients": int(total_negative),
        "positive_patient_ratio": float(total_positive / total_patients) if total_patients > 0 else 0.0,
        "num_bbox_rows": int(total_bbox_rows),
        "num_positive_bbox_rows": int(total_positive_bbox_rows),
        "num_missing_pngs": int(patient_df["missing_png"].sum()) if "missing_png" in patient_df.columns else 0,
        "num_missing_dicoms": int(patient_df["missing_dicom"].sum()) if "missing_dicom" in patient_df.columns else 0,
        "split_distribution": split_dist_df.to_dict(orient="records"),
        "split_counts_by_patient": split_df["split"].value_counts().to_dict(),
    }

    if "class" in patient_df.columns:
        class_counts = patient_df["class"].fillna("Unknown").astype(str).value_counts(dropna=False).to_dict()
        payload["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}

    if "patient_sex" in patient_df.columns:
        sex_counts = patient_df["patient_sex"].fillna("Unknown").astype(str).value_counts(dropna=False).to_dict()
        payload["patient_sex_counts"] = {str(k): int(v) for k, v in sex_counts.items()}

    return payload


# =========================================================
# Main process
# =========================================================

def process_rsna_dataset(
    png_dir: str,
    labels_csv: str,
    out_dir: str,
    detailed_class_info_csv: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    dicom_dir: Optional[str] = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    drop_missing_pngs: bool = True,
    read_dicom_headers: bool = False,
) -> None:
    ensure_dir(out_dir)

    LOGGER.info("Labels okunuyor...")
    labels_df = safe_read_csv(labels_csv)
    if labels_df is None:
        raise FileNotFoundError(f"Labels CSV bulunamadı: {labels_csv}")
    labels_df = normalize_labels_df(labels_df)

    LOGGER.info("PNG index oluşturuluyor...")
    png_df = build_png_index(png_dir)

    LOGGER.info("Class info okunuyor...")
    class_info_df = normalize_class_info_df(safe_read_csv(detailed_class_info_csv))

    LOGGER.info("Metadata okunuyor...")
    metadata_df = normalize_metadata_df(safe_read_csv(metadata_csv))

    dicom_df = None
    if dicom_dir:
        LOGGER.info("Opsiyonel DICOM index oluşturuluyor...")
        dicom_df = build_dicom_index(dicom_dir)

    LOGGER.info("Patient-level tablo oluşturuluyor...")
    patient_df = build_patient_level_table(
        labels_df=labels_df,
        png_df=png_df,
        class_info_df=class_info_df,
        metadata_df=metadata_df,
        dicom_df=dicom_df,
        read_dicom_headers=read_dicom_headers,
    )

    LOGGER.info("BBox-level tablo oluşturuluyor...")
    bbox_df = build_bbox_level_table(
        labels_df=labels_df,
        png_df=png_df,
        class_info_df=class_info_df,
        dicom_df=dicom_df,
    )

    if drop_missing_pngs:
        before_pat = len(patient_df)
        before_box = len(bbox_df)

        patient_df = patient_df[patient_df["has_png"] == 1].copy().reset_index(drop=True)
        bbox_df = bbox_df[bbox_df["image_exists"] == 1].copy().reset_index(drop=True)

        LOGGER.info(
            "Eksik PNG kayıtları çıkarıldı | patient: %d -> %d | bbox rows: %d -> %d",
            before_pat, len(patient_df), before_box, len(bbox_df)
        )

    if len(patient_df) == 0:
        raise RuntimeError("Patient-level tablo boş kaldı. PNG eşleşmeleri ve labels kontrol edilmeli.")

    LOGGER.info("Split CSV oluşturuluyor...")
    split_only_df = build_split_csv(
        patient_df=patient_df,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    patient_df = patient_df.merge(split_only_df, on="patientId", how="left")
    bbox_df = bbox_df.merge(split_only_df, on="patientId", how="left")

    split_dist_df = split_distribution(patient_df)
    class_dist_df = class_distribution(patient_df)

    master_csv = os.path.join(out_dir, "master_dataset.csv")
    patient_meta_csv = os.path.join(out_dir, "rsna_patient_metadata.csv")
    bbox_meta_csv = os.path.join(out_dir, "rsna_bbox_metadata.csv")
    split_csv = os.path.join(out_dir, "split.csv")
    train_csv = os.path.join(out_dir, "rsna_train.csv")
    val_csv = os.path.join(out_dir, "rsna_val.csv")
    test_csv = os.path.join(out_dir, "rsna_test.csv")
    split_dist_csv = os.path.join(out_dir, "rsna_split_distribution.csv")
    class_dist_csv = os.path.join(out_dir, "rsna_class_distribution.csv")
    summary_json_path = os.path.join(out_dir, "rsna_dataset_summary.json")

    LOGGER.info("Çıktılar yazılıyor...")

    patient_df.to_csv(master_csv, index=False)
    patient_df.to_csv(patient_meta_csv, index=False)
    bbox_df.to_csv(bbox_meta_csv, index=False)
    split_only_df.to_csv(split_csv, index=False)

    patient_df[patient_df["split"] == "train"].to_csv(train_csv, index=False)
    patient_df[patient_df["split"] == "val"].to_csv(val_csv, index=False)
    patient_df[patient_df["split"] == "test"].to_csv(test_csv, index=False)

    split_dist_df.to_csv(split_dist_csv, index=False)
    class_dist_df.to_csv(class_dist_csv, index=False)

    summary = build_summary_json(
        patient_df=patient_df,
        bbox_df=bbox_df,
        split_df=split_only_df,
        split_dist_df=split_dist_df,
    )

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    LOGGER.info("Tamamlandı.")
    LOGGER.info("Master CSV           : %s", master_csv)
    LOGGER.info("Patient metadata CSV : %s", patient_meta_csv)
    LOGGER.info("BBox metadata CSV    : %s", bbox_meta_csv)
    LOGGER.info("Split CSV            : %s", split_csv)
    LOGGER.info("Train CSV            : %s", train_csv)
    LOGGER.info("Val CSV              : %s", val_csv)
    LOGGER.info("Test CSV             : %s", test_csv)
    LOGGER.info("Summary JSON         : %s", summary_json_path)


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSNA train verisinden patient-level ve bbox-level merkezi dataset metadata üret."
    )

    parser.add_argument("--paths", type=str, default=None, help="İsteğe bağlı paths.yaml")
    parser.add_argument("--png-dir", type=str, default=None, help="Train PNG klasörü (ana kaynak)")
    parser.add_argument("--labels-csv", type=str, default=None, help="stage_2_train_labels.csv")
    parser.add_argument("--out-dir", type=str, default=None, help="Çıktı klasörü")

    parser.add_argument("--dicom-dir", type=str, default=None, help="Opsiyonel train DICOM klasörü")
    parser.add_argument("--detailed-class-info-csv", type=str, default=None, help="stage_2_detailed_class_info.csv")
    parser.add_argument("--metadata-csv", type=str, default=None, help="metadata_clean.csv")

    parser.add_argument("--val-size", type=float, default=0.15, help="Validation oranı")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test oranı")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--keep-missing-pngs",
        action="store_true",
        help="PNG'si eksik kayıtları koru. Varsayılan: eksik PNG kayıtları düşürülür."
    )
    parser.add_argument(
        "--read-dicom-headers",
        action="store_true",
        help="DICOM header metadata da oku (Rows, Columns, ViewPosition vs.)."
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging seviyesi")

    args = parser.parse_args()
    cfg = load_yaml_config(args.paths) if args.paths else {}

    if args.png_dir is None:
        args.png_dir = deep_get(cfg, ["data", "train_png_dir"])

    if args.labels_csv is None:
        args.labels_csv = deep_get(cfg, ["data", "train_labels_csv"])

    if args.detailed_class_info_csv is None:
        args.detailed_class_info_csv = deep_get(cfg, ["data", "detailed_class_info_csv"])

    if args.metadata_csv is None:
        args.metadata_csv = deep_get(cfg, ["data", "metadata_csv"])

    if args.dicom_dir is None:
        args.dicom_dir = deep_get(cfg, ["data", "train_dicom_dir"])

    args.out_dir = resolve_output_dir(cfg, args.out_dir)

    if not args.png_dir:
        raise ValueError("png-dir boş bırakılamaz. Bu scriptte PNG ana görüntü kaynağıdır.")
    if not args.labels_csv:
        raise ValueError("labels-csv boş bırakılamaz.")
    if not args.out_dir:
        raise ValueError("out-dir boş bırakılamaz.")

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    LOGGER.info("Başlıyor: build_rsna_dataset.py")
    LOGGER.info("png_dir=%s", args.png_dir)
    LOGGER.info("labels_csv=%s", args.labels_csv)
    LOGGER.info("out_dir=%s", args.out_dir)
    LOGGER.info("dicom_dir=%s", args.dicom_dir)
    LOGGER.info("detailed_class_info_csv=%s", args.detailed_class_info_csv)
    LOGGER.info("metadata_csv=%s", args.metadata_csv)
    LOGGER.info("val_size=%s", str(args.val_size))
    LOGGER.info("test_size=%s", str(args.test_size))
    LOGGER.info("seed=%s", str(args.seed))
    LOGGER.info("read_dicom_headers=%s", str(args.read_dicom_headers))
    LOGGER.info("drop_missing_pngs=%s", str(not args.keep_missing_pngs))

    process_rsna_dataset(
        png_dir=args.png_dir,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        detailed_class_info_csv=args.detailed_class_info_csv,
        metadata_csv=args.metadata_csv,
        dicom_dir=args.dicom_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        drop_missing_pngs=(not args.keep_missing_pngs),
        read_dicom_headers=args.read_dicom_headers,
    )


if __name__ == "__main__":
    main()