#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_classifier_inputs.py

Amaç:
    Segmentasyon çıktıları, bbox anotasyonları ve ROI stratejilerine göre
    classifier girişlerini üretmek.

Desteklenen ROI kaynakları:
    1) "lung_mask"     : segmentasyon maskesinden tight bbox çıkar
    2) "rsna_bbox"     : RSNA pneumonia bbox anotasyonundan ROI çıkar
    3) "union"         : lung_mask bbox ile rsna_bbox union
    4) "intersection"  : lung_mask bbox ile rsna_bbox intersection
    5) "full"          : tüm görüntü
    6) "lung_crop_pad" : lung area tight crop + square pad
    7) "lung_only"     : görüntü * mask
    8) "bbox_only"     : bbox içini koru, dışı sıfırla

Not:
    - Train için labels zorunludur.
    - Test için labels zorunlu değildir; target=0, has_bbox=0 atanır.
"""

from __future__ import annotations

import os
import json
import yaml
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger("build_classifier_inputs")


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
# Data structures
# =========================================================

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class SampleMeta:
    patient_id: str
    image_path: str
    mask_path: Optional[str]
    roi_source: str
    out_image_path: str
    out_masked_path: Optional[str]
    target: int
    split: str
    has_bbox: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int
    orig_h: int
    orig_w: int
    out_h: int
    out_w: int
    lung_area_ratio: float
    bbox_area_ratio: float


# =========================================================
# Config utils
# =========================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = get_project_root() / path_obj
    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(d: dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =========================================================
# IO utils
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_gray_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return img


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadı: {path}")
    return (mask > 127).astype(np.uint8) * 255


def write_image(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Görüntü yazılamadı: {path}")


# =========================================================
# Geometry utils
# =========================================================

def clip_box(box: BoundingBox, w: int, h: int) -> BoundingBox:
    x1 = max(0, min(box.x, max(0, w - 1)))
    y1 = max(0, min(box.y, max(0, h - 1)))
    x2 = max(0, min(box.x2, w))
    y2 = max(0, min(box.y2, h))
    return BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def union_boxes(a: Optional[BoundingBox], b: Optional[BoundingBox]) -> Optional[BoundingBox]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = max(a.x2, b.x2)
    y2 = max(a.y2, b.y2)
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def intersect_boxes(a: Optional[BoundingBox], b: Optional[BoundingBox]) -> Optional[BoundingBox]:
    if a is None or b is None:
        return None

    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def expand_box(box: BoundingBox, margin_ratio: float, w: int, h: int) -> BoundingBox:
    mx = int(round(box.width * margin_ratio))
    my = int(round(box.height * margin_ratio))
    expanded = BoundingBox(
        box.x - mx,
        box.y - my,
        box.width + 2 * mx,
        box.height + 2 * my
    )
    return clip_box(expanded, w, h)


def square_box(box: BoundingBox, w: int, h: int) -> BoundingBox:
    side = max(box.width, box.height)
    cx = box.x + box.width / 2.0
    cy = box.y + box.height / 2.0

    x = int(round(cx - side / 2.0))
    y = int(round(cy - side / 2.0))
    sq = BoundingBox(x, y, int(side), int(side))
    return clip_box(sq, w, h)


def crop_from_box(img: np.ndarray, box: BoundingBox) -> np.ndarray:
    return img[box.y:box.y2, box.x:box.x2]


# =========================================================
# Mask / bbox helpers
# =========================================================

def bbox_from_mask(mask: np.ndarray) -> Optional[BoundingBox]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return BoundingBox(x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def bbox_from_rsna_rows(rows: Optional[pd.DataFrame]) -> Optional[BoundingBox]:
    if rows is None or len(rows) == 0:
        return None

    valid = rows[(rows["Target"] == 1)]
    if len(valid) == 0:
        return None

    x1 = int(valid["x"].min())
    y1 = int(valid["y"].min())
    x2 = int((valid["x"] + valid["width"]).max())
    y2 = int((valid["y"] + valid["height"]).max())

    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def area_ratio_from_mask(mask: Optional[np.ndarray]) -> float:
    if mask is None:
        return 0.0
    total = float(mask.shape[0] * mask.shape[1])
    if total == 0:
        return 0.0
    return float((mask > 0).sum()) / total


def area_ratio_from_box(box: Optional[BoundingBox], w: int, h: int) -> float:
    if box is None or w <= 0 or h <= 0:
        return 0.0
    return float(box.width * box.height) / float(w * h)


# =========================================================
# Image processing
# =========================================================

def minmax_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)


def resize_keep_ratio_with_pad(img: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    canvas = np.full((size, size), pad_value, dtype=resized.dtype)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def mask_outside_box(img: np.ndarray, box: Optional[BoundingBox]) -> np.ndarray:
    out = np.zeros_like(img)
    if box is None or not box.is_valid():
        return out
    out[box.y:box.y2, box.x:box.x2] = img[box.y:box.y2, box.x:box.x2]
    return out


def apply_lung_only(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return img
    out = np.zeros_like(img)
    out[mask > 0] = img[mask > 0]
    return out


# =========================================================
# Dataset table builders
# =========================================================

def normalize_patient_id(value) -> str:
    return str(value).strip()


def build_image_table(images_dir: str, image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> pd.DataFrame:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir bulunamadı: {images_dir}")

    rows = []
    for name in sorted(os.listdir(images_dir)):
        lower = name.lower()
        if lower.endswith(image_exts):
            patient_id = os.path.splitext(name)[0]
            rows.append({
                "patientId": normalize_patient_id(patient_id),
                "image_path": os.path.join(images_dir, name)
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"Görüntü bulunamadı: {images_dir}")
    return df


def build_mask_table(masks_dir: Optional[str]) -> pd.DataFrame:
    if not masks_dir or not os.path.isdir(masks_dir):
        return pd.DataFrame(columns=["patientId", "mask_path"])

    rows = []
    for name in sorted(os.listdir(masks_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            patient_id = os.path.splitext(name)[0]
            rows.append({
                "patientId": normalize_patient_id(patient_id),
                "mask_path": os.path.join(masks_dir, name)
            })

    return pd.DataFrame(rows)


def read_rsna_labels(labels_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if labels_csv is None:
        return None

    df = pd.read_csv(labels_csv)
    expected = {"patientId", "x", "y", "width", "height", "Target"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError(f"RSNA labels csv eksik sütunlar: {missing}")

    df["patientId"] = df["patientId"].astype(str)
    df["Target"] = pd.to_numeric(df["Target"], errors="coerce").fillna(0).astype(int)
    return df


def patient_target_from_rows(rows: Optional[pd.DataFrame]) -> int:
    if rows is None or len(rows) == 0:
        return 0
    return int((rows["Target"] == 1).any())


def build_split_df(
    image_df: pd.DataFrame,
    split_csv: Optional[str] = None,
    split_col: str = "split",
    patient_col: str = "patientId",
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    default_split: str = "train"
) -> pd.DataFrame:
    if split_csv and os.path.isfile(split_csv):
        df = pd.read_csv(split_csv)
        if patient_col not in df.columns or split_col not in df.columns:
            raise ValueError(f"Split CSV içinde '{patient_col}' ve '{split_col}' sütunları olmalı.")
        df[patient_col] = df[patient_col].astype(str)
        return df[[patient_col, split_col]].copy()

    ids = image_df[patient_col].astype(str).tolist()
    if default_split in {"train", "val", "test"} and len(ids) > 0:
        return pd.DataFrame([{patient_col: pid, split_col: default_split} for pid in ids])

    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:n_train + n_val + n_test])

    rows = []
    for pid in ids:
        if pid in train_ids:
            split = "train"
        elif pid in val_ids:
            split = "val"
        else:
            split = "test"
        rows.append({patient_col: pid, split_col: split})

    return pd.DataFrame(rows)


# =========================================================
# ROI logic
# =========================================================

def resolve_roi_box(
    roi_source: str,
    image_shape: Tuple[int, int],
    lung_box: Optional[BoundingBox],
    rsna_box: Optional[BoundingBox],
    margin_ratio: float
) -> Optional[BoundingBox]:
    h, w = image_shape

    if roi_source == "full":
        return BoundingBox(0, 0, w, h)

    if roi_source == "lung_mask":
        box = lung_box
    elif roi_source == "rsna_bbox":
        box = rsna_box
    elif roi_source == "union":
        box = union_boxes(lung_box, rsna_box)
    elif roi_source == "intersection":
        box = intersect_boxes(lung_box, rsna_box)
    elif roi_source == "lung_crop_pad":
        box = lung_box
    elif roi_source == "lung_only":
        return BoundingBox(0, 0, w, h)
    elif roi_source == "bbox_only":
        box = rsna_box if rsna_box is not None else BoundingBox(0, 0, w, h)
    else:
        raise ValueError(f"Desteklenmeyen roi_source: {roi_source}")

    if box is None:
        return BoundingBox(0, 0, w, h)

    box = clip_box(box, w, h)
    if margin_ratio > 0:
        box = expand_box(box, margin_ratio, w, h)

    if roi_source == "lung_crop_pad":
        box = square_box(box, w, h)

    return clip_box(box, w, h)


def build_roi_image(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    roi_box: BoundingBox,
    roi_source: str,
    out_size: int,
    apply_clahe_flag: bool
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    work = image.copy()

    if apply_clahe_flag:
        work = clahe_enhance(work)

    masked_img = None

    if roi_source == "lung_only":
        masked_img = apply_lung_only(work, mask)
        roi = crop_from_box(masked_img, roi_box)
    elif roi_source == "bbox_only":
        masked_img = mask_outside_box(work, roi_box)
        roi = crop_from_box(masked_img, roi_box)
    else:
        roi = crop_from_box(work, roi_box)

    roi = minmax_uint8(roi)
    roi_resized = resize_keep_ratio_with_pad(roi, out_size, pad_value=0)

    masked_resized = None
    if masked_img is not None:
        masked_crop = crop_from_box(masked_img, roi_box)
        masked_crop = minmax_uint8(masked_crop)
        masked_resized = resize_keep_ratio_with_pad(masked_crop, out_size, pad_value=0)

    return roi_resized, masked_resized


# =========================================================
# Visualization
# =========================================================

def draw_bbox(img: np.ndarray, box: Optional[BoundingBox], color: int = 255, thickness: int = 2) -> np.ndarray:
    out = img.copy()
    if box is None or not box.is_valid():
        return out
    cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2), color, thickness)
    return out


def save_preview(
    save_path: str,
    image: np.ndarray,
    lung_box: Optional[BoundingBox],
    rsna_box: Optional[BoundingBox],
    roi_box: Optional[BoundingBox]
) -> None:
    base = image.copy()
    lung_v = draw_bbox(base, lung_box, color=180, thickness=2)
    rsna_v = draw_bbox(lung_v, rsna_box, color=220, thickness=2)
    roi_v = draw_bbox(rsna_v, roi_box, color=255, thickness=2)
    write_image(save_path, roi_v)


# =========================================================
# Main processing
# =========================================================

def process_dataset(
    images_dir: str,
    masks_dir: Optional[str],
    labels_csv: Optional[str],
    out_dir: str,
    roi_source: str,
    out_size: int,
    margin_ratio: float,
    split_csv: Optional[str],
    split_col: str,
    seed: int,
    apply_clahe_flag: bool,
    save_previews: bool,
    default_split: str = "train"
) -> None:
    ensure_dir(out_dir)
    roi_img_dir = os.path.join(out_dir, "images")
    roi_aux_dir = os.path.join(out_dir, "masked")
    preview_dir = os.path.join(out_dir, "previews")
    meta_dir = os.path.join(out_dir, "meta")

    ensure_dir(roi_img_dir)
    ensure_dir(roi_aux_dir)
    ensure_dir(meta_dir)
    if save_previews:
        ensure_dir(preview_dir)

    image_df = build_image_table(images_dir)
    mask_df = build_mask_table(masks_dir)
    label_df = read_rsna_labels(labels_csv)
    split_df = build_split_df(
        image_df=image_df,
        split_csv=split_csv,
        split_col=split_col,
        seed=seed,
        default_split=default_split
    )

    merged = image_df.merge(mask_df, on="patientId", how="left")
    merged = merged.merge(split_df, on="patientId", how="left")

    if split_col not in merged.columns:
        raise ValueError(f"Split sütunu bulunamadı: {split_col}")

    label_groups = {}
    if label_df is not None:
        label_groups = {pid: g.copy() for pid, g in label_df.groupby("patientId")}

    meta_rows: List[dict] = []

    for row in tqdm(merged.itertuples(index=False), total=len(merged), desc="ROI build"):
        patient_id = normalize_patient_id(row.patientId)
        image_path = row.image_path
        mask_path = getattr(row, "mask_path", None)
        split = getattr(row, split_col)

        image = read_gray_image(image_path)
        h, w = image.shape[:2]

        mask = None
        if isinstance(mask_path, str) and os.path.isfile(mask_path):
            mask = read_mask(mask_path)

        label_rows = label_groups.get(patient_id, None)
        target = patient_target_from_rows(label_rows)
        rsna_box = bbox_from_rsna_rows(label_rows) if label_rows is not None else None
        lung_box = bbox_from_mask(mask) if mask is not None else None

        roi_box = resolve_roi_box(
            roi_source=roi_source,
            image_shape=(h, w),
            lung_box=lung_box,
            rsna_box=rsna_box,
            margin_ratio=margin_ratio
        )

        roi_img, masked_img = build_roi_image(
            image=image,
            mask=mask,
            roi_box=roi_box,
            roi_source=roi_source,
            out_size=out_size,
            apply_clahe_flag=apply_clahe_flag
        )

        out_image_path = os.path.join(roi_img_dir, f"{patient_id}.png")
        write_image(out_image_path, roi_img)

        out_masked_path = None
        if masked_img is not None:
            out_masked_path = os.path.join(roi_aux_dir, f"{patient_id}.png")
            write_image(out_masked_path, masked_img)

        if save_previews:
            preview_path = os.path.join(preview_dir, f"{patient_id}.png")
            save_preview(preview_path, image, lung_box, rsna_box, roi_box)

        bbox_vals = rsna_box if rsna_box is not None else BoundingBox(0, 0, 0, 0)

        meta = SampleMeta(
            patient_id=patient_id,
            image_path=image_path,
            mask_path=mask_path if isinstance(mask_path, str) else None,
            roi_source=roi_source,
            out_image_path=out_image_path,
            out_masked_path=out_masked_path,
            target=int(target),
            split=str(split),
            has_bbox=int(rsna_box is not None),
            bbox_x=int(bbox_vals.x),
            bbox_y=int(bbox_vals.y),
            bbox_w=int(bbox_vals.width),
            bbox_h=int(bbox_vals.height),
            roi_x=int(roi_box.x),
            roi_y=int(roi_box.y),
            roi_w=int(roi_box.width),
            roi_h=int(roi_box.height),
            orig_h=int(h),
            orig_w=int(w),
            out_h=int(out_size),
            out_w=int(out_size),
            lung_area_ratio=float(area_ratio_from_mask(mask)),
            bbox_area_ratio=float(area_ratio_from_box(rsna_box, w, h))
        )
        meta_rows.append(asdict(meta))

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = os.path.join(meta_dir, "classifier_inputs.csv")
    meta_df.to_csv(meta_csv, index=False)

    for split_name in ["train", "val", "test"]:
        split_out = meta_df[meta_df["split"] == split_name].copy()
        split_out.to_csv(os.path.join(meta_dir, f"{split_name}.csv"), index=False)

    summary = {
        "num_samples": int(len(meta_df)),
        "num_positive": int(meta_df["target"].sum()),
        "num_negative": int((meta_df["target"] == 0).sum()),
        "roi_source": roi_source,
        "out_size": int(out_size),
        "margin_ratio": float(margin_ratio),
        "apply_clahe": bool(apply_clahe_flag),
        "split_counts": {str(k): int(v) for k, v in meta_df["split"].value_counts(dropna=False).to_dict().items()}
    }

    with open(os.path.join(meta_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    LOGGER.info("Tamamlandı.")
    LOGGER.info("Metadata CSV: %s", meta_csv)
    LOGGER.info("Özet: %s", summary)


# =========================================================
# Argument parsing
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifier giriş ROI veri seti üretimi")

    parser.add_argument(
        "--config",
        type=str,
        default="/content/drive/MyDrive/Spring Semester/medical image analysis project/configs/classifier.yaml"
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="/content/drive/MyDrive/Spring Semester/medical image analysis project/configs/paths.yaml"
    )

    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--labels-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument(
        "--roi-source",
        type=str,
        default=None,
        choices=[
            "lung_mask",
            "rsna_bbox",
            "union",
            "intersection",
            "full",
            "lung_crop_pad",
            "lung_only",
            "bbox_only"
        ]
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--margin-ratio", type=float, default=None)
    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-clahe", action="store_true")
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def resolve_runtime_args(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config) if os.path.isfile(args.config) else {}
    paths = load_yaml(args.paths) if os.path.isfile(args.paths) else {}

    if args.mode == "train":
        images_dir = args.images_dir or deep_get(paths, ["data", "train_png_dir"])
        masks_dir = (
            args.masks_dir or
            deep_get(paths, ["segmentation", "train_masks_dir"]) or
            deep_get(paths, ["data", "train_masks_dir"])
        )
        labels_csv = args.labels_csv or deep_get(paths, ["data", "train_labels_csv"])
        default_split = "train"
    else:
        images_dir = args.images_dir or deep_get(paths, ["data", "test_png_dir"])
        masks_dir = (
            args.masks_dir or
            deep_get(paths, ["segmentation", "test_masks_dir"]) or
            deep_get(paths, ["data", "test_masks_dir"])
        )
        labels_csv = args.labels_csv
        default_split = "test"

    split_csv = (
        args.split_csv or
        deep_get(paths, ["data", "master_dataset_csv"]) or
        deep_get(paths, ["outputs", "rsna_master_dataset_csv"]) or
        deep_get(paths, ["data", "split_csv"])
    )

    out_dir = (
        args.out_dir or
        deep_get(paths, ["outputs", "classifier_inputs"]) or
        deep_get(paths, ["classifier_inputs"]) or
        str(get_project_root() / "outputs" / f"classifier_inputs_{args.mode}")
    )

    roi_source = args.roi_source or deep_get(config, ["roi", "source"], "lung_crop_pad")
    image_size = args.image_size or deep_get(config, ["dataset", "image_size"], 512)

    margin_ratio = args.margin_ratio
    if margin_ratio is None:
        margin_ratio = deep_get(config, ["roi", "margin_ratio"], 0.05)

    apply_clahe_flag = bool(args.apply_clahe or deep_get(config, ["preprocess", "clahe"], False))

    if not images_dir:
        raise ValueError("images_dir çözümlenemedi.")
    if args.mode == "train" and not labels_csv:
        raise ValueError("Train modu için labels_csv zorunlu.")

    return {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "labels_csv": labels_csv,
        "out_dir": out_dir,
        "roi_source": roi_source,
        "out_size": int(image_size),
        "margin_ratio": float(margin_ratio),
        "split_csv": split_csv,
        "split_col": args.split_col,
        "seed": int(args.seed),
        "apply_clahe_flag": apply_clahe_flag,
        "save_previews": bool(args.save_previews),
        "default_split": default_split,
    }


# =========================================================
# Entry
# =========================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    runtime = resolve_runtime_args(args)

    LOGGER.info("Başlıyor: build_classifier_inputs.py")
    for k, v in runtime.items():
        LOGGER.info("%s = %s", k, v)

    process_dataset(**runtime)


if __name__ == "__main__":
    main()