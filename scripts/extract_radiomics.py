"""
extract_radiomics.py

Purpose
-------
Extract handcrafted radiomics-style features from chest X-ray images
using a lung mask or ROI mask.

This implementation is designed to be lightweight and robust for
project usage without requiring a full PyRadiomics installation.

Inputs
------
1) images_dir:
   usually preprocessed images or ROI images

2) masks_dir:
   binary lung / ROI masks

3) Optional labels CSV:
   RSNA labels CSV, used only to attach target labels

Outputs
-------
radiomics_features.csv

Feature Groups
--------------
- Basic geometry:
    area, bbox area, extent
- First-order intensity:
    mean, std, min, max, median, p10, p25, p75, p90
- Distribution:
    skewness, kurtosis, energy, entropy
- Texture:
    GLCM contrast, dissimilarity, homogeneity, ASM, energy, correlation
- Histogram:
    normalized histogram bins
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------
# YAML / PATH helpers
# ---------------------------------------------------------

def get_project_root_from_this_file() -> Path:
    # .../<project_root>/scripts/extract_radiomics.py
    return Path(__file__).resolve().parents[1]


def deep_get(d: Dict[str, Any], keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_yaml_config(yaml_path: Optional[str]) -> Dict[str, Any]:
    if yaml_path is None:
        return {}

    project_root = get_project_root_from_this_file()
    yaml_file = Path(yaml_path)

    if not yaml_file.is_absolute():
        yaml_file = project_root / yaml_file

    if not yaml_file.exists():
        return {}

    if yaml is None:
        raise ImportError("PyYAML is required to load YAML config files.")

    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: Optional[str], project_root: Optional[Path] = None) -> Optional[str]:
    if path_value is None:
        return None

    path_value = str(path_value).strip()
    if path_value == "":
        return None

    p = Path(path_value)
    if p.is_absolute():
        return str(p)

    if project_root is None:
        project_root = get_project_root_from_this_file()

    return str((project_root / p).resolve())


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def load_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_binary_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    mask = (mask > 0).astype(np.uint8)
    return mask


def find_existing_file(base_dir: str, patient_id: str, extensions: List[str]) -> Optional[str]:
    for ext in extensions:
        path = os.path.join(base_dir, f"{patient_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def get_masked_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = image[mask > 0]
    return vals.astype(np.float32)


# ---------------------------------------------------------
# Geometry features
# ---------------------------------------------------------

def compute_geometry_features(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask > 0)

    area = int(mask.sum())

    if len(xs) == 0 or len(ys) == 0:
        return {
            "roi_area": 0.0,
            "roi_bbox_area": 0.0,
            "roi_extent": 0.0,
            "roi_aspect_ratio": 0.0,
            "roi_fill_ratio": 0.0,
        }

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bbox_w = int(x_max - x_min + 1)
    bbox_h = int(y_max - y_min + 1)
    bbox_area = int(bbox_w * bbox_h)

    extent = safe_div(area, bbox_area)
    aspect_ratio = safe_div(bbox_w, bbox_h)
    fill_ratio = safe_div(area, mask.shape[0] * mask.shape[1])

    return {
        "roi_area": float(area),
        "roi_bbox_area": float(bbox_area),
        "roi_extent": float(extent),
        "roi_aspect_ratio": float(aspect_ratio),
        "roi_fill_ratio": float(fill_ratio),
    }


# ---------------------------------------------------------
# First-order intensity features
# ---------------------------------------------------------

def compute_first_order_features(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "int_mean": 0.0,
            "int_std": 0.0,
            "int_min": 0.0,
            "int_max": 0.0,
            "int_median": 0.0,
            "int_p10": 0.0,
            "int_p25": 0.0,
            "int_p75": 0.0,
            "int_p90": 0.0,
            "int_range": 0.0,
            "int_iqr": 0.0,
        }

    p10 = np.percentile(values, 10)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    p90 = np.percentile(values, 90)

    return {
        "int_mean": float(np.mean(values)),
        "int_std": float(np.std(values)),
        "int_min": float(np.min(values)),
        "int_max": float(np.max(values)),
        "int_median": float(np.median(values)),
        "int_p10": float(p10),
        "int_p25": float(p25),
        "int_p75": float(p75),
        "int_p90": float(p90),
        "int_range": float(np.max(values) - np.min(values)),
        "int_iqr": float(p75 - p25),
    }


# ---------------------------------------------------------
# Distribution features
# ---------------------------------------------------------

def compute_distribution_features(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "dist_skewness": 0.0,
            "dist_kurtosis": 0.0,
            "dist_energy": 0.0,
            "dist_entropy": 0.0,
        }

    mean = np.mean(values)
    std = np.std(values)

    if std < 1e-8:
        skewness = 0.0
        kurtosis = 0.0
    else:
        centered = values - mean
        skewness = np.mean((centered / std) ** 3)
        kurtosis = np.mean((centered / std) ** 4)

    values_norm = values / 255.0
    energy = np.sum(values_norm ** 2) / max(1, len(values_norm))

    hist, _ = np.histogram(values, bins=32, range=(0, 256), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log2(hist))

    return {
        "dist_skewness": float(skewness),
        "dist_kurtosis": float(kurtosis),
        "dist_energy": float(energy),
        "dist_entropy": float(entropy),
    }


# ---------------------------------------------------------
# Histogram features
# ---------------------------------------------------------

def compute_histogram_features(values: np.ndarray, bins: int = 16) -> Dict[str, float]:
    feats = {}
    if values.size == 0:
        for i in range(bins):
            feats[f"hist_bin_{i:02d}"] = 0.0
        return feats

    hist, _ = np.histogram(values, bins=bins, range=(0, 256), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)

    for i in range(bins):
        feats[f"hist_bin_{i:02d}"] = float(hist[i])

    return feats


# ---------------------------------------------------------
# GLCM helpers
# ---------------------------------------------------------

def quantize_image(image: np.ndarray, levels: int = 16) -> np.ndarray:
    """
    Quantize grayscale image from [0,255] -> [0, levels-1]
    """
    if levels <= 1:
        raise ValueError("levels must be > 1")
    q = np.floor(image.astype(np.float32) / 256.0 * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    return q


def compute_glcm_matrix(
    image_q: np.ndarray,
    mask: np.ndarray,
    dx: int,
    dy: int,
    levels: int,
) -> np.ndarray:
    """
    Compute masked GLCM for one offset.
    """
    h, w = image_q.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for y in range(h):
        ny = y + dy
        if ny < 0 or ny >= h:
            continue

        for x in range(w):
            nx = x + dx
            if nx < 0 or nx >= w:
                continue

            if mask[y, x] > 0 and mask[ny, nx] > 0:
                i = image_q[y, x]
                j = image_q[ny, nx]
                glcm[i, j] += 1.0

    total = glcm.sum()
    if total > 0:
        glcm /= total

    return glcm


def glcm_stats(glcm: np.ndarray) -> Dict[str, float]:
    """
    Standard texture measures.
    """
    levels = glcm.shape[0]
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")

    contrast = np.sum(glcm * ((i_idx - j_idx) ** 2))
    dissimilarity = np.sum(glcm * np.abs(i_idx - j_idx))
    homogeneity = np.sum(glcm / (1.0 + (i_idx - j_idx) ** 2))
    asm = np.sum(glcm ** 2)
    energy = np.sqrt(max(asm, 0.0))

    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    sigma_i = np.sqrt(np.sum(((i_idx - mu_i) ** 2) * glcm))
    sigma_j = np.sqrt(np.sum(((j_idx - mu_j) ** 2) * glcm))

    if sigma_i < 1e-12 or sigma_j < 1e-12:
        correlation = 0.0
    else:
        correlation = np.sum(((i_idx - mu_i) * (j_idx - mu_j) * glcm)) / (sigma_i * sigma_j)

    return {
        "glcm_contrast": float(contrast),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_homogeneity": float(homogeneity),
        "glcm_asm": float(asm),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation),
    }


def compute_glcm_features(image: np.ndarray, mask: np.ndarray, levels: int = 16) -> Dict[str, float]:
    """
    Average GLCM stats across 4 directions:
    0°, 45°, 90°, 135°
    """
    if mask.sum() == 0:
        return {
            "glcm_contrast": 0.0,
            "glcm_dissimilarity": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_asm": 0.0,
            "glcm_energy": 0.0,
            "glcm_correlation": 0.0,
        }

    image_q = quantize_image(image, levels=levels)

    offsets = [
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    stats_list = []

    for dx, dy in offsets:
        glcm = compute_glcm_matrix(image_q, mask, dx=dx, dy=dy, levels=levels)
        stats = glcm_stats(glcm)
        stats_list.append(stats)

    out = {}
    for key in stats_list[0].keys():
        out[key] = float(np.mean([s[key] for s in stats_list]))

    return out


# ---------------------------------------------------------
# Target label helper
# ---------------------------------------------------------

def load_target_map(labels_csv: Optional[str]) -> Dict[str, int]:
    """
    Load patient -> target map from RSNA labels CSV.
    """
    if labels_csv is None or not os.path.exists(labels_csv):
        return {}

    df = pd.read_csv(labels_csv)

    if "patientId" not in df.columns or "Target" not in df.columns:
        return {}

    grouped = df.groupby("patientId")["Target"].max().reset_index()
    target_map = {
        str(row["patientId"]): int(row["Target"])
        for _, row in grouped.iterrows()
    }
    return target_map


# ---------------------------------------------------------
# Per-patient extraction
# ---------------------------------------------------------

def extract_features_for_case(
    patient_id: str,
    image_path: str,
    mask_path: str,
    target_map: Dict[str, int],
    histogram_bins: int = 16,
    glcm_levels: int = 16,
) -> Dict[str, float]:
    image = load_grayscale_image(image_path)
    mask = load_binary_mask(mask_path)

    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    values = get_masked_pixels(image, mask)

    feats = {
        "patient_id": patient_id,
        "image_path": image_path,
        "mask_path": mask_path,
        "target": int(target_map.get(patient_id, -1)),
    }

    feats.update(compute_geometry_features(mask))
    feats.update(compute_first_order_features(values))
    feats.update(compute_distribution_features(values))
    feats.update(compute_histogram_features(values, bins=histogram_bins))
    feats.update(compute_glcm_features(image, mask, levels=glcm_levels))

    return feats


# ---------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------

def collect_patient_ids(images_dir: str, masks_dir: str) -> List[str]:
    image_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    mask_ids = {
        os.path.splitext(f)[0]
        for f in os.listdir(masks_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    return sorted(list(image_ids.intersection(mask_ids)))


def process_dataset(
    images_dir: str,
    masks_dir: str,
    output_csv: str,
    labels_csv: Optional[str] = None,
    histogram_bins: int = 16,
    glcm_levels: int = 16,
    summary_json: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")

    patient_ids = collect_patient_ids(images_dir, masks_dir)
    target_map = load_target_map(labels_csv)

    records = []

    for patient_id in tqdm(patient_ids, desc="Extracting radiomics"):
        image_path = find_existing_file(images_dir, patient_id, [".png", ".jpg", ".jpeg"])
        mask_path = find_existing_file(masks_dir, patient_id, [".png", ".jpg", ".jpeg"])

        if image_path is None:
            print(f"[WARN] Missing image for {patient_id}, skipping.")
            continue

        if mask_path is None:
            print(f"[WARN] Missing mask for {patient_id}, skipping.")
            continue

        try:
            rec = extract_features_for_case(
                patient_id=patient_id,
                image_path=image_path,
                mask_path=mask_path,
                target_map=target_map,
                histogram_bins=histogram_bins,
                glcm_levels=glcm_levels,
            )
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Failed for {patient_id}: {e}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("patient_id").reset_index(drop=True)

    ensure_dir(os.path.dirname(output_csv))
    df.to_csv(output_csv, index=False)

    if summary_json is not None:
        summary = {
            "images_dir": images_dir,
            "masks_dir": masks_dir,
            "output_csv": output_csv,
            "num_cases": int(len(df)),
            "histogram_bins": int(histogram_bins),
            "glcm_levels": int(glcm_levels),
            "mean_roi_area": float(df["roi_area"].mean()) if len(df) > 0 else 0.0,
            "mean_int_mean": float(df["int_mean"].mean()) if len(df) > 0 else 0.0,
            "mean_glcm_contrast": float(df["glcm_contrast"].mean()) if len(df) > 0 else 0.0,
        }
        ensure_dir(os.path.dirname(summary_json))
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return df


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    if len(df) == 0:
        print("No radiomics features extracted.")
        return

    print("\n========== RADIOMICS SUMMARY ==========")
    print(f"Number of cases        : {len(df)}")
    if "target" in df.columns:
        known = df[df["target"] >= 0]
        if len(known) > 0:
            print(f"Positive cases         : {int((known['target'] == 1).sum())}")
            print(f"Negative cases         : {int((known['target'] == 0).sum())}")
    print(f"Mean ROI area          : {df['roi_area'].mean():.2f}")
    print(f"Mean intensity mean    : {df['int_mean'].mean():.4f}")
    print(f"Mean GLCM contrast     : {df['glcm_contrast'].mean():.4f}")
    print("=======================================\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract radiomics-style features from images and masks.")
    parser.add_argument("--paths_yaml", type=str, default="configs/paths.yaml")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to process.",
    )
    parser.add_argument("--images_dir", type=str, default=None, help="Directory containing input images")
    parser.add_argument("--masks_dir", type=str, default=None, help="Directory containing binary masks")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--labels_csv", type=str, default=None, help="Optional RSNA labels CSV")
    parser.add_argument("--histogram_bins", type=int, default=16, help="Number of histogram bins")
    parser.add_argument("--glcm_levels", type=int, default=16, help="GLCM quantization levels")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = get_project_root_from_this_file()
    cfg = load_yaml_config(args.paths_yaml)
    split = args.split.lower()

    images_dir = args.images_dir
    if images_dir is None:
        images_dir = deep_get(cfg, ["roi", f"{split}_roi_images_dir"], None)
        if images_dir is None:
            images_dir = deep_get(cfg, ["data", f"{split}_png_dir"], None)

    masks_dir = args.masks_dir
    if masks_dir is None:
        masks_dir = deep_get(cfg, ["roi", f"{split}_roi_masks_dir"], None)
        if masks_dir is None:
            masks_dir = deep_get(cfg, ["segmentation", f"{split}_masks_dir"], None)
        if masks_dir is None:
            masks_dir = deep_get(cfg, ["data", f"{split}_masks_dir"], None)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = deep_get(cfg, ["radiomics", f"{split}_output_csv"], None)
        if output_csv is None:
            output_csv = deep_get(cfg, ["radiomics", "output_csv"], None)

    summary_json = args.summary_json
    if summary_json is None:
        summary_json = deep_get(cfg, ["radiomics", f"{split}_summary_json"], None)

    labels_csv = args.labels_csv
    if labels_csv is None and split == "train":
        labels_csv = deep_get(cfg, ["data", "train_labels_csv"], None)

    images_dir = resolve_path(images_dir, project_root)
    masks_dir = resolve_path(masks_dir, project_root)
    output_csv = resolve_path(output_csv, project_root)
    summary_json = resolve_path(summary_json, project_root)
    labels_csv = resolve_path(labels_csv, project_root)

    if images_dir is None:
        raise ValueError("images_dir could not be resolved from CLI or YAML.")
    if masks_dir is None:
        raise ValueError("masks_dir could not be resolved from CLI or YAML.")
    if output_csv is None:
        raise ValueError("output_csv could not be resolved from CLI or YAML.")

    df = process_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_csv=output_csv,
        labels_csv=labels_csv,
        histogram_bins=args.histogram_bins,
        glcm_levels=args.glcm_levels,
        summary_json=summary_json,
    )

    print_summary(df)
    print(f"Saved radiomics features to: {output_csv}")
    if summary_json is not None:
        print(f"Saved radiomics summary to: {summary_json}")


if __name__ == "__main__":
    main()