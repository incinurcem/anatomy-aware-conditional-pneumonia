"""
generate_roi.py

Purpose
-------
Generate anatomy-constrained ROI crops from chest X-ray images
using predicted or ground-truth lung masks.

Pipeline
--------
image + lung_mask -> union bbox -> optional padding -> crop -> save

Inputs
------
1) images_dir
   preprocessed images, usually:
   data/processed_pre/<split>/images_png/<patient_id>.png

2) masks_dir
   segmentation masks, usually:
   <masks_dir>/<patient_id>.png
   Binary mask image (0 background, >0 lung)

Outputs
-------
1) cropped images:
   <roi_images_dir>/<patient_id>.png

2) cropped masks:
   <roi_masks_dir>/<patient_id>.png

3) roi metadata:
   <output_csv>

CSV Columns
-----------
patient_id
orig_h
orig_w
x1
y1
x2
y2
roi_w
roi_h
padding_ratio
min_padding_px
mask_area
bbox_area
extent
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
    # .../<project_root>/scripts/generate_roi.py
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


def load_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_binary_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def find_existing_file(base_dir: str, patient_id: str, extensions: List[str]) -> Optional[str]:
    for ext in extensions:
        path = os.path.join(base_dir, f"{patient_id}{ext}")
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------
# ROI bbox extraction
# ---------------------------------------------------------

def get_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return bbox from binary mask as:
    (x1, y1, x2, y2)
    where x2,y2 are exclusive bounds.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def apply_padding_to_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    padding_ratio: float = 0.05,
    min_padding_px: int = 8,
) -> Tuple[int, int, int, int]:
    """
    Expand bbox by padding_ratio around width/height.
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape

    bw = x2 - x1
    bh = y2 - y1

    pad_x = max(min_padding_px, int(round(bw * padding_ratio)))
    pad_y = max(min_padding_px, int(round(bh * padding_ratio)))

    x1n = max(0, x1 - pad_x)
    y1n = max(0, y1 - pad_y)
    x2n = min(w, x2 + pad_x)
    y2n = min(h, y2 + pad_y)

    return x1n, y1n, x2n, y2n


def crop_array(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return arr[y1:y2, x1:x2]


# ---------------------------------------------------------
# Optional mask cleanup
# ---------------------------------------------------------

def clean_mask(mask: np.ndarray, min_component_area: int = 256) -> np.ndarray:
    """
    Remove tiny connected components from mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros_like(mask_u8)

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == label_idx] = 1

    return cleaned


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in binary mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape

    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, flood_inv)

    return (filled > 0).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, min_component_area: int = 256, fill_holes: bool = True) -> np.ndarray:
    out = clean_mask(mask, min_component_area=min_component_area)
    if fill_holes:
        out = fill_mask_holes(out)
    return (out > 0).astype(np.uint8)


# ---------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------

def process_single_case(
    patient_id: str,
    image_path: str,
    mask_path: str,
    roi_images_dir: str,
    roi_masks_dir: Optional[str],
    padding_ratio: float,
    min_padding_px: int,
    save_mask: bool,
    postprocess: bool,
    min_component_area: int,
    overwrite: bool,
) -> Optional[Dict[str, Any]]:
    image = load_grayscale_image(image_path)
    mask = load_binary_mask(mask_path)

    if image.shape != mask.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0).astype(np.uint8)

    if postprocess:
        mask = postprocess_mask(
            mask,
            min_component_area=min_component_area,
            fill_holes=True,
        )

    bbox = get_mask_bbox(mask)
    if bbox is None:
        print(f"[WARN] Empty mask for {patient_id}, skipping.")
        return None

    bbox = apply_padding_to_bbox(
        bbox=bbox,
        image_shape=image.shape,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
    )

    roi_img = crop_array(image, bbox)
    roi_mask = crop_array(mask, bbox)

    if roi_img.size == 0 or roi_mask.size == 0:
        print(f"[WARN] Invalid crop for {patient_id}, skipping.")
        return None

    ensure_dir(roi_images_dir)
    out_img_path = os.path.join(roi_images_dir, f"{patient_id}.png")
    if overwrite or not os.path.exists(out_img_path):
        success = cv2.imwrite(out_img_path, roi_img)
        if not success:
            raise RuntimeError(f"Failed to save ROI image: {out_img_path}")

    out_mask_path = ""
    if save_mask and roi_masks_dir is not None:
        ensure_dir(roi_masks_dir)
        out_mask_path = os.path.join(roi_masks_dir, f"{patient_id}.png")
        if overwrite or not os.path.exists(out_mask_path):
            success = cv2.imwrite(out_mask_path, (roi_mask * 255).astype(np.uint8))
            if not success:
                raise RuntimeError(f"Failed to save ROI mask: {out_mask_path}")

    x1, y1, x2, y2 = bbox
    roi_h, roi_w = roi_img.shape[:2]

    mask_area = int(mask.sum())
    bbox_area = int((x2 - x1) * (y2 - y1))
    extent = float(mask_area / bbox_area) if bbox_area > 0 else 0.0

    record = {
        "patient_id": patient_id,
        "image_path": image_path,
        "mask_path": mask_path,
        "roi_image_path": out_img_path,
        "roi_mask_path": out_mask_path,
        "orig_h": int(image.shape[0]),
        "orig_w": int(image.shape[1]),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "roi_w": int(roi_w),
        "roi_h": int(roi_h),
        "padding_ratio": float(padding_ratio),
        "min_padding_px": int(min_padding_px),
        "mask_area": int(mask_area),
        "bbox_area": int(bbox_area),
        "extent": float(extent),
    }

    return record


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
    roi_images_dir: str,
    roi_masks_dir: Optional[str],
    output_csv: str,
    padding_ratio: float = 0.05,
    min_padding_px: int = 8,
    save_mask: bool = True,
    postprocess: bool = True,
    min_component_area: int = 256,
    overwrite: bool = False,
    summary_json: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")

    patient_ids = collect_patient_ids(images_dir, masks_dir)
    records = []

    for patient_id in tqdm(patient_ids, desc="Generating ROI"):
        image_path = find_existing_file(images_dir, patient_id, [".png", ".jpg", ".jpeg"])
        mask_path = find_existing_file(masks_dir, patient_id, [".png", ".jpg", ".jpeg"])

        if image_path is None:
            print(f"[WARN] Missing image for {patient_id}, skipping.")
            continue

        if mask_path is None:
            print(f"[WARN] Missing mask for {patient_id}, skipping.")
            continue

        try:
            rec = process_single_case(
                patient_id=patient_id,
                image_path=image_path,
                mask_path=mask_path,
                roi_images_dir=roi_images_dir,
                roi_masks_dir=roi_masks_dir,
                padding_ratio=padding_ratio,
                min_padding_px=min_padding_px,
                save_mask=save_mask,
                postprocess=postprocess,
                min_component_area=min_component_area,
                overwrite=overwrite,
            )
            if rec is not None:
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
            "roi_images_dir": roi_images_dir,
            "roi_masks_dir": roi_masks_dir,
            "output_csv": output_csv,
            "num_cases": int(len(df)),
            "mean_roi_w": float(df["roi_w"].mean()) if len(df) > 0 else 0.0,
            "mean_roi_h": float(df["roi_h"].mean()) if len(df) > 0 else 0.0,
            "mean_bbox_area": float(df["bbox_area"].mean()) if len(df) > 0 else 0.0,
            "mean_extent": float(df["extent"].mean()) if len(df) > 0 else 0.0,
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
        print("No ROI records generated.")
        return

    print("\n============= ROI SUMMARY =============")
    print(f"Number of ROI crops : {len(df)}")
    print(f"Mean ROI width      : {df['roi_w'].mean():.2f}")
    print(f"Mean ROI height     : {df['roi_h'].mean():.2f}")
    print(f"Mean bbox area      : {df['bbox_area'].mean():.2f}")
    print(f"Mean extent         : {df['extent'].mean():.4f}")
    print("=======================================\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ROI crops from images using lung masks.")
    parser.add_argument("--paths_yaml", type=str, default="configs/paths.yaml")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to process.",
    )

    parser.add_argument("--images_dir", type=str, default=None, help="Directory containing original images")
    parser.add_argument("--masks_dir", type=str, default=None, help="Directory containing lung masks")
    parser.add_argument("--roi_images_dir", type=str, default=None, help="Directory to save cropped ROI images")
    parser.add_argument("--roi_masks_dir", type=str, default=None, help="Directory to save cropped ROI masks")
    parser.add_argument("--output_csv", type=str, default=None, help="CSV path for ROI metadata")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional summary JSON path")

    parser.add_argument("--padding_ratio", type=float, default=0.05, help="Relative padding around mask bbox")
    parser.add_argument("--min_padding_px", type=int, default=8, help="Minimum padding in pixels")
    parser.add_argument("--no_save_mask", action="store_true", help="Do not save cropped ROI masks")
    parser.add_argument("--no_postprocess", action="store_true", help="Disable mask postprocessing")
    parser.add_argument("--min_component_area", type=int, default=256, help="Minimum connected component area")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ROI files")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = get_project_root_from_this_file()
    cfg = load_yaml_config(args.paths_yaml)

    split = args.split.lower()

    images_dir = args.images_dir
    if images_dir is None:
        images_dir = deep_get(cfg, ["data", f"{split}_png_dir"], None)

    masks_dir = args.masks_dir
    if masks_dir is None:
        masks_dir = deep_get(cfg, ["segmentation", f"{split}_masks_dir"], None)
        if masks_dir is None:
            masks_dir = deep_get(cfg, ["data", f"{split}_masks_dir"], None)

    roi_images_dir = args.roi_images_dir
    if roi_images_dir is None:
        roi_images_dir = deep_get(cfg, ["roi", f"{split}_roi_images_dir"], None)
        if roi_images_dir is None:
            base_roi_dir = deep_get(cfg, ["roi", "roi_dir"], None)
            if base_roi_dir is not None:
                roi_images_dir = os.path.join(base_roi_dir, split, "images")

    roi_masks_dir = args.roi_masks_dir
    if roi_masks_dir is None:
        roi_masks_dir = deep_get(cfg, ["roi", f"{split}_roi_masks_dir"], None)
        if roi_masks_dir is None and roi_images_dir is not None:
            roi_masks_dir = os.path.join(os.path.dirname(roi_images_dir), "masks")

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = deep_get(cfg, ["roi", f"{split}_roi_metadata_csv"], None)
        if output_csv is None:
            output_csv = deep_get(cfg, ["roi", "output_csv"], None)
        if output_csv is None and roi_images_dir is not None:
            output_csv = os.path.join(os.path.dirname(roi_images_dir), f"{split}_roi_metadata.csv")

    summary_json = args.summary_json
    if summary_json is None:
        summary_json = deep_get(cfg, ["roi", f"{split}_roi_summary_json"], None)

    images_dir = resolve_path(images_dir, project_root)
    masks_dir = resolve_path(masks_dir, project_root)
    roi_images_dir = resolve_path(roi_images_dir, project_root)
    roi_masks_dir = resolve_path(roi_masks_dir, project_root)
    output_csv = resolve_path(output_csv, project_root)
    summary_json = resolve_path(summary_json, project_root)

    if images_dir is None:
        raise ValueError("images_dir could not be resolved from CLI or YAML.")
    if masks_dir is None:
        raise ValueError("masks_dir could not be resolved from CLI or YAML.")
    if roi_images_dir is None:
        raise ValueError("roi_images_dir could not be resolved from CLI or YAML.")
    if output_csv is None:
        raise ValueError("output_csv could not be resolved from CLI or YAML.")

    df = process_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        roi_images_dir=roi_images_dir,
        roi_masks_dir=roi_masks_dir,
        output_csv=output_csv,
        padding_ratio=args.padding_ratio,
        min_padding_px=args.min_padding_px,
        save_mask=not args.no_save_mask,
        postprocess=not args.no_postprocess,
        min_component_area=args.min_component_area,
        overwrite=args.overwrite,
        summary_json=summary_json,
    )

    print_summary(df)
    print(f"Saved ROI metadata to: {output_csv}")
    if summary_json is not None:
        print(f"Saved ROI summary to: {summary_json}")


if __name__ == "__main__":
    main()