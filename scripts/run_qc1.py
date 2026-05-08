"""
run_qc1.py

QC-1 : Segmentation Quality Control

Updated
-------
- Reads default paths from configs/paths.yaml
- Supports train/test split
- Saves both CSV and JSON summary
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label

try:
    import yaml
except ImportError:
    yaml = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    yaml_file = Path(yaml_path)
    if not yaml_file.is_absolute():
        yaml_file = get_project_root() / yaml_file

    if not yaml_file.exists():
        return {}

    if yaml is None:
        raise ImportError("PyYAML is required to load paths.yaml")

    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(d: Dict[str, Any], keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_args():
    parser = argparse.ArgumentParser(description="Run QC1 on segmentation masks")
    parser.add_argument("--paths_yaml", type=str, default="configs/paths.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    paths_cfg = load_yaml_config(args.paths_yaml)

    if args.mask_dir is None:
        if args.split == "train":
            args.mask_dir = deep_get(paths_cfg, ["segmentation", "train_masks_dir"], "outputs/segmentation_masks/train")
        else:
            args.mask_dir = deep_get(paths_cfg, ["segmentation", "test_masks_dir"], "outputs/segmentation_masks/test")

    if args.image_dir is None:
        if args.split == "train":
            args.image_dir = deep_get(paths_cfg, ["data", "train_png_dir"], "data/processed_pre/train/images_png")
        else:
            args.image_dir = deep_get(paths_cfg, ["data", "test_png_dir"], "data/processed_pre/test/images_png")

    qc_dir = deep_get(paths_cfg, ["qc", "output_dir"], "qc")
    if args.output_csv is None:
        args.output_csv = os.path.join(qc_dir, f"qc1_results_{args.split}.csv")
    if args.output_json is None:
        args.output_json = os.path.join(qc_dir, f"qc1_summary_{args.split}.json")

    return args


def compute_mask_stats(mask: np.ndarray):
    h, w = mask.shape
    image_area = h * w
    lung_area = int(np.sum(mask > 0))
    coverage = lung_area / image_area if image_area > 0 else 0.0
    labeled = label(mask > 0)
    num_regions = int(labeled.max())
    return lung_area, coverage, num_regions


def qc_rules(coverage: float, num_regions: int) -> str:
    flags: List[str] = []

    if coverage < 0.05:
        flags.append("mask_too_small")
    if coverage > 0.60:
        flags.append("mask_too_large")
    if num_regions == 0:
        flags.append("empty_mask")
    if num_regions > 3:
        flags.append("too_many_regions")

    if not flags:
        flags.append("ok")

    return "|".join(flags)


def run_qc1(mask_dir: str, image_dir: str, output_csv: str, output_json: str):
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    mask_files = sorted([
        x for x in os.listdir(mask_dir)
        if os.path.splitext(x)[1].lower() in valid_exts
    ])

    results = []

    for mask_name in tqdm(mask_files, desc="QC1"):
        mask_path = os.path.join(mask_dir, mask_name)
        image_id = os.path.splitext(mask_name)[0]

        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                results.append({
                    "image_id": image_id,
                    "filename": mask_name,
                    "status": "unreadable_mask",
                })
                continue

            lung_area, coverage, num_regions = compute_mask_stats(mask)
            qc_flag = qc_rules(coverage, num_regions)

            image_exists = False
            if image_dir is not None:
                for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    if os.path.exists(os.path.join(image_dir, f"{image_id}{ext}")):
                        image_exists = True
                        break

            results.append({
                "image_id": image_id,
                "filename": mask_name,
                "lung_area": lung_area,
                "coverage": float(coverage),
                "num_regions": num_regions,
                "image_exists": int(image_exists),
                "qc_flag": qc_flag,
                "status": "ok",
            })

        except Exception as e:
            results.append({
                "image_id": image_id,
                "filename": mask_name,
                "status": "error",
                "error_message": str(e),
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    summary = {
        "mask_dir": mask_dir,
        "image_dir": image_dir,
        "num_masks": int(len(df)),
        "num_ok": int((df.get("status", pd.Series(dtype=str)) == "ok").sum()) if "status" in df.columns else 0,
        "num_unreadable": int((df.get("status", pd.Series(dtype=str)) == "unreadable_mask").sum()) if "status" in df.columns else 0,
        "num_error": int((df.get("status", pd.Series(dtype=str)) == "error").sum()) if "status" in df.columns else 0,
        "qc_flag_counts": df["qc_flag"].value_counts(dropna=False).to_dict() if "qc_flag" in df.columns else {},
        "output_csv": output_csv,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("QC1 completed")
    print("Results saved to:", output_csv)
    print("Summary saved to:", output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    run_qc1(
        mask_dir=args.mask_dir,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )