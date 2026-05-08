"""
run_qc0.py

QC-0 : Basic Image Quality Control

Updated
-------
- Reads default image directory from configs/paths.yaml
- Uses preprocess outputs by default
- Saves both CSV and JSON summary
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Run QC0 on preprocessed PNG images")
    parser.add_argument("--paths_yaml", type=str, default="configs/paths.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    paths_cfg = load_yaml_config(args.paths_yaml)

    if args.image_dir is None:
        if args.split == "train":
            args.image_dir = deep_get(paths_cfg, ["data", "train_png_dir"], "data/processed_pre/train/images_png")
        else:
            args.image_dir = deep_get(paths_cfg, ["data", "test_png_dir"], "data/processed_pre/test/images_png")

    qc_dir = deep_get(paths_cfg, ["qc", "output_dir"], "qc")
    if args.output_csv is None:
        args.output_csv = os.path.join(qc_dir, f"qc0_results_{args.split}.csv")
    if args.output_json is None:
        args.output_json = os.path.join(qc_dir, f"qc0_summary_{args.split}.json")

    return args


def compute_basic_stats(img: np.ndarray):
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))
    min_intensity = float(np.min(img))
    max_intensity = float(np.max(img))
    contrast = std_intensity
    return mean_intensity, std_intensity, min_intensity, max_intensity, contrast


def qc_rules(mean_intensity: float, contrast: float) -> str:
    flags: List[str] = []

    if mean_intensity < 40:
        flags.append("too_dark")
    if mean_intensity > 220:
        flags.append("too_bright")
    if contrast < 20:
        flags.append("low_contrast")

    if not flags:
        flags.append("ok")

    return "|".join(flags)


def run_qc(image_dir: str, output_csv: str, output_json: str):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted([
        x for x in os.listdir(image_dir)
        if os.path.splitext(x)[1].lower() in valid_exts
    ])

    results = []

    for img_name in tqdm(image_files, desc="QC0"):
        img_path = os.path.join(image_dir, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                results.append({"image_id": img_name, "status": "unreadable"})
                continue

            mean_intensity, std_intensity, min_intensity, max_intensity, contrast = compute_basic_stats(img)
            qc_flag = qc_rules(mean_intensity, contrast)

            results.append({
                "image_id": os.path.splitext(img_name)[0],
                "filename": img_name,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "min_intensity": min_intensity,
                "max_intensity": max_intensity,
                "contrast": contrast,
                "qc_flag": qc_flag,
                "status": "ok",
            })

        except Exception as e:
            results.append({
                "image_id": os.path.splitext(img_name)[0],
                "filename": img_name,
                "status": "error",
                "error_message": str(e),
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    summary = {
        "image_dir": image_dir,
        "num_images": int(len(df)),
        "num_ok": int((df.get("status", pd.Series(dtype=str)) == "ok").sum()) if "status" in df.columns else 0,
        "num_unreadable": int((df.get("status", pd.Series(dtype=str)) == "unreadable").sum()) if "status" in df.columns else 0,
        "num_error": int((df.get("status", pd.Series(dtype=str)) == "error").sum()) if "status" in df.columns else 0,
        "qc_flag_counts": df["qc_flag"].value_counts(dropna=False).to_dict() if "qc_flag" in df.columns else {},
        "output_csv": output_csv,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("QC0 completed")
    print("Results saved to:", output_csv)
    print("Summary saved to:", output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = parse_args()
    run_qc(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )