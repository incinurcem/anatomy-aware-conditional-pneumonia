import os
import re
import cv2
import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    while stem.endswith(".png") or stem.endswith(".jpg") or stem.endswith(".jpeg"):
        stem = Path(stem).stem.lower()

    suffix_patterns = [
        r'[_\- ]?mask$',
        r'[_\- ]?masks$',
        r'[_\- ]?lung$',
        r'[_\- ]?lungs$',
        r'[_\- ]?seg$',
        r'[_\- ]?label$',
        r'[_\- ]?labels$',
    ]
    for pat in suffix_patterns:
        stem = re.sub(pat, '', stem)

    stem = stem.strip("_- ")
    return stem


def is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def collect_files(root: str):
    files = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isfile(p) and is_image_file(p):
            files.append(p)
    return files


def build_pairs(image_root: str, mask_root: str):
    image_files = collect_files(image_root)
    mask_files = collect_files(mask_root)

    image_map = {normalize_stem(os.path.basename(p)): p for p in image_files}
    mask_map = {normalize_stem(os.path.basename(p)): p for p in mask_files}

    common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
    pairs = []
    for key in common_keys:
        pairs.append({
            "case_id": key,
            "image_path": image_map[key],
            "mask_path": mask_map[key],
        })
    return pairs


def load_mask_as_binary(path: str):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask okunamadı: {path}")
    return (mask > 0).astype(np.uint8)


def dice_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    denom = y_true.sum() + y_pred.sum()
    if denom == 0:
        return 1.0
    return (2.0 * inter) / denom


def iou_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0
    return inter / union


def find_prediction_file(pred_dir, case_id):
    candidates = [
        os.path.join(pred_dir, f"{case_id}.png"),
        os.path.join(pred_dir, f"{case_id}.jpg"),
        os.path.join(pred_dir, f"{case_id}.jpeg"),
        os.path.join(pred_dir, f"{case_id}.tif"),
        os.path.join(pred_dir, f"{case_id}.tiff"),
        os.path.join(pred_dir, f"{case_id}.bmp"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    globbed = glob.glob(os.path.join(pred_dir, f"{case_id}*"))
    for g in globbed:
        if os.path.isfile(g) and is_image_file(g):
            return g
    return None


def evaluate_pairs(pairs, pred_dir):
    rows = []
    total = len(pairs)

    for idx, item in enumerate(pairs, start=1):
        case_id = item["case_id"]
        gt_path = item["mask_path"]
        pred_path = find_prediction_file(pred_dir, case_id)

        if pred_path is None:
            rows.append({
                "case_id": case_id,
                "gt_mask_path": gt_path,
                "pred_mask_path": None,
                "dice": np.nan,
                "iou": np.nan,
                "hd95": np.nan,
                "assd": np.nan,
                "status": "prediction_missing",
            })
            continue

        gt = load_mask_as_binary(gt_path)
        pred = load_mask_as_binary(pred_path)

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        rows.append({
            "case_id": case_id,
            "gt_mask_path": gt_path,
            "pred_mask_path": pred_path,
            "dice": float(dice_score(gt, pred)),
            "iou": float(iou_score(gt, pred)),
            "hd95": np.nan,
            "assd": np.nan,
            "status": "ok",
        })

        if idx % 25 == 0 or idx == total:
            print(f"[INFO] evaluated {idx}/{total}")

    return pd.DataFrame(rows)


def summarize_metrics(df):
    ok_df = df[df["status"] == "ok"].copy()
    return {
        "num_total": int(len(df)),
        "num_ok": int(len(ok_df)),
        "num_missing_prediction": int((df["status"] == "prediction_missing").sum()),
        "dice_mean": float(ok_df["dice"].mean()) if len(ok_df) else None,
        "dice_std": float(ok_df["dice"].std()) if len(ok_df) else None,
        "iou_mean": float(ok_df["iou"].mean()) if len(ok_df) else None,
        "iou_std": float(ok_df["iou"].std()) if len(ok_df) else None,
        "hd95_mean": None,
        "hd95_std": None,
        "assd_mean": None,
        "assd_std": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    pairs = build_pairs(args.image_root, args.mask_root)
    print(f"[INFO] matched pairs: {len(pairs)}")

    df = evaluate_pairs(pairs, args.pred_dir)
    csv_path = os.path.join(args.output_dir, "segmentation_metrics_per_case.csv")
    json_path = os.path.join(args.output_dir, "segmentation_metrics_summary.json")

    df.to_csv(csv_path, index=False)

    summary = summarize_metrics(df)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[INFO] saved:")
    print(csv_path)
    print(json_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()