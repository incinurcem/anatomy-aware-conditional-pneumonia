import os
import json
import cv2
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 127).astype(np.uint8)
    return m


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return inter / union


def area_error(pred, gt):
    return float(abs(pred.astype(bool).sum() - gt.astype(bool).sum()))


def get_surface(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    surface = np.logical_xor(mask, eroded)
    return surface


def hd95(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)
    if len(all_dists) == 0:
        return 0.0
    return float(np.percentile(all_dists, 95))


def match_predictions_and_gt(pred_dir, gt_dir):
    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}
    pairs = []

    for pred_path in pred_files:
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        if stem in gt_map:
            pairs.append((pred_path, gt_map[stem], stem))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    args = parser.parse_args()

    pairs = match_predictions_and_gt(args.pred_dir, args.gt_dir)
    if len(pairs) == 0:
        raise RuntimeError("Prediction-GT eşleşmesi bulunamadı.")

    rows = []
    for pred_path, gt_path, case_id in tqdm(pairs, desc="Evaluating"):
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        row = {
            "case_id": case_id,
            "dice": dice_score(pred, gt),
            "iou": iou_score(pred, gt),
            "hd95": hd95(pred, gt),
            "area_error": area_error(pred, gt),
            "pred_area": int(pred.sum()),
            "gt_area": int(gt.sum())
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "n_cases": int(len(df)),
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std(ddof=1)) if len(df) > 1 else 0.0,
        "iou_mean": float(df["iou"].mean()),
        "iou_std": float(df["iou"].std(ddof=1)) if len(df) > 1 else 0.0,
        "hd95_mean": float(df["hd95"].dropna().mean()) if df["hd95"].notna().any() else None,
        "hd95_std": float(df["hd95"].dropna().std(ddof=1)) if df["hd95"].notna().sum() > 1 else 0.0,
        "area_error_mean": float(df["area_error"].mean()),
        "area_error_std": float(df["area_error"].std(ddof=1)) if len(df) > 1 else 0.0
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()