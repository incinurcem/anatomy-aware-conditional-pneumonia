import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

from common_eval_utils import save_json


def load_mask(path: str, threshold: int = 0) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.sum(y_true * y_pred)
    denom = np.sum(y_true) + np.sum(y_pred)
    if denom == 0:
        return 1.0
    return float((2.0 * intersection) / (denom + 1e-12))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.sum(y_true * y_pred)
    union = np.sum((y_true + y_pred) > 0)
    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-12))


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if (tp + fn) == 0:
        return 1.0
    return float(tp / (tp + fn + 1e-12))


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (tn + fp) == 0:
        return 1.0
    return float(tn / (tn + fp + 1e-12))


def precision_score_seg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (tp + fp) == 0:
        return 1.0
    return float(tp / (tp + fp + 1e-12))


def rvd_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    gt_vol = np.sum(y_true)
    pred_vol = np.sum(y_pred)
    if gt_vol == 0 and pred_vol == 0:
        return 0.0
    if gt_vol == 0:
        return float("inf")
    return float((pred_vol - gt_vol) / (gt_vol + 1e-12))


def volume_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    gt_vol = np.sum(y_true)
    pred_vol = np.sum(y_pred)
    denom = gt_vol + pred_vol
    if denom == 0:
        return 1.0
    return float(1.0 - abs(pred_vol - gt_vol) / (denom + 1e-12))


def get_surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = binary_erosion(mask.astype(bool))
    return mask.astype(bool) ^ eroded


def surface_distances(mask_gt: np.ndarray, mask_pred: np.ndarray, spacing=(1.0, 1.0)) -> np.ndarray:
    surf_gt = get_surface(mask_gt)
    surf_pred = get_surface(mask_pred)

    if surf_gt.sum() == 0 and surf_pred.sum() == 0:
        return np.array([0.0], dtype=np.float32)
    if surf_gt.sum() == 0 or surf_pred.sum() == 0:
        h, w = mask_gt.shape
        diag = float(np.sqrt((h * spacing[0]) ** 2 + (w * spacing[1]) ** 2))
        return np.array([diag], dtype=np.float32)

    dt_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    dt_pred = distance_transform_edt(~surf_pred, sampling=spacing)
    dist_pred_to_gt = dt_gt[surf_pred]
    dist_gt_to_pred = dt_pred[surf_gt]
    return np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)


def hd95(mask_gt: np.ndarray, mask_pred: np.ndarray, spacing=(1.0, 1.0)) -> float:
    dists = surface_distances(mask_gt, mask_pred, spacing=spacing)
    return float(np.percentile(dists, 95))


def assd(mask_gt: np.ndarray, mask_pred: np.ndarray, spacing=(1.0, 1.0)) -> float:
    dists = surface_distances(mask_gt, mask_pred, spacing=spacing)
    return float(np.mean(dists))


def boundary_f1(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    surf_gt = get_surface(mask_gt).astype(np.uint8)
    surf_pred = get_surface(mask_pred).astype(np.uint8)
    return dice_score(surf_gt, surf_pred)


def match_prediction_and_gt(pred_dir: str, gt_dir: str):
    pred_paths = sorted(glob(os.path.join(pred_dir, "*")))
    pairs = []
    for pred_path in pred_paths:
        name = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, name)
        if os.path.exists(gt_path):
            pairs.append((name, pred_path, gt_path))
    return pairs


def bootstrap_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = series.dropna().values.astype(np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        means.append(float(vals[idx].mean()))
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pixel_spacing_x", type=float, default=1.0)
    parser.add_argument("--pixel_spacing_y", type=float, default=1.0)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    spacing = (args.pixel_spacing_y, args.pixel_spacing_x)

    pairs = match_prediction_and_gt(args.pred_dir, args.gt_dir)
    if not pairs:
        raise ValueError("No matching prediction and ground-truth files found.")

    all_rows = []
    for name, pred_path, gt_path in pairs:
        pred = load_mask(pred_path, threshold=args.threshold)
        gt = load_mask(gt_path, threshold=args.threshold)
        row = {
            "image_name": name,
            "dice": dice_score(gt, pred),
            "iou": iou_score(gt, pred),
            "sensitivity": sensitivity_score(gt, pred),
            "specificity": specificity_score(gt, pred),
            "precision": precision_score_seg(gt, pred),
            "hd95": hd95(gt, pred, spacing=spacing),
            "assd": assd(gt, pred, spacing=spacing),
            "boundary_f1": boundary_f1(gt, pred),
            "rvd": rvd_score(gt, pred),
            "volume_similarity": volume_similarity(gt, pred),
            "gt_positive_pixels": int(gt.sum()),
            "pred_positive_pixels": int(pred.sum()),
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.output_dir, "segmentation_casewise_metrics.csv"), index=False)

    summary = {"n_cases": int(len(df))}
    for metric in ["dice", "iou", "sensitivity", "specificity", "precision", "hd95", "assd", "boundary_f1", "rvd", "volume_similarity"]:
        summary[f"{metric}_mean"] = float(df[metric].mean())
        summary[f"{metric}_std"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
        mean_b, lo, hi = bootstrap_ci(df[metric], n_boot=args.n_boot, seed=42)
        summary[f"{metric}_ci_mean"] = mean_b
        summary[f"{metric}_ci_lower_95"] = lo
        summary[f"{metric}_ci_upper_95"] = hi

    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "segmentation_summary.csv"), index=False)
    save_json(os.path.join(args.output_dir, "segmentation_summary.json"), summary)

    print("===== SEGMENTATION EVALUATION COMPLETE =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
