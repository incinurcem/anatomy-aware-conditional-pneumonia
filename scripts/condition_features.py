import os
import cv2
import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image could not be read: {path}")
    return img


def binarize_mask(mask: np.ndarray, threshold: int = 127):
    return (mask > threshold).astype(np.uint8)


def largest_two_components(mask: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        component_areas.append((i, area))

    component_areas = sorted(component_areas, key=lambda x: x[1], reverse=True)
    keep_ids = [cid for cid, _ in component_areas[:2]]

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for cid in keep_ids:
        cleaned[labels == cid] = 1
    return cleaned


def postprocess_mask(mask: np.ndarray):
    mask = binarize_mask(mask)
    mask = largest_two_components(mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def get_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def safe_crop(img: np.ndarray, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)
    return img[y1:y2 + 1, x1:x2 + 1]


def compute_edge_map(img: np.ndarray):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return mag


def compute_intensity_stats(values: np.ndarray, prefix: str):
    out = {}
    if values.size == 0:
        out[f"{prefix}_mean"] = 0.0
        out[f"{prefix}_std"] = 0.0
        out[f"{prefix}_min"] = 0.0
        out[f"{prefix}_max"] = 0.0
        out[f"{prefix}_q25"] = 0.0
        out[f"{prefix}_q50"] = 0.0
        out[f"{prefix}_q75"] = 0.0
        return out

    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_q25"] = float(np.percentile(values, 25))
    out[f"{prefix}_q50"] = float(np.percentile(values, 50))
    out[f"{prefix}_q75"] = float(np.percentile(values, 75))
    return out


def build_single_record(
    image_path: str,
    mask_path: str,
    roi_dir: str,
    masked_roi_dir: str,
    mask_crop_dir: str,
    resize_to: int = 224
):
    image = read_gray(image_path)
    raw_mask = read_gray(mask_path)

    mask = postprocess_mask(raw_mask)

    orig_h, orig_w = image.shape[:2]
    image_area_pixels = orig_h * orig_w
    lung_area_pixels = int(mask.sum())
    lung_area_ratio = float(lung_area_pixels / (image_area_pixels + 1e-8))

    x1, y1, x2, y2 = get_bbox(mask)
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)
    bbox_area = int(bbox_w * bbox_h)
    bbox_area_ratio = float(bbox_area / (image_area_pixels + 1e-8))

    fg_ys, fg_xs = np.where(mask > 0)
    if len(fg_xs) > 0:
        center_x = float(np.mean(fg_xs))
        center_y = float(np.mean(fg_ys))
    else:
        center_x = 0.0
        center_y = 0.0

    roi = safe_crop(image, (x1, y1, x2, y2))
    mask_crop = safe_crop(mask * 255, (x1, y1, x2, y2))
    masked_image = (image * mask).astype(np.uint8)

    roi_resized = cv2.resize(roi, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    masked_roi_crop = safe_crop(masked_image, (x1, y1, x2, y2))
    masked_roi_resized = cv2.resize(masked_roi_crop, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    mask_crop_resized = cv2.resize(mask_crop, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    roi_path = os.path.join(roi_dir, f"{base_name}.png")
    masked_roi_path = os.path.join(masked_roi_dir, f"{base_name}.png")
    mask_crop_path = os.path.join(mask_crop_dir, f"{base_name}.png")

    cv2.imwrite(roi_path, roi_resized)
    cv2.imwrite(masked_roi_path, masked_roi_resized)
    cv2.imwrite(mask_crop_path, mask_crop_resized)

    img_all = image.astype(np.float32).reshape(-1)
    fg_vals = image[mask > 0].astype(np.float32)
    edge_map = compute_edge_map(image.astype(np.float32))
    edge_all = edge_map.reshape(-1)
    edge_fg = edge_map[mask > 0]

    record = {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "mask_path": mask_path,
        "roi_path": roi_path,
        "masked_roi_path": masked_roi_path,
        "mask_crop_path": mask_crop_path,
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "lung_area_pixels": int(lung_area_pixels),
        "image_area_pixels": int(image_area_pixels),
        "lung_area_ratio": float(lung_area_ratio),
        "lung_bbox_x1": int(x1),
        "lung_bbox_y1": int(y1),
        "lung_bbox_x2": int(x2),
        "lung_bbox_y2": int(y2),
        "lung_bbox_w": int(bbox_w),
        "lung_bbox_h": int(bbox_h),
        "lung_bbox_area": int(bbox_area),
        "lung_bbox_area_ratio": float(bbox_area_ratio),
        "mask_center_x": float(center_x),
        "mask_center_y": float(center_y),
    }

    record.update(compute_intensity_stats(img_all, "img"))
    record.update(compute_intensity_stats(fg_vals, "img_fg"))
    record["edge_mean_all"] = float(np.mean(edge_all)) if edge_all.size > 0 else 0.0
    record["edge_mean_fg"] = float(np.mean(edge_fg)) if edge_fg.size > 0 else 0.0
    record["edge_std_all"] = float(np.std(edge_all)) if edge_all.size > 0 else 0.0
    record["edge_std_fg"] = float(np.std(edge_fg)) if edge_fg.size > 0 else 0.0

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--roi_dir", type=str, required=True)
    parser.add_argument("--masked_roi_dir", type=str, required=True)
    parser.add_argument("--mask_crop_dir", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="image_path")
    parser.add_argument("--mask_col", type=str, default="mask_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--resize_to", type=int, default=224)
    args = parser.parse_args()

    ensure_dir(args.roi_dir)
    ensure_dir(args.masked_roi_dir)
    ensure_dir(args.mask_crop_dir)
    ensure_dir(os.path.dirname(args.output_csv))

    df = pd.read_csv(args.input_csv)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ROI + condition features"):
        image_path = row[args.image_col]
        mask_path = row[args.mask_col]

        rec = build_single_record(
            image_path=image_path,
            mask_path=mask_path,
            roi_dir=args.roi_dir,
            masked_roi_dir=args.masked_roi_dir,
            mask_crop_dir=args.mask_crop_dir,
            resize_to=args.resize_to
        )

        if args.label_col in df.columns:
            rec[args.label_col] = int(row[args.label_col])

        if "patientId" in df.columns:
            rec["patientId"] = row["patientId"]

        if "split" in df.columns:
            rec["split"] = row["split"]

        records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Toplam örnek: {len(out_df)}")
    print(f"CSV kaydedildi: {args.output_csv}")
    print(out_df.head())


if __name__ == "__main__":
    main()