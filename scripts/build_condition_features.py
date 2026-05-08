import os
import cv2
import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

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
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
    return img[y1:y2 + 1, x1:x2 + 1]

def compute_edge_map(img: np.ndarray):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sobelx ** 2 + sobely ** 2)

def compute_intensity_stats(values: np.ndarray, prefix: str):
    out = {}
    if values.size == 0:
        for m in ["mean", "std", "min", "max", "q25", "q50", "q75"]: out[f"{prefix}_{m}"] = 0.0
        return out
    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_q25"] = float(np.percentile(values, 25))
    out[f"{prefix}_q50"] = float(np.percentile(values, 50))
    out[f"{prefix}_q75"] = float(np.percentile(values, 75))
    return out

def compute_entropy(values):
    if values.size == 0: return 0.0
    counts = np.histogram(values, bins=256, range=(0, 255))[0]
    return entropy(counts, base=2)

def compute_glcm_features(roi):
    roi_reduced = (roi // 16).astype(np.uint8)
    glcm = graycomatrix(roi_reduced, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 16, symmetric=True, normed=True)
    return {
        "glcm_contrast": float(np.mean(graycoprops(glcm, 'contrast'))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, 'homogeneity'))),
        "glcm_energy": float(np.mean(graycoprops(glcm, 'energy'))),
        "glcm_correlation": float(np.mean(graycoprops(glcm, 'correlation')))
    }

def build_single_record(image_path, mask_path, plain_dir, roi_dir, masked_roi_dir, mask_crop_dir, resize_to=224):
    image = read_gray(image_path)
    mask = postprocess_mask(read_gray(mask_path))
    
    # Geometrik Analiz
    orig_h, orig_w = image.shape[:2]
    lung_area = int(mask.sum())
    image_area = orig_h * orig_w
    mid_w_orig = orig_w // 2
    l_a, r_a = int(mask[:, :mid_w_orig].sum()), int(mask[:, mid_w_orig:].sum())
    x1, y1, x2, y2 = get_bbox(mask)
    bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # ROI İşlemleri
    roi = safe_crop(image, (x1, y1, x2, y2))
    roi_pixels = image[mask > 0].astype(np.float32)
    rh, rw = roi.shape
    mh, mw = rh // 2, rw // 2
    left_mean = np.mean(roi[:, :mw]) if mw > 0 else 0
    right_mean = np.mean(roi[:, mw:]) if mw > 0 else 0
    upper_mean = np.mean(roi[:mh, :]) if mh > 0 else 0
    lower_mean = np.mean(roi[mh:, :]) if mh > 0 else 0
    
    stats = compute_intensity_stats(roi_pixels, "roi")
    stats["roi_iqr"] = stats["roi_q75"] - stats["roi_q25"]
    edge_map = compute_edge_map(roi.astype(np.float32))

    record = {
        "image_path": image_path, "mask_path": mask_path,
        "mask_area_ratio": float(lung_area / image_area),
        "mask_left_right_asymmetry": float(abs(l_a - r_a) / (lung_area + 1e-8)),
        "mask_bbox_fill_ratio": float(lung_area / (bbox_area + 1e-8)),
        "mask_center_x": float(np.mean(np.where(mask > 0)[1]) / orig_w) if lung_area > 0 else 0.5,
        "mask_center_y": float(np.mean(np.where(mask > 0)[0]) / orig_h) if lung_area > 0 else 0.5,
        "roi_laplacian_var": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        "roi_entropy": compute_entropy(roi_pixels),
        "roi_gradient_mean": float(np.mean(edge_map)),
        "roi_left_right_mean_diff": float(abs(left_mean - right_mean)),
        "roi_upper_lower_mean_diff": float(abs(upper_mean - lower_mean)),
    }
    record.update(stats)
    if lung_area > 0: record.update(compute_glcm_features(roi))
    else: record.update({"glcm_contrast":0,"glcm_homogeneity":0,"glcm_energy":0,"glcm_correlation":0})

    # Kayıt İşlemleri (Plain Eklendi)
    base = os.path.splitext(os.path.basename(image_path))[0]
    record["plain_path"] = os.path.join(plain_dir, f"{base}.png")
    record["roi_path"] = os.path.join(roi_dir, f"{base}.png")
    record["masked_roi_path"] = os.path.join(masked_roi_dir, f"{base}.png")
    record["mask_crop_path"] = os.path.join(mask_crop_dir, f"{base}.png")

    cv2.imwrite(record["plain_path"], cv2.resize(image, (resize_to, resize_to)))
    cv2.imwrite(record["roi_path"], cv2.resize(roi, (resize_to, resize_to)))
    cv2.imwrite(record["masked_roi_path"], cv2.resize((image*mask).astype(np.uint8)[y1:y2+1, x1:x2+1], (resize_to, resize_to)))
    cv2.imwrite(record["mask_crop_path"], cv2.resize((mask*255)[y1:y2+1, x1:x2+1], (resize_to, resize_to), interpolation=cv2.INTER_NEAREST))
    
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="image_path")
    parser.add_argument("--mask_col", type=str, default="mask_path")
    parser.add_argument("--resize_to", type=int, default=224)
    args = parser.parse_args()

    splits = {"train": args.train_csv, "val": args.val_csv, "test": args.test_csv}

    for split_name, csv_path in splits.items():
        print(f"\n[INFO] {split_name.upper()} split işleniyor...")
        split_dir = os.path.join(args.output_root, split_name)
        # Plain klasörü buraya eklendi
        dirs = {d: os.path.join(split_dir, d) for d in ["plain", "roi", "masked_roi", "masks"]}
        for d in dirs.values(): ensure_dir(d)
        
        df = pd.read_csv(csv_path)
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
            res = build_single_record(row[args.image_col], row[args.mask_col], 
                                      dirs["plain"], dirs["roi"], dirs["masked_roi"], dirs["masks"], 
                                      args.resize_to)
            res.update({k: row[k] for k in df.columns if k not in res})
            records.append(res)
        
        out_path = os.path.join(split_dir, f"{split_name}_conditional_safe.csv")
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"✅ {split_name} kaydedildi: {out_path}")

if __name__ == "__main__":
    main()