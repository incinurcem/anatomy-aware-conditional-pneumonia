import os
import json
import argparse
import traceback
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_mkdir_for_file(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)


def load_grayscale_image(path: str) -> np.ndarray:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Geçersiz görüntü yolu: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")

    return img


def safe_quantize_uint8(img: np.ndarray, levels: int = 32) -> np.ndarray:
    img = np.asarray(img, dtype=np.uint8)
    if levels >= 256:
        return img.copy()

    step = max(1, 256 // levels)
    q = img // step
    q[q >= levels] = levels - 1
    return q.astype(np.uint8)


def get_best_image_path(row: pd.Series) -> str:
    """
    Kesin karar:
    1) roi_path
    2) masked_roi_path
    3) image_path

    Amaç:
    - Öncelik ROI tabanlı medikal feature
    - ROI yoksa pipeline kırılmasın
    """
    for col in ["roi_path", "masked_roi_path", "image_path"]:
        if col in row and pd.notna(row[col]) and isinstance(row[col], str) and row[col].strip():
            return row[col]
    raise ValueError("row içinde kullanılabilir görüntü yolu yok (roi_path/masked_roi_path/image_path)")


def get_mask_path(row: pd.Series) -> Optional[str]:
    if "mask_path" in row and pd.notna(row["mask_path"]) and isinstance(row["mask_path"], str) and row["mask_path"].strip():
        return row["mask_path"]
    return None


def sanitize_feature_value(x: float) -> float:
    if x is None:
        return 0.0
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return x


# =========================================================
# Feature extraction
# =========================================================
def compute_basic_intensity_features(img: np.ndarray) -> Dict[str, float]:
    x = img.astype(np.float32).reshape(-1)

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    feats = {
        "roi_mean": np.mean(x),
        "roi_std": np.std(x),
        "roi_median": np.median(x),
        "roi_q1": q1,
        "roi_q3": q3,
        "roi_iqr": q3 - q1,
        "roi_min": np.min(x),
        "roi_max": np.max(x),
    }
    return {k: sanitize_feature_value(v) for k, v in feats.items()}


def compute_entropy_features(img: np.ndarray) -> Dict[str, float]:
    return {
        "roi_entropy": sanitize_feature_value(shannon_entropy(img))
    }


def compute_edge_features(img: np.ndarray) -> Dict[str, float]:
    img_f = img.astype(np.float32)

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    edge_thresh = np.percentile(grad_mag, 75)
    edge_density = np.mean(grad_mag > edge_thresh)

    lap = cv2.Laplacian(img_f, cv2.CV_32F)

    feats = {
        "roi_edge_density_sobel": edge_density,
        "roi_laplacian_var": np.var(lap),
        "roi_gradient_mean": np.mean(grad_mag),
        "roi_gradient_std": np.std(grad_mag),
    }
    return {k: sanitize_feature_value(v) for k, v in feats.items()}


def compute_foreground_features(img: np.ndarray) -> Dict[str, float]:
    # Otsu ile kaba foreground oranı
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = np.mean(th > 0)

    return {
        "roi_foreground_ratio": sanitize_feature_value(fg_ratio)
    }


def compute_asymmetry_features(img: np.ndarray) -> Dict[str, float]:
    h, w = img.shape
    left = img[:, :w // 2]
    right = img[:, w // 2:]
    upper = img[:h // 2, :]
    lower = img[h // 2:, :]

    left_mean = np.mean(left) if left.size else 0.0
    right_mean = np.mean(right) if right.size else 0.0
    upper_mean = np.mean(upper) if upper.size else 0.0
    lower_mean = np.mean(lower) if lower.size else 0.0

    feats = {
        "roi_left_right_mean_diff": abs(left_mean - right_mean),
        "roi_upper_lower_mean_diff": abs(upper_mean - lower_mean),
    }
    return {k: sanitize_feature_value(v) for k, v in feats.items()}


def compute_glcm_features(img: np.ndarray, levels: int = 32) -> Dict[str, float]:
    q = safe_quantize_uint8(img, levels=levels)

    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True
    )

    feats = {
        "glcm_contrast": np.mean(graycoprops(glcm, "contrast")),
        "glcm_homogeneity": np.mean(graycoprops(glcm, "homogeneity")),
        "glcm_energy": np.mean(graycoprops(glcm, "energy")),
        "glcm_correlation": np.mean(graycoprops(glcm, "correlation")),
    }
    return {k: sanitize_feature_value(v) for k, v in feats.items()}


def compute_mask_features(mask_img: Optional[np.ndarray]) -> Dict[str, float]:
    # Prediction mask varsa kullan
    if mask_img is None:
        return {
            "mask_area_ratio": 0.0,
            "mask_left_area_ratio": 0.0,
            "mask_right_area_ratio": 0.0,
            "mask_left_right_asymmetry": 0.0,
            "mask_center_x": 0.5,
            "mask_center_y": 0.5,
            "mask_bbox_fill_ratio": 0.0,
        }

    mask = (mask_img > 0).astype(np.uint8)
    h, w = mask.shape
    total_pixels = max(1, h * w)

    area = int(mask.sum())
    area_ratio = area / total_pixels

    left = mask[:, :w // 2]
    right = mask[:, w // 2:]
    left_area = int(left.sum())
    right_area = int(right.sum())

    left_ratio = left_area / total_pixels
    right_ratio = right_area / total_pixels
    asym = abs(left_area - right_area) / max(1, area)

    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cx = xs.mean() / max(1, (w - 1))
        cy = ys.mean() / max(1, (h - 1))

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bbox_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        bbox_fill = area / bbox_area
    else:
        cx, cy, bbox_fill = 0.5, 0.5, 0.0

    feats = {
        "mask_area_ratio": area_ratio,
        "mask_left_area_ratio": left_ratio,
        "mask_right_area_ratio": right_ratio,
        "mask_left_right_asymmetry": asym,
        "mask_center_x": cx,
        "mask_center_y": cy,
        "mask_bbox_fill_ratio": bbox_fill,
    }
    return {k: sanitize_feature_value(v) for k, v in feats.items()}


def extract_safe_features(image_path: str, mask_path: Optional[str] = None) -> Dict[str, float]:
    img = load_grayscale_image(image_path)

    feats = {}
    feats.update(compute_basic_intensity_features(img))
    feats.update(compute_entropy_features(img))
    feats.update(compute_edge_features(img))
    feats.update(compute_foreground_features(img))
    feats.update(compute_asymmetry_features(img))
    feats.update(compute_glcm_features(img, levels=32))

    mask_img = None
    if mask_path is not None and os.path.exists(mask_path):
        mask_img = load_grayscale_image(mask_path)

    feats.update(compute_mask_features(mask_img))
    return feats


# =========================================================
# Split processing
# =========================================================
LEAKAGE_COLS = [
    "bbox_count",
    "bbox_total_area_ratio",
    "bbox_max_area_ratio",
    "has_left_lesion",
    "has_right_lesion",
    "has_bilateral_lesion",
    "lesion_center_x_mean",
    "lesion_center_y_mean",
]


def extract_split_raw_features(
    split_name: str,
    input_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = pd.read_csv(input_csv)

    if len(df) == 0:
        raise ValueError(f"{split_name} CSV boş: {input_csv}")

    out_rows = []
    err_rows = []
    feature_order = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[{split_name}] raw feature extraction"):
        try:
            image_id = str(row["image_id"]) if "image_id" in df.columns else str(idx)
            image_path = get_best_image_path(row)
            mask_path = get_mask_path(row)

            feats = extract_safe_features(image_path=image_path, mask_path=mask_path)

            if feature_order is None:
                feature_order = list(feats.keys())

            clean_row = row.to_dict()

            for col in LEAKAGE_COLS:
                if col in clean_row:
                    clean_row.pop(col)

            clean_row["selected_image_path"] = image_path
            clean_row["selected_mask_path"] = mask_path if mask_path is not None else ""

            for k, v in feats.items():
                clean_row[k] = v

            out_rows.append(clean_row)

        except Exception as e:
            err_rows.append({
                "split": split_name,
                "row_index": int(idx),
                "image_id": row["image_id"] if "image_id" in row else "",
                "error": str(e),
                "traceback": traceback.format_exc(limit=1),
            })

    out_df = pd.DataFrame(out_rows)
    err_df = pd.DataFrame(err_rows)

    if feature_order is None:
        raise RuntimeError(f"{split_name} için hiç feature çıkarılamadı.")

    return out_df, err_df, feature_order


def compute_train_scaler(train_df: pd.DataFrame, feature_order: List[str]) -> Dict[str, Dict[str, float]]:
    X = train_df[feature_order].astype(np.float32).values

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0

    scaler = {
        "feature_order": feature_order,
        "mean": {k: float(v) for k, v in zip(feature_order, mean)},
        "std": {k: float(v) for k, v in zip(feature_order, std)},
        "condition_dim": len(feature_order),
    }
    return scaler


def normalize_feature_vector(feats_row: pd.Series, scaler: Dict[str, Dict[str, float]]) -> np.ndarray:
    feature_order = scaler["feature_order"]
    mean_dict = scaler["mean"]
    std_dict = scaler["std"]

    vals = []
    for f in feature_order:
        x = sanitize_feature_value(feats_row[f])
        mu = float(mean_dict[f])
        sd = float(std_dict[f])
        vals.append((x - mu) / sd)

    return np.asarray(vals, dtype=np.float32)


def save_split_outputs(
    split_name: str,
    df_features: pd.DataFrame,
    err_df: pd.DataFrame,
    scaler: Dict[str, Dict[str, float]],
    output_root: str,
):
    split_root = os.path.join(output_root, split_name)
    vectors_dir = os.path.join(split_root, "vectors")
    ensure_dir(split_root)
    ensure_dir(vectors_dir)

    output_rows = []

    for idx, row in tqdm(df_features.iterrows(), total=len(df_features), desc=f"[{split_name}] saving normalized vectors"):
        image_id = str(row["image_id"]) if "image_id" in df_features.columns else str(idx)

        vec = normalize_feature_vector(row, scaler)
        vec_path = os.path.join(vectors_dir, f"{image_id}.npy")
        np.save(vec_path, vec)

        out_row = row.to_dict()
        out_row["condition_vector_path"] = vec_path
        output_rows.append(out_row)

    out_df = pd.DataFrame(output_rows)

    out_csv = os.path.join(split_root, f"{split_name}_conditional_safe.csv")
    err_csv = os.path.join(split_root, f"{split_name}_conditional_safe_errors.csv")
    meta_json = os.path.join(split_root, f"{split_name}_conditional_safe_meta.json")

    out_df.to_csv(out_csv, index=False)
    err_df.to_csv(err_csv, index=False)

    meta = {
        "split": split_name,
        "num_total_saved": int(len(out_df)),
        "num_errors": int(len(err_df)),
        "condition_dim": int(scaler["condition_dim"]),
        "feature_order": scaler["feature_order"],
        "notes": [
            "Leakage-free feature set",
            "No GT bbox derived features used",
            "ROI/image features + optional prediction mask anatomy features used",
            "All vectors normalized using TRAIN split statistics only"
        ]
    }

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_csv, err_csv, meta_json, vectors_dir, out_df


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Build all safe condition vectors for train/val/test in one run.")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_root)

    # ---------- Step 1: raw feature extraction ----------
    train_df, train_err, train_feature_order = extract_split_raw_features("train", args.train_csv)
    val_df, val_err, val_feature_order = extract_split_raw_features("val", args.val_csv)
    test_df, test_err, test_feature_order = extract_split_raw_features("test", args.test_csv)

    if train_feature_order != val_feature_order or train_feature_order != test_feature_order:
        raise RuntimeError("Feature order splitler arasında farklı. Bu olmamalı.")

    feature_order = train_feature_order

    # ---------- Step 2: fit scaler only on train ----------
    scaler = compute_train_scaler(train_df, feature_order)

    scaler_json = os.path.join(args.output_root, "train_feature_scaler.json")
    feature_order_json = os.path.join(args.output_root, "feature_order.json")
    summary_json = os.path.join(args.output_root, "build_summary.json")

    with open(scaler_json, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2, ensure_ascii=False)

    with open(feature_order_json, "w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2, ensure_ascii=False)

    # ---------- Step 3: save normalized vectors + csvs ----------
    train_out_csv, train_err_csv, train_meta_json, train_vec_dir, train_out_df = save_split_outputs(
        "train", train_df, train_err, scaler, args.output_root
    )
    val_out_csv, val_err_csv, val_meta_json, val_vec_dir, val_out_df = save_split_outputs(
        "val", val_df, val_err, scaler, args.output_root
    )
    test_out_csv, test_err_csv, test_meta_json, test_vec_dir, test_out_df = save_split_outputs(
        "test", test_df, test_err, scaler, args.output_root
    )

    summary = {
        "train_csv_input": args.train_csv,
        "val_csv_input": args.val_csv,
        "test_csv_input": args.test_csv,
        "output_root": args.output_root,
        "condition_dim": int(scaler["condition_dim"]),
        "feature_order": feature_order,
        "outputs": {
            "train_csv": train_out_csv,
            "val_csv": val_out_csv,
            "test_csv": test_out_csv,
            "train_error_csv": train_err_csv,
            "val_error_csv": val_err_csv,
            "test_error_csv": test_err_csv,
            "train_meta_json": train_meta_json,
            "val_meta_json": val_meta_json,
            "test_meta_json": test_meta_json,
            "train_vectors_dir": train_vec_dir,
            "val_vectors_dir": val_vec_dir,
            "test_vectors_dir": test_vec_dir,
            "scaler_json": scaler_json,
            "feature_order_json": feature_order_json,
        },
        "counts": {
            "train_kept": int(len(train_out_df)),
            "val_kept": int(len(val_out_df)),
            "test_kept": int(len(test_out_df)),
            "train_errors": int(len(train_err)),
            "val_errors": int(len(val_err)),
            "test_errors": int(len(test_err)),
        },
        "leakage_columns_removed_if_present": LEAKAGE_COLS,
        "decision": "ROI-first safe condition vector pipeline with optional prediction-mask anatomy features",
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 120)
    print("[INFO] ALL SAFE CONDITION VECTORS FINISHED")
    print("=" * 120)
    print(f"[INFO] OUTPUT ROOT   : {args.output_root}")
    print(f"[INFO] CONDITION DIM : {scaler['condition_dim']}")
    print(f"[INFO] FEATURES      : {feature_order}")
    print("-" * 120)
    print(f"[INFO] TRAIN CSV     : {train_out_csv}")
    print(f"[INFO] VAL CSV       : {val_out_csv}")
    print(f"[INFO] TEST CSV      : {test_out_csv}")
    print("-" * 120)
    print(f"[INFO] TRAIN ERRORS  : {len(train_err)}")
    print(f"[INFO] VAL ERRORS    : {len(val_err)}")
    print(f"[INFO] TEST ERRORS   : {len(test_err)}")
    print("-" * 120)
    print(f"[INFO] SCALER JSON   : {scaler_json}")
    print(f"[INFO] SUMMARY JSON  : {summary_json}")
    print("=" * 120)


if __name__ == "__main__":
    main()