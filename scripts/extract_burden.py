"""
extract_burden.py
#d
Purpose
-------
Extract pneumonia burden-related quantitative features from
RSNA bounding boxes and lung masks.

Expected Inputs
---------------
1) RSNA labels CSV:
   columns typically include:
   patientId, x, y, width, height, Target

2) Lung masks directory:
   data/lung_masks/<patient_id>.png
   - binary mask image (0 background, >0 lung)

Outputs
-------
outputs/burden_scores.csv

Columns
-------
patient_id
target
num_boxes
total_bbox_area
lung_area
burden_ratio
left_lung_area
right_lung_area
left_bbox_area
right_bbox_area
left_burden_ratio
right_burden_ratio
bilateral_involvement
normalized_burden_score
"""

import os
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load binary lung mask as uint8 0/1.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    mask = (mask > 0).astype(np.uint8)
    return mask


def bbox_to_int(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    """
    Convert bbox coordinates to integer safe format:
    (x1, y1, x2, y2)
    """
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = max(x1, int(round(x + w)))
    y2 = max(y1, int(round(y + h)))
    return x1, y1, x2, y2


def clip_bbox_to_image(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Clip bbox to image boundaries.
    """
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def compute_bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


# ---------------------------------------------------------
# Mask / lung side analysis
# ---------------------------------------------------------

def split_left_right_lung(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into left/right halves in image coordinates.
    Note:
    image left half != anatomical left lung.
    For consistency we call them:
    - image_left
    - image_right
    But output columns use left/right in image-space.

    In chest X-ray display convention this is often acceptable for
    quantitative burden, as long as used consistently.
    """
    h, w = mask.shape
    mid = w // 2

    left_mask = np.zeros_like(mask, dtype=np.uint8)
    right_mask = np.zeros_like(mask, dtype=np.uint8)

    left_mask[:, :mid] = mask[:, :mid]
    right_mask[:, mid:] = mask[:, mid:]

    return left_mask, right_mask


def bbox_mask_intersection_area(
    bbox: Tuple[int, int, int, int],
    region_mask: np.ndarray
) -> int:
    """
    Compute intersection area between bbox rectangle and binary region mask.
    """
    x1, y1, x2, y2 = bbox
    h, w = region_mask.shape

    x1, y1, x2, y2 = clip_bbox_to_image(x1, y1, x2, y2, w, h)

    if x2 <= x1 or y2 <= y1:
        return 0

    patch = region_mask[y1:y2, x1:x2]
    return int(patch.sum())


def bbox_center_x(bbox: Tuple[int, int, int, int]) -> float:
    x1, _, x2, _ = bbox
    return (x1 + x2) / 2.0


# ---------------------------------------------------------
# Main burden extraction logic
# ---------------------------------------------------------

def load_rsna_labels(csv_path: str) -> pd.DataFrame:
    """
    Load RSNA labels CSV.
    Expected columns:
    patientId, x, y, width, height, Target
    """
    df = pd.read_csv(csv_path)

    required_cols = {"patientId", "Target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Some negative cases may have empty bbox fields.
    for col in ["x", "y", "width", "height"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def build_patient_groups(df: pd.DataFrame) -> List[str]:
    return sorted(df["patientId"].astype(str).unique().tolist())


def get_patient_rows(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    sub = df[df["patientId"].astype(str) == str(patient_id)].copy()
    return sub


def extract_patient_burden(
    patient_id: str,
    patient_df: pd.DataFrame,
    lung_mask: np.ndarray
) -> Dict:
    """
    Extract burden metrics for one patient.
    """
    h, w = lung_mask.shape

    total_lung_area = int(lung_mask.sum())

    left_lung_mask, right_lung_mask = split_left_right_lung(lung_mask)
    left_lung_area = int(left_lung_mask.sum())
    right_lung_area = int(right_lung_mask.sum())

    # Determine patient target
    # If any row has Target=1 -> positive
    target = int((patient_df["Target"].fillna(0).astype(int) == 1).any())

    bbox_areas = []
    left_bbox_area = 0
    right_bbox_area = 0
    left_has_lesion = False
    right_has_lesion = False

    positive_rows = patient_df[patient_df["Target"].fillna(0).astype(int) == 1].copy()

    for _, row in positive_rows.iterrows():
        x = row["x"]
        y = row["y"]
        bw = row["width"]
        bh = row["height"]

        if pd.isna(x) or pd.isna(y) or pd.isna(bw) or pd.isna(bh):
            continue

        bbox = bbox_to_int(x, y, bw, bh)
        bbox = clip_bbox_to_image(*bbox, width=w, height=h)

        area = compute_bbox_area(*bbox)
        bbox_areas.append(area)

        left_inter = bbox_mask_intersection_area(bbox, left_lung_mask)
        right_inter = bbox_mask_intersection_area(bbox, right_lung_mask)

        left_bbox_area += left_inter
        right_bbox_area += right_inter

        if left_inter > 0:
            left_has_lesion = True
        if right_inter > 0:
            right_has_lesion = True

        # Fallback if no mask intersection but bbox exists:
        # use bbox center to assign side
        if left_inter == 0 and right_inter == 0 and area > 0:
            cx = bbox_center_x(bbox)
            if cx < (w / 2):
                left_bbox_area += area
                left_has_lesion = True
            else:
                right_bbox_area += area
                right_has_lesion = True

    total_bbox_area = int(sum(bbox_areas))
    num_boxes = int(len(bbox_areas))

    burden_ratio = safe_div(total_bbox_area, total_lung_area)
    left_burden_ratio = safe_div(left_bbox_area, left_lung_area)
    right_burden_ratio = safe_div(right_bbox_area, right_lung_area)

    bilateral_involvement = int(left_has_lesion and right_has_lesion)

    # Simple normalized burden score in [0,1]-ish domain, clipped
    # combines total burden + bilateral effect + lesion count effect
    normalized_burden_score = (
        0.70 * min(1.0, burden_ratio) +
        0.20 * bilateral_involvement +
        0.10 * min(1.0, num_boxes / 4.0)
    )
    normalized_burden_score = float(np.clip(normalized_burden_score, 0.0, 1.0))

    return {
        "patient_id": patient_id,
        "target": target,
        "num_boxes": num_boxes,
        "total_bbox_area": total_bbox_area,
        "lung_area": total_lung_area,
        "burden_ratio": burden_ratio,
        "left_lung_area": left_lung_area,
        "right_lung_area": right_lung_area,
        "left_bbox_area": left_bbox_area,
        "right_bbox_area": right_bbox_area,
        "left_burden_ratio": left_burden_ratio,
        "right_burden_ratio": right_burden_ratio,
        "bilateral_involvement": bilateral_involvement,
        "normalized_burden_score": normalized_burden_score,
    }


# ---------------------------------------------------------
# Full dataset processing
# ---------------------------------------------------------

def process_dataset(
    labels_csv: str,
    masks_dir: str,
    output_csv: str,
    skip_missing_masks: bool = True
) -> pd.DataFrame:
    """
    Run burden extraction for all patients.
    """
    df = load_rsna_labels(labels_csv)
    patient_ids = build_patient_groups(df)

    records = []

    for patient_id in tqdm(patient_ids, desc="Extracting burden"):
        patient_df = get_patient_rows(df, patient_id)
        mask_path = os.path.join(masks_dir, f"{patient_id}.png")

        if not os.path.exists(mask_path):
            if skip_missing_masks:
                print(f"[WARN] Missing mask for {patient_id}, skipping.")
                continue
            raise FileNotFoundError(f"Missing mask for patient {patient_id}: {mask_path}")

        try:
            lung_mask = load_mask(mask_path)
            rec = extract_patient_burden(patient_id, patient_df, lung_mask)
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Failed for {patient_id}: {e}")
            continue

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values("patient_id").reset_index(drop=True)

    ensure_dir(os.path.dirname(output_csv))
    out_df.to_csv(output_csv, index=False)

    return out_df


# ---------------------------------------------------------
# Optional summary print
# ---------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    if len(df) == 0:
        print("No burden records generated.")
        return

    print("\n========== BURDEN SUMMARY ==========")
    print(f"Number of patients: {len(df)}")
    print(f"Positive patients  : {int(df['target'].sum())}")
    print(f"Mean burden ratio  : {df['burden_ratio'].mean():.6f}")
    print(f"Mean norm. burden  : {df['normalized_burden_score'].mean():.6f}")
    print(f"Bilateral cases    : {int(df['bilateral_involvement'].sum())}")
    print("====================================\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract pneumonia burden features from RSNA labels and lung masks.")
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to RSNA stage_2_train_labels.csv"
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        required=True,
        help="Directory containing lung masks as PNG files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/burden_scores.csv",
        help="Path to output burden CSV"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail on missing masks instead of skipping"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = process_dataset(
        labels_csv=args.labels_csv,
        masks_dir=args.masks_dir,
        output_csv=args.output_csv,
        skip_missing_masks=not args.strict
    )

    print_summary(df)
    print(f"Saved burden scores to: {args.output_csv}")


if __name__ == "__main__":
    main()