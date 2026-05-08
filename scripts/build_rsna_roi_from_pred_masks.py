# build_rsna_roi_from_pred_masks.py

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"CSV boş: {path}")
    return df


def normalize_columns(df):
    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["patientid", "imageid", "id"]:
            rename_map[c] = "image_id"
        elif lc in ["image", "img_path", "path", "image_path"]:
            rename_map[c] = "image_path"
        elif lc in ["target", "class", "pneumonia", "label"]:
            rename_map[c] = "label"
        elif lc == "split":
            rename_map[c] = "split"
    df = df.rename(columns=rename_map)

    required = ["image_id", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}. Mevcut: {df.columns.tolist()}")

    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return img


def binarize_mask(mask, threshold=127):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, m = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return m


def keep_largest_components(mask, keep_n=2):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        areas.append((i, area))

    areas = sorted(areas, key=lambda x: x[1], reverse=True)[:keep_n]

    out = np.zeros_like(mask)
    for lab, _ in areas:
        out[labels == lab] = 255
    return out


def compute_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


def add_margin_to_bbox(x1, y1, x2, y2, h, w, margin_ratio=0.05):
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    return x1, y1, x2, y2


def process_one(image_path, mask_path, roi_path, masked_roi_path, keep_n=2, margin_ratio=0.05):
    image = load_gray(image_path)
    mask = load_gray(mask_path)
    mask = binarize_mask(mask)
    mask = keep_largest_components(mask, keep_n=keep_n)

    bbox = compute_bbox(mask)
    if bbox is None:
        return False, "empty_mask", None

    h, w = image.shape
    x1, y1, x2, y2 = add_margin_to_bbox(*bbox, h, w, margin_ratio=margin_ratio)

    roi = image[y1:y2+1, x1:x2+1].copy()

    masked = cv2.bitwise_and(image, image, mask=mask)
    masked_roi = masked[y1:y2+1, x1:x2+1].copy()

    os.makedirs(os.path.dirname(roi_path), exist_ok=True)
    os.makedirs(os.path.dirname(masked_roi_path), exist_ok=True)

    cv2.imwrite(roi_path, roi)
    cv2.imwrite(masked_roi_path, masked_roi)

    bbox_info = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1
    }
    return True, "ok", bbox_info


def process_split(csv_path, mask_dir, roi_dir, masked_roi_dir, output_csv,
                  keep_n=2, margin_ratio=0.05):
    df = normalize_columns(read_csv_safe(csv_path))

    rows = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path}"):
        image_id = str(row["image_id"])
        image_path = str(row["image_path"])
        mask_path = os.path.join(mask_dir, f"{image_id}.png")
        roi_path = os.path.join(roi_dir, f"{image_id}.png")
        masked_roi_path = os.path.join(masked_roi_dir, f"{image_id}.png")

        if not os.path.exists(image_path):
            errors.append({"image_id": image_id, "reason": "missing_image", "path": image_path})
            continue
        if not os.path.exists(mask_path):
            errors.append({"image_id": image_id, "reason": "missing_mask", "path": mask_path})
            continue

        ok, status, bbox_info = process_one(
            image_path=image_path,
            mask_path=mask_path,
            roi_path=roi_path,
            masked_roi_path=masked_roi_path,
            keep_n=keep_n,
            margin_ratio=margin_ratio
        )

        if not ok:
            errors.append({"image_id": image_id, "reason": status, "path": mask_path})
            continue

        out_row = row.to_dict()
        out_row["mask_path"] = mask_path
        out_row["roi_path"] = roi_path
        out_row["masked_roi_path"] = masked_roi_path
        out_row.update(bbox_info)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    err_csv = output_csv.replace(".csv", "_errors.csv")
    pd.DataFrame(errors).to_csv(err_csv, index=False)

    print("=" * 80)
    print(f"INPUT CSV   : {csv_path}")
    print(f"OUTPUT CSV  : {output_csv}")
    print(f"TOTAL       : {len(df)}")
    print(f"KEPT        : {len(out_df)}")
    print(f"DROPPED     : {len(errors)}")
    if len(out_df) > 0 and "label" in out_df.columns:
        print("Label dağılımı:")
        print(out_df["label"].value_counts(dropna=False))
    print(f"ERROR CSV   : {err_csv}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True,
                        help="infer_lung_masks_nnunet_rsna.py output_dir")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--keep_n_components", type=int, default=2)
    parser.add_argument("--margin_ratio", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # train
    process_split(
        csv_path=args.train_csv,
        mask_dir=os.path.join(args.pred_root, "train", "masks"),
        roi_dir=os.path.join(args.output_root, "train", "roi"),
        masked_roi_dir=os.path.join(args.output_root, "train", "masked_roi"),
        output_csv=os.path.join(args.output_root, "train_classifier_ready.csv"),
        keep_n=args.keep_n_components,
        margin_ratio=args.margin_ratio
    )

    # val
    process_split(
        csv_path=args.val_csv,
        mask_dir=os.path.join(args.pred_root, "val", "masks"),
        roi_dir=os.path.join(args.output_root, "val", "roi"),
        masked_roi_dir=os.path.join(args.output_root, "val", "masked_roi"),
        output_csv=os.path.join(args.output_root, "val_classifier_ready.csv"),
        keep_n=args.keep_n_components,
        margin_ratio=args.margin_ratio
    )

    # test
    process_split(
        csv_path=args.test_csv,
        mask_dir=os.path.join(args.pred_root, "test", "masks"),
        roi_dir=os.path.join(args.output_root, "test", "roi"),
        masked_roi_dir=os.path.join(args.output_root, "test", "masked_roi"),
        output_csv=os.path.join(args.output_root, "test_classifier_ready.csv"),
        keep_n=args.keep_n_components,
        margin_ratio=args.margin_ratio
    )


if __name__ == "__main__":
    main()