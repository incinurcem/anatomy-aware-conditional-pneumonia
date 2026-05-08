import os
import shutil
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_unlink(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[WARN] Silinemedi: {path} -> {e}")


def list_pngs(folder):
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def normalize_mask_filename(name: str) -> str:
    if name.lower().endswith(".png.png"):
        return name[:-4]
    return name


def find_id_col(df):
    for c in ["image_id", "patientId", "patient_id", "id", "ImageId"]:
        if c in df.columns:
            return c
    raise ValueError(f"ID column not found. Columns: {list(df.columns)}")


def find_label_col(df):
    for c in ["label", "Target", "target", "class", "pneumonia"]:
        if c in df.columns:
            return c
    raise ValueError(f"Label column not found. Columns: {list(df.columns)}")


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def resolve_existing_temp_dirs(temp_root, split_name):
    split_root = os.path.join(temp_root, split_name)

    input_candidates = [
        os.path.join(split_root, "inputs"),
        os.path.join(split_root, "input"),
    ]
    output_candidates = [
        os.path.join(split_root, "preds"),
        os.path.join(split_root, "pred"),
        os.path.join(split_root, "output"),
    ]

    input_dir = None
    output_dir = None

    for p in input_candidates:
        if os.path.exists(p):
            input_dir = p
            break

    for p in output_candidates:
        if os.path.exists(p):
            output_dir = p
            break

    return input_dir, output_dir


def quick_count_report(name, folder):
    count = len(list_pngs(folder)) if os.path.exists(folder) else "DIR YOK"
    print(f"{name}: {folder}")
    print(f"exists: {os.path.exists(folder)}")
    print(f"png count: {count}")
    print("-" * 100)


def copy_file(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def sync_dir_pngs(src_dir, dst_dir, clear_dst=False):
    ensure_dir(dst_dir)

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    if clear_dst:
        for f in os.listdir(dst_dir):
            if f.lower().endswith(".png"):
                safe_unlink(os.path.join(dst_dir, f))

    src_files = list_pngs(src_dir)
    copied = 0

    for f in tqdm(src_files, desc=f"Copy PNGs {os.path.basename(src_dir)} -> {os.path.basename(dst_dir)}"):
        copy_file(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        copied += 1

    print("=" * 100)
    print("DIR SYNC DONE")
    print(f"SRC    : {src_dir}")
    print(f"DST    : {dst_dir}")
    print(f"FILES  : {len(src_files)}")
    print(f"COPIED : {copied}")
    return copied


def sync_file_to_drive(src_file, dst_file):
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    ensure_dir(os.path.dirname(dst_file))
    shutil.copy2(src_file, dst_file)
    print(f"[FILE COPIED] {src_file} -> {dst_file}")


def sync_temp_predictions_to_work_masks(
    split_name,
    temp_root,
    work_mask_dir,
    clear_work_dir=False,
):
    ensure_dir(work_mask_dir)

    temp_input_dir, temp_output_dir = resolve_existing_temp_dirs(temp_root, split_name)

    print("=" * 100)
    print(f"[{split_name.upper()}] TEMP DURUMU")
    print(f"Temp input dir  : {temp_input_dir}")
    print(f"Temp output dir : {temp_output_dir}")
    print(f"Work mask dir   : {work_mask_dir}")

    if temp_output_dir is None or not os.path.exists(temp_output_dir):
        raise FileNotFoundError(
            f"[{split_name}] Prediction klasörü bulunamadı. Beklenen: preds / pred / output"
        )

    pred_files = list_pngs(temp_output_dir)
    print(f"Existing preds  : {len(pred_files)}")

    if len(pred_files) == 0:
        raise RuntimeError(f"[{split_name}] Temp prediction klasörü boş: {temp_output_dir}")

    if clear_work_dir:
        for f in os.listdir(work_mask_dir):
            if f.lower().endswith(".png"):
                safe_unlink(os.path.join(work_mask_dir, f))

    copied = 0
    for f in tqdm(pred_files, desc=f"Temp preds -> work masks [{split_name}]"):
        src = os.path.join(temp_output_dir, f)
        dst = os.path.join(work_mask_dir, normalize_mask_filename(f))
        shutil.copy2(src, dst)
        copied += 1

    print("=" * 100)
    print("SYNC TEMP PREDICTIONS TO WORK MASKS")
    print(f"Split         : {split_name}")
    print(f"Pred dir      : {temp_output_dir}")
    print(f"Work mask dir : {work_mask_dir}")
    print(f"Pred count    : {len(pred_files)}")
    print(f"Copied        : {copied}")

    return copied


def binarize_mask(mask):
    """
    nnUNet maskeleri bazen:
      - 0 / 1
      - 0 / 255
      - float probability benzeri
      - uint16 / int16
    gelebiliyor.

    Bu fonksiyon tüm bu durumları güvenli biçimde 0/255 binary maskeye çevirir.
    """
    if mask is None:
        return None

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[..., 0]

    # NaN / inf temizle
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

    # Tamamen boşsa direkt dön
    if mask.size == 0:
        return None

    max_val = float(mask.max())
    min_val = float(mask.min())

    # Tamamen sıfırsa boş maske
    if max_val <= 0:
        return np.zeros(mask.shape, dtype=np.uint8)

    # 0/1 veya 0/1'e yakın çıktı
    if max_val <= 1.0:
        binary = (mask > 0).astype(np.uint8) * 255
        return binary

    # 0/255 tipi klasik maskeler
    # burada threshold 127 yerine "0'dan büyük her şey foreground" alıyoruz
    binary = (mask > 0).astype(np.uint8) * 255
    return binary

def keep_largest_components(mask, max_components=2):
    if mask is None:
        return None

    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return (mask * 255).astype(np.uint8)

    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comps.append((i, area))

    comps.sort(key=lambda x: x[1], reverse=True)
    keep_ids = [cid for cid, _ in comps[:max_components]]

    out = np.zeros_like(mask, dtype=np.uint8)
    for cid in keep_ids:
        out[labels == cid] = 255

    return out


def get_bbox_from_mask(mask, pad_ratio=0.05):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h, w = mask.shape[:2]
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return x1, y1, x2, y2


def crop_with_bbox(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2 + 1, x1:x2 + 1]


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def generate_rois_for_split(
    split_name,
    split_csv,
    image_dir,
    mask_dir,
    roi_dir,
    masked_roi_dir,
    metadata_csv,
    error_csv,
):
    ensure_dir(roi_dir)
    ensure_dir(masked_roi_dir)

    df = pd.read_csv(split_csv)
    id_col = find_id_col(df)

    rows = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating ROI [{split_name}]"):
        image_id = str(row[id_col])
        png_name = f"{image_id}.png"

        image_path = os.path.join(image_dir, png_name)
        mask_path = os.path.join(mask_dir, png_name)
        roi_path = os.path.join(roi_dir, png_name)
        masked_roi_path = os.path.join(masked_roi_dir, png_name)

        if not os.path.exists(image_path):
            errors.append({
                "image_id": image_id,
                "reason": "missing_image",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue

        if not os.path.exists(mask_path):
            errors.append({
                "image_id": image_id,
                "reason": "missing_mask",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue
        img = read_gray(image_path)
        mask = read_gray(mask_path)

        if img is None:
            errors.append({
                "image_id": image_id,
                "reason": "image_read_failed",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue

        if mask is None:
            errors.append({
                "image_id": image_id,
                "reason": "mask_read_failed",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue

        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        raw_mask_unique = np.unique(mask)
        raw_mask_max = float(mask.max()) if mask.size else 0.0

        mask = binarize_mask(mask)

        if mask is None:
            errors.append({
                "image_id": image_id,
                "reason": "mask_binarize_failed",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue

        fg_pixels_before = int((mask > 0).sum())

        mask = keep_largest_components(mask, max_components=2)

        fg_pixels_after = int((mask > 0).sum())

        if fg_pixels_after == 0:
            errors.append({
                "image_id": image_id,
                "reason": "empty_mask_after_binarize",
                "image_path": image_path,
                "mask_path": mask_path,
                "raw_mask_max": raw_mask_max,
                "raw_mask_unique_sample": raw_mask_unique[:10].tolist(),
                "fg_pixels_before": fg_pixels_before,
                "fg_pixels_after": fg_pixels_after,
            })
            continue

        bbox = get_bbox_from_mask(mask, pad_ratio=0.05)
        if bbox is None:
            errors.append({
                "image_id": image_id,
                "reason": "empty_mask_bbox",
                "image_path": image_path,
                "mask_path": mask_path,
                "raw_mask_max": raw_mask_max,
                "raw_mask_unique_sample": raw_mask_unique[:10].tolist(),
                "fg_pixels_before": fg_pixels_before,
                "fg_pixels_after": fg_pixels_after,
            })
            continue

        roi = crop_with_bbox(img, bbox)
        masked_full = apply_mask(img, mask)
        masked_roi = crop_with_bbox(masked_full, bbox)

        if roi.size == 0 or masked_roi.size == 0:
            errors.append({
                "image_id": image_id,
                "reason": "empty_crop",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue

        ok1 = cv2.imwrite(roi_path, roi)
        ok2 = cv2.imwrite(masked_roi_path, masked_roi)

        if not ok1 or not ok2:
            errors.append({
                "image_id": image_id,
                "reason": "write_failed",
                "image_path": image_path,
                "mask_path": mask_path,
            })
            continue

        rows.append({
            "image_id": image_id,
            "split": split_name,
            "image_path": image_path,
            "mask_path": mask_path,
            "roi_path": roi_path,
            "masked_roi_path": masked_roi_path,
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "roi_h": int(roi.shape[0]),
            "roi_w": int(roi.shape[1]),
            "fg_pixels_before": fg_pixels_before,
            "fg_pixels_after": fg_pixels_after,
   
        })

    out_df = pd.DataFrame(rows)
    err_df = pd.DataFrame(errors)

    out_df.to_csv(metadata_csv, index=False)
    err_df.to_csv(error_csv, index=False)

    print("=" * 100)
    print(f"[{split_name}] ROI generation bitti")
    print(f"Success : {len(out_df)}")
    print(f"Failed  : {len(err_df)}")
    print(f"ROI dir : {roi_dir}")
    print(f"Masked  : {masked_roi_dir}")

    if len(err_df) > 0:
        print(err_df["reason"].value_counts().head(10))

    return out_df, err_df


def build_classifier_csv(
    split_name,
    split_csv,
    image_dir,
    mask_dir,
    roi_dir,
    masked_roi_dir,
    out_csv,
):
    df = pd.read_csv(split_csv)
    id_col = find_id_col(df)
    label_col = find_label_col(df)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building classifier CSV [{split_name}]"):
        image_id = str(row[id_col])
        label = int(row[label_col])

        rows.append({
            "image_id": image_id,
            "split": split_name,
            "label": label,
            "image_path": os.path.join(image_dir, f"{image_id}.png"),
            "mask_path": os.path.join(mask_dir, f"{image_id}.png"),
            "roi_path": os.path.join(roi_dir, f"{image_id}.png"),
            "masked_roi_path": os.path.join(masked_roi_dir, f"{image_id}.png"),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    print("=" * 100)
    print(f"[{split_name}] classifier.csv yazıldı")
    print(f"Rows    : {len(out_df)}")
    print(f"Out CSV : {out_csv}")

    return out_df


def build_classifier_ready_csv(
    split_name,
    input_csv,
    output_csv,
    error_csv,
):
    df = pd.read_csv(input_csv)

    kept = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating ready CSV [{split_name}]"):
        missing = []

        for col in ["image_path", "mask_path", "roi_path", "masked_roi_path"]:
            p = row[col]
            if pd.isna(p) or (not os.path.exists(p)):
                missing.append(col)

        if len(missing) == 0:
            kept.append(row.to_dict())
        else:
            errors.append({
                "image_id": row["image_id"],
                "reason": "missing_files:" + ",".join(missing),
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "roi_path": row["roi_path"],
                "masked_roi_path": row["masked_roi_path"],
            })

    kept_df = pd.DataFrame(kept)
    err_df = pd.DataFrame(errors)

    kept_df.to_csv(output_csv, index=False)
    err_df.to_csv(error_csv, index=False)

    print("=" * 100)
    print(f"[{split_name}] classifier_ready.csv yazıldı")
    print(f"TOTAL   : {len(df)}")
    print(f"KEPT    : {len(kept_df)}")
    print(f"DROPPED : {len(err_df)}")
    print(f"OUT     : {output_csv}")
    print(f"ERROR   : {error_csv}")

    if len(err_df) > 0:
        print(err_df["reason"].value_counts().head(10))

    return kept_df, err_df


def sync_work_outputs_to_drive(args):
    print("\n" + "#" * 120)
    print("STEP 4/4 - COPY WORK OUTPUTS FROM /content TO DRIVE")
    print("#" * 120)

    # masks
    sync_dir_pngs(args.work_train_masks_dir, args.train_masks_dir, clear_dst=args.clear_train_masks_on_drive)
    sync_dir_pngs(args.work_val_masks_dir, args.val_masks_dir, clear_dst=args.clear_val_masks_on_drive)
    sync_dir_pngs(args.work_test_masks_dir, args.test_masks_dir, clear_dst=args.clear_test_masks_on_drive)

    # roi
    sync_dir_pngs(args.work_train_roi_dir, args.train_roi_dir, clear_dst=args.clear_train_roi_on_drive)
    sync_dir_pngs(args.work_val_roi_dir, args.val_roi_dir, clear_dst=args.clear_val_roi_on_drive)
    sync_dir_pngs(args.work_test_roi_dir, args.test_roi_dir, clear_dst=args.clear_test_roi_on_drive)

    # masked roi
    sync_dir_pngs(args.work_train_masked_roi_dir, args.train_masked_roi_dir, clear_dst=args.clear_train_masked_roi_on_drive)
    sync_dir_pngs(args.work_val_masked_roi_dir, args.val_masked_roi_dir, clear_dst=args.clear_val_masked_roi_on_drive)
    sync_dir_pngs(args.work_test_masked_roi_dir, args.test_masked_roi_dir, clear_dst=args.clear_test_masked_roi_on_drive)

    # csv files
    sync_file_to_drive(args.work_train_classifier_csv, args.train_classifier_csv)
    sync_file_to_drive(args.work_val_classifier_csv, args.val_classifier_csv)
    sync_file_to_drive(args.work_test_classifier_csv, args.test_classifier_csv)

    sync_file_to_drive(args.work_train_classifier_ready_csv, args.train_classifier_ready_csv)
    sync_file_to_drive(args.work_val_classifier_ready_csv, args.val_classifier_ready_csv)
    sync_file_to_drive(args.work_test_classifier_ready_csv, args.test_classifier_ready_csv)

    sync_file_to_drive(args.work_train_classifier_ready_errors_csv, args.train_classifier_ready_errors_csv)
    sync_file_to_drive(args.work_val_classifier_ready_errors_csv, args.val_classifier_ready_errors_csv)
    sync_file_to_drive(args.work_test_classifier_ready_errors_csv, args.test_classifier_ready_errors_csv)

    sync_file_to_drive(
        os.path.join(args.work_roi_meta_dir, "train_roi_metadata.csv"),
        os.path.join(args.roi_meta_dir, "train_roi_metadata.csv")
    )
    sync_file_to_drive(
        os.path.join(args.work_roi_meta_dir, "val_roi_metadata.csv"),
        os.path.join(args.roi_meta_dir, "val_roi_metadata.csv")
    )
    sync_file_to_drive(
        os.path.join(args.work_roi_meta_dir, "test_roi_metadata.csv"),
        os.path.join(args.roi_meta_dir, "test_roi_metadata.csv")
    )

    sync_file_to_drive(
        os.path.join(args.work_roi_meta_dir, "train_roi_errors.csv"),
        os.path.join(args.roi_meta_dir, "train_roi_errors.csv")
    )
    sync_file_to_drive(
        os.path.join(args.work_roi_meta_dir, "val_roi_errors.csv"),
        os.path.join(args.roi_meta_dir, "val_roi_errors.csv")
    )
    sync_file_to_drive(
        os.path.join(args.work_roi_meta_dir, "test_roi_errors.csv"),
        os.path.join(args.roi_meta_dir, "test_roi_errors.csv")
    )


def main():
    parser = argparse.ArgumentParser()

    # split csv
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    # images
    parser.add_argument("--images_dir", type=str, required=True)

    # drive final outputs
    parser.add_argument("--train_masks_dir", type=str, required=True)
    parser.add_argument("--val_masks_dir", type=str, required=True)
    parser.add_argument("--test_masks_dir", type=str, required=True)

    parser.add_argument("--train_roi_dir", type=str, required=True)
    parser.add_argument("--val_roi_dir", type=str, required=True)
    parser.add_argument("--test_roi_dir", type=str, required=True)

    parser.add_argument("--train_masked_roi_dir", type=str, required=True)
    parser.add_argument("--val_masked_roi_dir", type=str, required=True)
    parser.add_argument("--test_masked_roi_dir", type=str, required=True)

    parser.add_argument("--train_classifier_csv", type=str, required=True)
    parser.add_argument("--val_classifier_csv", type=str, required=True)
    parser.add_argument("--test_classifier_csv", type=str, required=True)

    parser.add_argument("--train_classifier_ready_csv", type=str, required=True)
    parser.add_argument("--val_classifier_ready_csv", type=str, required=True)
    parser.add_argument("--test_classifier_ready_csv", type=str, required=True)

    parser.add_argument("--train_classifier_ready_errors_csv", type=str, required=True)
    parser.add_argument("--val_classifier_ready_errors_csv", type=str, required=True)
    parser.add_argument("--test_classifier_ready_errors_csv", type=str, required=True)

    parser.add_argument("--roi_meta_dir", type=str, required=True)
    parser.add_argument("--temp_infer_root", type=str, required=True)

    # work root in /content
    parser.add_argument("--work_root", type=str, required=True)

    # optional clean flags for /content work dirs
    parser.add_argument("--clear_work_masks", action="store_true")

    # optional clean flags for drive destination dirs
    parser.add_argument("--clear_train_masks_on_drive", action="store_true")
    parser.add_argument("--clear_val_masks_on_drive", action="store_true")
    parser.add_argument("--clear_test_masks_on_drive", action="store_true")

    parser.add_argument("--clear_train_roi_on_drive", action="store_true")
    parser.add_argument("--clear_val_roi_on_drive", action="store_true")
    parser.add_argument("--clear_test_roi_on_drive", action="store_true")

    parser.add_argument("--clear_train_masked_roi_on_drive", action="store_true")
    parser.add_argument("--clear_val_masked_roi_on_drive", action="store_true")
    parser.add_argument("--clear_test_masked_roi_on_drive", action="store_true")

    args = parser.parse_args()

    # /content work paths
    args.work_train_masks_dir = os.path.join(args.work_root, "train", "masks")
    args.work_val_masks_dir = os.path.join(args.work_root, "val", "masks")
    args.work_test_masks_dir = os.path.join(args.work_root, "test", "masks")

    args.work_train_roi_dir = os.path.join(args.work_root, "train", "roi")
    args.work_val_roi_dir = os.path.join(args.work_root, "val", "roi")
    args.work_test_roi_dir = os.path.join(args.work_root, "test", "roi")

    args.work_train_masked_roi_dir = os.path.join(args.work_root, "train", "masked_roi")
    args.work_val_masked_roi_dir = os.path.join(args.work_root, "val", "masked_roi")
    args.work_test_masked_roi_dir = os.path.join(args.work_root, "test", "masked_roi")

    args.work_split_csv_dir = os.path.join(args.work_root, "splits")
    args.work_roi_meta_dir = os.path.join(args.work_root, "metadata")

    args.work_train_classifier_csv = os.path.join(args.work_split_csv_dir, "train_classifier.csv")
    args.work_val_classifier_csv = os.path.join(args.work_split_csv_dir, "val_classifier.csv")
    args.work_test_classifier_csv = os.path.join(args.work_split_csv_dir, "test_classifier.csv")

    args.work_train_classifier_ready_csv = os.path.join(args.work_split_csv_dir, "train_classifier_ready.csv")
    args.work_val_classifier_ready_csv = os.path.join(args.work_split_csv_dir, "val_classifier_ready.csv")
    args.work_test_classifier_ready_csv = os.path.join(args.work_split_csv_dir, "test_classifier_ready.csv")

    args.work_train_classifier_ready_errors_csv = os.path.join(args.work_split_csv_dir, "train_classifier_ready_errors.csv")
    args.work_val_classifier_ready_errors_csv = os.path.join(args.work_split_csv_dir, "val_classifier_ready_errors.csv")
    args.work_test_classifier_ready_errors_csv = os.path.join(args.work_split_csv_dir, "test_classifier_ready_errors.csv")

    # create all dirs
    for d in [
        args.work_root,
        args.work_train_masks_dir, args.work_val_masks_dir, args.work_test_masks_dir,
        args.work_train_roi_dir, args.work_val_roi_dir, args.work_test_roi_dir,
        args.work_train_masked_roi_dir, args.work_val_masked_roi_dir, args.work_test_masked_roi_dir,
        args.work_split_csv_dir, args.work_roi_meta_dir,
        args.train_masks_dir, args.val_masks_dir, args.test_masks_dir,
        args.train_roi_dir, args.val_roi_dir, args.test_roi_dir,
        args.train_masked_roi_dir, args.val_masked_roi_dir, args.test_masked_roi_dir,
        os.path.dirname(args.train_classifier_csv),
        os.path.dirname(args.val_classifier_csv),
        os.path.dirname(args.test_classifier_csv),
        os.path.dirname(args.train_classifier_ready_csv),
        os.path.dirname(args.val_classifier_ready_csv),
        os.path.dirname(args.test_classifier_ready_csv),
        args.roi_meta_dir,
    ]:
        ensure_dir(d)

    print("\n" + "#" * 120)
    print("INPUT SUMMARY")
    print("#" * 120)
    print(f"WORK ROOT : {args.work_root}")
    quick_count_report("IMAGES_DIR", args.images_dir)
    quick_count_report("DRIVE TRAIN_MASKS_DIR BEFORE", args.train_masks_dir)
    quick_count_report("DRIVE VAL_MASKS_DIR BEFORE", args.val_masks_dir)
    quick_count_report("DRIVE TEST_MASKS_DIR BEFORE", args.test_masks_dir)

    print("\n" + "#" * 120)
    print("STEP 1/4 - TEMP PREDICTIONS TO /content WORK MASKS")
    print("#" * 120)

    sync_temp_predictions_to_work_masks(
        split_name="train",
        temp_root=args.temp_infer_root,
        work_mask_dir=args.work_train_masks_dir,
        clear_work_dir=args.clear_work_masks,
    )
    sync_temp_predictions_to_work_masks(
        split_name="val",
        temp_root=args.temp_infer_root,
        work_mask_dir=args.work_val_masks_dir,
        clear_work_dir=args.clear_work_masks,
    )
    sync_temp_predictions_to_work_masks(
        split_name="test",
        temp_root=args.temp_infer_root,
        work_mask_dir=args.work_test_masks_dir,
        clear_work_dir=args.clear_work_masks,
    )

    print("\n" + "#" * 120)
    print("WORK MASK COUNTS AFTER TEMP SYNC")
    print("#" * 120)
    quick_count_report("WORK TRAIN MASKS", args.work_train_masks_dir)
    quick_count_report("WORK VAL MASKS", args.work_val_masks_dir)
    quick_count_report("WORK TEST MASKS", args.work_test_masks_dir)

    print("\n" + "#" * 120)
    print("STEP 2/4 - ROI GENERATION IN /content")
    print("#" * 120)

    generate_rois_for_split(
        split_name="train",
        split_csv=args.train_csv,
        image_dir=args.images_dir,
        mask_dir=args.work_train_masks_dir,
        roi_dir=args.work_train_roi_dir,
        masked_roi_dir=args.work_train_masked_roi_dir,
        metadata_csv=os.path.join(args.work_roi_meta_dir, "train_roi_metadata.csv"),
        error_csv=os.path.join(args.work_roi_meta_dir, "train_roi_errors.csv"),
    )

    generate_rois_for_split(
        split_name="val",
        split_csv=args.val_csv,
        image_dir=args.images_dir,
        mask_dir=args.work_val_masks_dir,
        roi_dir=args.work_val_roi_dir,
        masked_roi_dir=args.work_val_masked_roi_dir,
        metadata_csv=os.path.join(args.work_roi_meta_dir, "val_roi_metadata.csv"),
        error_csv=os.path.join(args.work_roi_meta_dir, "val_roi_errors.csv"),
    )

    generate_rois_for_split(
        split_name="test",
        split_csv=args.test_csv,
        image_dir=args.images_dir,
        mask_dir=args.work_test_masks_dir,
        roi_dir=args.work_test_roi_dir,
        masked_roi_dir=args.work_test_masked_roi_dir,
        metadata_csv=os.path.join(args.work_roi_meta_dir, "test_roi_metadata.csv"),
        error_csv=os.path.join(args.work_roi_meta_dir, "test_roi_errors.csv"),
    )

    print("\n" + "#" * 120)
    print("STEP 3/4 - CLASSIFIER CSV BUILD IN /content")
    print("#" * 120)

    build_classifier_csv(
        split_name="train",
        split_csv=args.train_csv,
        image_dir=args.images_dir,
        mask_dir=args.work_train_masks_dir,
        roi_dir=args.work_train_roi_dir,
        masked_roi_dir=args.work_train_masked_roi_dir,
        out_csv=args.work_train_classifier_csv,
    )
    build_classifier_csv(
        split_name="val",
        split_csv=args.val_csv,
        image_dir=args.images_dir,
        mask_dir=args.work_val_masks_dir,
        roi_dir=args.work_val_roi_dir,
        masked_roi_dir=args.work_val_masked_roi_dir,
        out_csv=args.work_val_classifier_csv,
    )
    build_classifier_csv(
        split_name="test",
        split_csv=args.test_csv,
        image_dir=args.images_dir,
        mask_dir=args.work_test_masks_dir,
        roi_dir=args.work_test_roi_dir,
        masked_roi_dir=args.work_test_masked_roi_dir,
        out_csv=args.work_test_classifier_csv,
    )

    build_classifier_ready_csv(
        split_name="train",
        input_csv=args.work_train_classifier_csv,
        output_csv=args.work_train_classifier_ready_csv,
        error_csv=args.work_train_classifier_ready_errors_csv,
    )
    build_classifier_ready_csv(
        split_name="val",
        input_csv=args.work_val_classifier_csv,
        output_csv=args.work_val_classifier_ready_csv,
        error_csv=args.work_val_classifier_ready_errors_csv,
    )
    build_classifier_ready_csv(
        split_name="test",
        input_csv=args.work_test_classifier_csv,
        output_csv=args.work_test_classifier_ready_csv,
        error_csv=args.work_test_classifier_ready_errors_csv,
    )

    sync_work_outputs_to_drive(args)

    print("\n" + "#" * 120)
    print("FINAL DRIVE COUNTS")
    print("#" * 120)
    quick_count_report("DRIVE TRAIN MASKS AFTER", args.train_masks_dir)
    quick_count_report("DRIVE VAL MASKS AFTER", args.val_masks_dir)
    quick_count_report("DRIVE TEST MASKS AFTER", args.test_masks_dir)
    quick_count_report("DRIVE TRAIN ROI AFTER", args.train_roi_dir)
    quick_count_report("DRIVE VAL ROI AFTER", args.val_roi_dir)
    quick_count_report("DRIVE TEST ROI AFTER", args.test_roi_dir)
    quick_count_report("DRIVE TRAIN MASKED ROI AFTER", args.train_masked_roi_dir)
    quick_count_report("DRIVE VAL MASKED ROI AFTER", args.val_masked_roi_dir)
    quick_count_report("DRIVE TEST MASKED ROI AFTER", args.test_masked_roi_dir)

    print("\n" + "#" * 120)
    print("ALL DONE")
    print("#" * 120)


if __name__ == "__main__":
    main()