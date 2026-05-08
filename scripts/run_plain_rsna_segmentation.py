import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# =========================================================
# 0) SABİT AYARLAR
# =========================================================

PROJECT_ROOT = "/content/drive/MyDrive/Spring Semester/medical image analysis project"

# RSNA preprocess edilmiş görüntüler
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "data", "processed_pre", "train")
TEST_ROOT = os.path.join(PROJECT_ROOT, "data", "processed_pre", "test")

# Eğer görüntüler farklı alt klasördeyse burada aday klasörleri deniyoruz
TRAIN_IMAGE_CANDIDATES = [
    os.path.join(TRAIN_ROOT, "images_png"),
    os.path.join(TRAIN_ROOT, "images"),
    TRAIN_ROOT,
]

TEST_IMAGE_CANDIDATES = [
    os.path.join(TEST_ROOT, "images_png"),
    os.path.join(TEST_ROOT, "images"),
    TEST_ROOT,
]

# Split CSV klasörü
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# nnU-Net prediction output
PRED_ROOT = os.path.join(PROJECT_ROOT, "data", "nnunet_predictions")
os.makedirs(PRED_ROOT, exist_ok=True)

# ROI output
ROI_ROOT = os.path.join(PROJECT_ROOT, "data", "roi_outputs")
os.makedirs(ROI_ROOT, exist_ok=True)

# Eğitilmiş model klasörü (checkpoint dosyası değil!)
MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "nnUNet_results",
    "Dataset501_LungCXR",
    "nnUNetTrainer__nnUNetPlans__2d",
)

CHECKPOINT_NAME = "checkpoint_best.pth"
USE_FOLDS = (0,)

# Segmentasyon maskesi uzantısı
OUTPUT_MASK_EXT = ".png"

# Donanım
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Predictor worker sayıları
NUM_PROCESSES_PREPROCESSING = 2
NUM_PROCESSES_SEG_EXPORT = 2

# Train/val ayırma
VAL_RATIO = 0.10
RANDOM_SEED = 42

# ROI ayarları
ROI_PAD = 10
KEEP_COMPONENTS = 2

# Desteklenen görüntü uzantıları
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


# =========================================================
# 1) YARDIMCI FONKSİYONLAR
# =========================================================

def assert_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} bulunamadı: {path}")


def find_first_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Hiçbiri bulunamadı: {candidates}")


def list_image_files(image_dir: str):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = sorted(list(set(files)))
    return files


def patient_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    stem = stem.replace("_0000", "")
    return stem


def read_gray_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return img


def read_mask_as_binary(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask okunamadı: {path}")
    return (mask > 0).astype(np.uint8)


def keep_largest_components(mask_bin: np.ndarray, num_components: int = 2) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    if num_labels <= 1:
        return mask_bin

    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comps.append((i, area))

    comps = sorted(comps, key=lambda x: x[1], reverse=True)
    keep_ids = {i for i, _ in comps[:num_components]}

    out = np.zeros_like(mask_bin, dtype=np.uint8)
    for i in keep_ids:
        out[labels == i] = 1

    return out


def extract_bbox_from_mask(mask_bin: np.ndarray, pad: int = 10):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    h, w = mask_bin.shape
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    return x_min, y_min, x_max, y_max


def crop_with_bbox(img: np.ndarray, bbox):
    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max + 1, x_min:x_max + 1]


def make_masked_image(img: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    return (img * mask_bin).astype(img.dtype)


def build_mask_index(pred_dir: str):
    idx = {}
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.nii.gz"]:
        for p in glob.glob(os.path.join(pred_dir, ext)):
            base = os.path.basename(p)
            if base.endswith(".nii.gz"):
                stem = base[:-7]
            else:
                stem = os.path.splitext(base)[0]
            stem = stem.replace("_0000", "")
            idx[stem] = p
    return idx


# =========================================================
# 2) RSNA TRAIN / VAL / TEST SPLIT CSV OLUŞTUR
# =========================================================

def build_segmentation_input_csvs():
    train_image_dir = find_first_existing_dir(TRAIN_IMAGE_CANDIDATES)
    test_image_dir = find_first_existing_dir(TEST_IMAGE_CANDIDATES)

    train_files = list_image_files(train_image_dir)
    test_files = list_image_files(test_image_dir)

    if len(train_files) == 0:
        raise RuntimeError(f"Train görüntüsü bulunamadı: {train_image_dir}")
    if len(test_files) == 0:
        raise RuntimeError(f"Test görüntüsü bulunamadı: {test_image_dir}")

    train_df = pd.DataFrame({
        "patientId": [patient_id_from_path(p) for p in train_files],
        "image_path": train_files,
        "split": "train",
    })

    test_df = pd.DataFrame({
        "patientId": [patient_id_from_path(p) for p in test_files],
        "image_path": test_files,
        "split": "test",
    })

    # train -> train/val ayır
    rng = random.Random(RANDOM_SEED)
    ids = list(train_df["patientId"].astype(str).unique())
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * VAL_RATIO))
    val_ids = set(ids[:n_val])

    val_df = train_df[train_df["patientId"].astype(str).isin(val_ids)].copy()
    new_train_df = train_df[~train_df["patientId"].astype(str).isin(val_ids)].copy()

    new_train_df["split"] = "train"
    val_df["split"] = "val"

    train_csv = os.path.join(SPLITS_DIR, "train_segmentation_input.csv")
    val_csv = os.path.join(SPLITS_DIR, "val_segmentation_input.csv")
    test_csv = os.path.join(SPLITS_DIR, "test_segmentation_input.csv")

    new_train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("\n=== Split CSV oluşturuldu ===")
    print("train:", train_csv, "rows=", len(new_train_df))
    print("val  :", val_csv, "rows=", len(val_df))
    print("test :", test_csv, "rows=", len(test_df))

    print("\nTrain örnek:")
    print(new_train_df.head(3))
    print("\nVal örnek:")
    print(val_df.head(3))
    print("\nTest örnek:")
    print(test_df.head(3))


# =========================================================
# 3) NNUNET MODEL DOĞRULAMA
# =========================================================

def validate_model_folder():
    assert_exists(MODEL_FOLDER, "MODEL_FOLDER")
    assert_exists(os.path.join(MODEL_FOLDER, "dataset.json"), "dataset.json")
    assert_exists(os.path.join(MODEL_FOLDER, "plans.json"), "plans.json")

    for f in USE_FOLDS:
        fold_dir = os.path.join(MODEL_FOLDER, f"fold_{f}")
        ckpt_path = os.path.join(fold_dir, CHECKPOINT_NAME)
        assert_exists(fold_dir, f"fold_{f}")
        assert_exists(ckpt_path, f"fold_{f}/{CHECKPOINT_NAME}")

    print("\n=== Model doğrulandı ===")
    print("MODEL_FOLDER :", MODEL_FOLDER)
    print("CHECKPOINT   :", CHECKPOINT_NAME)
    print("USE_FOLDS    :", USE_FOLDS)
    print("DEVICE       :", DEVICE)


# =========================================================
# 4) TEK SPLIT İÇİN PLAIN SEGMENTATION INFERENCE
# =========================================================

def run_nnunet_inference_for_split(split_name: str, overwrite: bool = False):
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}_segmentation_input.csv")
    assert_exists(csv_path, f"{split_name}_segmentation_input.csv")

    out_dir = os.path.join(PRED_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n===== {split_name.upper()} SEGMENTATION =====")
    print("CSV:", csv_path)
    print("OUT:", out_dir)

    df = pd.read_csv(csv_path)
    df["patientId"] = df["patientId"].astype(str)
    df["image_path"] = df["image_path"].astype(str)

    df["image_exists"] = df["image_path"].apply(os.path.exists)
    missing_df = df[~df["image_exists"]].copy()

    if len(missing_df) > 0:
        print(f"[UYARI] {split_name}: {len(missing_df)} görüntü yolu bulunamadı.")
        print(missing_df[["patientId", "image_path"]].head(10))

    df_valid = df[df["image_exists"]].copy().reset_index(drop=True)

    if len(df_valid) == 0:
        raise RuntimeError(f"{split_name} için geçerli görüntü bulunamadı.")

    input_cases = []
    output_files = []

    for _, row in df_valid.iterrows():
        patient_id = str(row["patientId"])
        img_path = str(row["image_path"])
        out_path = os.path.join(out_dir, patient_id + OUTPUT_MASK_EXT)

        if (not overwrite) and os.path.exists(out_path):
            continue

        input_cases.append([img_path])  # tek modalite
        output_files.append(out_path)

    print(f"{split_name}: toplam geçerli görüntü = {len(df_valid)}")
    print(f"{split_name}: bu çalıştırmada üretilecek çıktı = {len(output_files)}")

    if len(output_files) > 0:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=(DEVICE == "cuda"),
            device=torch.device(DEVICE),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        predictor.initialize_from_trained_model_folder(
            MODEL_FOLDER,
            use_folds=USE_FOLDS,
            checkpoint_name=CHECKPOINT_NAME,
        )

        predictor.predict_from_files(
            input_cases,
            output_files,
            save_probabilities=False,
            overwrite=overwrite,
            num_processes_preprocessing=NUM_PROCESSES_PREPROCESSING,
            num_processes_segmentation_export=NUM_PROCESSES_SEG_EXPORT,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
    else:
        print(f"{split_name}: yeni üretilecek çıktı yok.")

    manifest_rows = []
    for _, row in df_valid.iterrows():
        patient_id = str(row["patientId"])
        img_path = str(row["image_path"])
        pred_path = os.path.join(out_dir, patient_id + OUTPUT_MASK_EXT)

        manifest_rows.append({
            "patientId": patient_id,
            "image_path": img_path,
            "mask_path": pred_path,
            "mask_exists": os.path.exists(pred_path),
            "split": split_name,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(out_dir, f"{split_name}_prediction_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"{split_name}: mask_exists = {int(manifest_df['mask_exists'].sum())}/{len(manifest_df)}")
    print("manifest:", manifest_path)

    return manifest_df


# =========================================================
# 5) SPLIT CSV'LERİNE MASK PATH EKLE
# =========================================================

def update_csv_with_masks(split_name: str):
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}_segmentation_input.csv")
    pred_dir = os.path.join(PRED_ROOT, split_name)

    assert_exists(csv_path, f"{split_name}_segmentation_input.csv")
    assert_exists(pred_dir, f"{split_name} prediction klasörü")

    df = pd.read_csv(csv_path)
    df["patientId"] = df["patientId"].astype(str)

    mask_index = build_mask_index(pred_dir)

    df["mask_path"] = df["patientId"].map(mask_index)
    df["mask_exists"] = df["mask_path"].apply(
        lambda x: os.path.exists(x) if isinstance(x, str) else False
    )

    matched = int(df["mask_exists"].sum())
    unmatched = int((~df["mask_exists"]).sum())

    print(f"\n[{split_name}] matched={matched}, unmatched={unmatched}")
    if unmatched > 0:
        print(df.loc[~df["mask_exists"], ["patientId", "image_path", "mask_path"]].head(10))

    df.to_csv(csv_path, index=False)
    print("updated:", csv_path)


# =========================================================
# 6) TEK ÖRNEK İÇİN ROI + MASKED ROI ÜRET
# =========================================================

def process_single_image_and_mask(image_path: str, mask_path: str):
    img = read_gray_image(image_path)
    mask_bin = read_mask_as_binary(mask_path)

    mask_bin = keep_largest_components(mask_bin, num_components=KEEP_COMPONENTS)

    bbox = extract_bbox_from_mask(mask_bin, pad=ROI_PAD)
    if bbox is None:
        return {
            "ok": False,
            "reason": "empty_mask",
            "bbox": None,
            "roi": None,
            "masked_roi": None,
        }

    roi = crop_with_bbox(img, bbox)
    masked_full = make_masked_image(img, mask_bin)
    masked_roi = crop_with_bbox(masked_full, bbox)

    return {
        "ok": True,
        "reason": None,
        "bbox": bbox,
        "roi": roi,
        "masked_roi": masked_roi,
    }


# =========================================================
# 7) TEK SPLIT İÇİN ROI + MASKED ROI ÜRET
# =========================================================

def generate_roi_for_split(split_name: str, overwrite: bool = False):
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}_segmentation_input.csv")
    assert_exists(csv_path, f"{split_name}_segmentation_input.csv")

    df = pd.read_csv(csv_path)

    if "mask_path" not in df.columns:
        raise ValueError(f"{csv_path} içinde mask_path kolonu yok.")

    if "mask_exists" in df.columns:
        df = df[df["mask_exists"] == True].copy()

    split_root = os.path.join(ROI_ROOT, split_name)
    roi_dir = os.path.join(split_root, "roi")
    masked_roi_dir = os.path.join(split_root, "masked_roi")
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(masked_roi_dir, exist_ok=True)

    df["patientId"] = df["patientId"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["mask_path"] = df["mask_path"].astype(str)

    roi_paths = []
    masked_roi_paths = []
    roi_exists_list = []
    masked_roi_exists_list = []
    bbox_xmin_list = []
    bbox_ymin_list = []
    bbox_xmax_list = []
    bbox_ymax_list = []
    bbox_w_list = []
    bbox_h_list = []
    status_list = []

    total = len(df)
    success = 0
    failed = 0

    print(f"\n===== {split_name.upper()} ROI =====")
    print("rows:", total)

    for idx, row in df.iterrows():
        patient_id = str(row["patientId"])
        image_path = str(row["image_path"])
        mask_path = str(row["mask_path"])

        roi_path = os.path.join(roi_dir, f"{patient_id}.png")
        masked_roi_path = os.path.join(masked_roi_dir, f"{patient_id}.png")

        try:
            if (not overwrite) and os.path.exists(roi_path) and os.path.exists(masked_roi_path):
                result = process_single_image_and_mask(image_path, mask_path)

                if result["ok"]:
                    x_min, y_min, x_max, y_max = result["bbox"]

                    roi_paths.append(roi_path)
                    masked_roi_paths.append(masked_roi_path)
                    roi_exists_list.append(True)
                    masked_roi_exists_list.append(True)

                    bbox_xmin_list.append(x_min)
                    bbox_ymin_list.append(y_min)
                    bbox_xmax_list.append(x_max)
                    bbox_ymax_list.append(y_max)
                    bbox_w_list.append(x_max - x_min + 1)
                    bbox_h_list.append(y_max - y_min + 1)
                    status_list.append("ok_existing")
                    success += 1
                else:
                    roi_paths.append(pd.NA)
                    masked_roi_paths.append(pd.NA)
                    roi_exists_list.append(False)
                    masked_roi_exists_list.append(False)

                    bbox_xmin_list.append(pd.NA)
                    bbox_ymin_list.append(pd.NA)
                    bbox_xmax_list.append(pd.NA)
                    bbox_ymax_list.append(pd.NA)
                    bbox_w_list.append(pd.NA)
                    bbox_h_list.append(pd.NA)
                    status_list.append(result["reason"])
                    failed += 1

                continue

            result = process_single_image_and_mask(image_path, mask_path)

            if not result["ok"]:
                roi_paths.append(pd.NA)
                masked_roi_paths.append(pd.NA)
                roi_exists_list.append(False)
                masked_roi_exists_list.append(False)

                bbox_xmin_list.append(pd.NA)
                bbox_ymin_list.append(pd.NA)
                bbox_xmax_list.append(pd.NA)
                bbox_ymax_list.append(pd.NA)
                bbox_w_list.append(pd.NA)
                bbox_h_list.append(pd.NA)
                status_list.append(result["reason"])

                failed += 1
                continue

            roi = result["roi"]
            masked_roi = result["masked_roi"]
            x_min, y_min, x_max, y_max = result["bbox"]

            cv2.imwrite(roi_path, roi)
            cv2.imwrite(masked_roi_path, masked_roi)

            roi_paths.append(roi_path)
            masked_roi_paths.append(masked_roi_path)
            roi_exists_list.append(os.path.exists(roi_path))
            masked_roi_exists_list.append(os.path.exists(masked_roi_path))

            bbox_xmin_list.append(x_min)
            bbox_ymin_list.append(y_min)
            bbox_xmax_list.append(x_max)
            bbox_ymax_list.append(y_max)
            bbox_w_list.append(x_max - x_min + 1)
            bbox_h_list.append(y_max - y_min + 1)
            status_list.append("ok")

            success += 1

        except Exception as e:
            roi_paths.append(pd.NA)
            masked_roi_paths.append(pd.NA)
            roi_exists_list.append(False)
            masked_roi_exists_list.append(False)

            bbox_xmin_list.append(pd.NA)
            bbox_ymin_list.append(pd.NA)
            bbox_xmax_list.append(pd.NA)
            bbox_ymax_list.append(pd.NA)
            bbox_w_list.append(pd.NA)
            bbox_h_list.append(pd.NA)
            status_list.append(f"error: {str(e)}")

            failed += 1

        if (idx + 1) % 500 == 0:
            print(f"{split_name}: {idx + 1}/{total} işlendi")

    df["roi_path"] = roi_paths
    df["masked_roi_path"] = masked_roi_paths
    df["roi_exists"] = roi_exists_list
    df["masked_roi_exists"] = masked_roi_exists_list
    df["bbox_xmin"] = bbox_xmin_list
    df["bbox_ymin"] = bbox_ymin_list
    df["bbox_xmax"] = bbox_xmax_list
    df["bbox_ymax"] = bbox_ymax_list
    df["bbox_width"] = bbox_w_list
    df["bbox_height"] = bbox_h_list
    df["roi_status"] = status_list

    df.to_csv(csv_path, index=False)

    print(f"\n{split_name} ROI tamamlandı.")
    print("success:", success)
    print("failed :", failed)
    print("updated csv:", csv_path)

    return df


# =========================================================
# 8) RAPORLAMA
# =========================================================

def print_summary():
    print("\n================ FINAL SUMMARY ================")
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(SPLITS_DIR, f"{split}_segmentation_input.csv")
        if not os.path.exists(csv_path):
            print(f"{split}: csv yok")
            continue

        df = pd.read_csv(csv_path)

        print(f"\n--- {split.upper()} ---")
        print("rows:", len(df))

        if "mask_exists" in df.columns:
            print("mask_exists sum:", int(df["mask_exists"].sum()))
        else:
            print("mask_exists sum: yok")

        if "roi_exists" in df.columns:
            print("roi_exists sum:", int(df["roi_exists"].sum()))
        else:
            print("roi_exists sum: yok")

        if "masked_roi_exists" in df.columns:
            print("masked_roi_exists sum:", int(df["masked_roi_exists"].sum()))
        else:
            print("masked_roi_exists sum: yok")

        cols = [
            c for c in [
                "patientId",
                "image_path",
                "mask_path",
                "roi_path",
                "masked_roi_path",
                "roi_status",
            ]
            if c in df.columns
        ]

        if len(cols) > 0:
            print(df[cols].head(3))


# =========================================================
# 9) MAIN
# =========================================================

def main():
    print("=== Plain RSNA Segmentation Pipeline Başlıyor ===")

    build_segmentation_input_csvs()
    validate_model_folder()

    run_nnunet_inference_for_split("train", overwrite=False)
    run_nnunet_inference_for_split("val", overwrite=False)
    run_nnunet_inference_for_split("test", overwrite=False)

    update_csv_with_masks("train")
    update_csv_with_masks("val")
    update_csv_with_masks("test")

    generate_roi_for_split("train", overwrite=False)
    generate_roi_for_split("val", overwrite=False)
    generate_roi_for_split("test", overwrite=False)

    print_summary()
    print("\n=== Plain RSNA Segmentation Pipeline Bitti ===")


if __name__ == "__main__":
    main()