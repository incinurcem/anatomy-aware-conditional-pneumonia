import os
import cv2
import json
import shutil
import random
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def is_image_file(path):
    return os.path.splitext(path.lower())[1] in IMG_EXTS


def list_all_images(root):
    files = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            full = os.path.join(dp, f)
            if is_image_file(full):
                files.append(full)
    files.sort()
    return files


def normalize_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower().strip()
    stem = stem.replace("_mask", "")
    stem = stem.replace("-mask", "")
    stem = stem.replace(" mask", "")
    stem = stem.replace("_label", "")
    stem = stem.replace("-label", "")
    stem = stem.replace(" label", "")
    return stem


def build_pairs(image_root, mask_root):
    image_files = list_all_images(image_root)
    mask_files = list_all_images(mask_root)

    image_map = {}
    for p in image_files:
        image_map[normalize_stem(p)] = p

    mask_map = {}
    for p in mask_files:
        mask_map[normalize_stem(p)] = p

    common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
    pairs = [(image_map[k], mask_map[k], k) for k in common_keys]

    return pairs, image_map, mask_map


def sanitize_to_grayscale_uint8(img):
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def sanitize_mask(mask):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8)
    return mask


def write_dataset_json(dataset_dir, dataset_name="LungMasksFT", labels=None):
    if labels is None:
        labels = {"background": 0, "lung": 1}

    dataset_json = {
        "channel_names": {"0": "xray"},
        "labels": labels,
        "numTraining": len(glob(os.path.join(dataset_dir, "labelsTr", "*.png"))),
        "file_ending": ".png"
    }

    with open(os.path.join(dataset_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True,
                        help="Kaggle dataset içindeki görüntü klasörü")
    parser.add_argument("--mask_root", type=str, required=True,
                        help="Kaggle dataset içindeki maske klasörü")
    parser.add_argument("--nnunet_raw_root", type=str, required=True,
                        help="Örn: /content/nnUNet_raw")
    parser.add_argument("--dataset_id", type=int, default=502)
    parser.add_argument("--dataset_name", type=str, default="LungMasksFT")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_folder = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    out_dir = os.path.join(args.nnunet_raw_root, dataset_folder)
    imagesTr = os.path.join(out_dir, "imagesTr")
    labelsTr = os.path.join(out_dir, "labelsTr")

    safe_mkdir(imagesTr)
    safe_mkdir(labelsTr)

    pairs, image_map, mask_map = build_pairs(args.image_root, args.mask_root)

    print(f"[INFO] Images found : {len(image_map)}")
    print(f"[INFO] Masks found  : {len(mask_map)}")
    print(f"[INFO] Matched pairs: {len(pairs)}")

    if len(pairs) == 0:
        raise RuntimeError("Hiç image-mask eşleşmesi bulunamadı.")

    valid_cases = []
    dropped = []

    for idx, (img_path, mask_path, stem) in enumerate(tqdm(pairs, desc="Converting")):
        img = sanitize_to_grayscale_uint8(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        mask = sanitize_mask(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED))

        if img is None or mask is None:
            dropped.append((img_path, mask_path, "read_error"))
            continue

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask.sum() == 0:
            dropped.append((img_path, mask_path, "empty_mask"))
            continue

        case_id = f"lung_{idx:04d}"

        img_out = os.path.join(imagesTr, f"{case_id}_0000.png")
        mask_out = os.path.join(labelsTr, f"{case_id}.png")

        cv2.imwrite(img_out, img)
        cv2.imwrite(mask_out, mask.astype(np.uint8))

        valid_cases.append(case_id)

    write_dataset_json(out_dir, dataset_name=args.dataset_name)

    # split file for nnU-Net
    random.shuffle(valid_cases)
    n_val = max(1, int(len(valid_cases) * args.val_ratio))
    val_cases = sorted(valid_cases[:n_val])
    train_cases = sorted(valid_cases[n_val:])

    splits_final = [{
        "train": train_cases,
        "val": val_cases
    }]

    preproc_dir = os.path.join(os.environ.get("nnUNet_preprocessed", "/content/drive/MyDrive/Spring Semester/medical image analysis project/data//nnUNet_preprocessed"),
                               dataset_folder)
    safe_mkdir(preproc_dir)

    with open(os.path.join(preproc_dir, "splits_final.json"), "w", encoding="utf-8") as f:
        json.dump(splits_final, f, indent=2)

    with open(os.path.join(out_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "images_found": len(image_map),
            "masks_found": len(mask_map),
            "matched_pairs": len(pairs),
            "valid_cases": len(valid_cases),
            "dropped_cases": len(dropped),
            "dataset_folder": dataset_folder
        }, f, indent=2)

    print(f"[DONE] nnU-Net raw dataset hazır: {out_dir}")
    print(f"[DONE] Train: {len(train_cases)} | Val: {len(val_cases)}")
    print(f"[DONE] splits_final.json: {os.path.join(preproc_dir, 'splits_final.json')}")


if __name__ == "__main__":
    main()