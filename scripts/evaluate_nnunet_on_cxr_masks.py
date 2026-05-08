import os
import re
import json
import math
import glob
import shutil
import random
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scipy.ndimage import binary_erosion, distance_transform_edt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_stem(name: str) -> str:
    """
    Dosya eşleştirmeyi daha sağlam yapmak için:
    - uzantıyı kaldır
    - _mask, -mask, mask, _lungs, vb. son ekleri temizle
    - yalnızca gövde adı ile eşleştir
    """
    stem = Path(name).stem.lower()

    # bazen .png.png gibi durumlar olabilir
    while stem.endswith(".png") or stem.endswith(".jpg") or stem.endswith(".jpeg"):
        stem = Path(stem).stem.lower()

    suffix_patterns = [
        r'[_\- ]?mask$',
        r'[_\- ]?masks$',
        r'[_\- ]?lung$',
        r'[_\- ]?lungs$',
        r'[_\- ]?seg$',
        r'[_\- ]?label$',
        r'[_\- ]?labels$',
    ]
    for pat in suffix_patterns:
        stem = re.sub(pat, '', stem)

    stem = stem.strip("_- ")
    return stem


def is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def collect_files(root: str):
    files = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isfile(p) and is_image_file(p):
            files.append(p)
    return files


def build_pairs(image_root: str, mask_root: str):
    image_files = collect_files(image_root)
    mask_files = collect_files(mask_root)

    image_map = {}
    for p in image_files:
        key = normalize_stem(os.path.basename(p))
        image_map[key] = p

    mask_map = {}
    for p in mask_files:
        key = normalize_stem(os.path.basename(p))
        mask_map[key] = p

    common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
    only_images = sorted(set(image_map.keys()) - set(mask_map.keys()))
    only_masks = sorted(set(mask_map.keys()) - set(image_map.keys()))

    pairs = []
    for key in common_keys:
        pairs.append({
            "case_id": key,
            "image_path": image_map[key],
            "mask_path": mask_map[key],
        })

    return pairs, only_images, only_masks


def load_mask_as_binary(path: str):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask okunamadı: {path}")
    mask = (mask > 0).astype(np.uint8)
    return mask


def load_pred_as_binary(path: str):
    pred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise ValueError(f"Prediction okunamadı: {path}")
    pred = (pred > 0).astype(np.uint8)
    return pred


def dice_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    inter = np.logical_and(y_true, y_pred).sum()
    denom = y_true.sum() + y_pred.sum()

    if denom == 0:
        return 1.0
    return (2.0 * inter) / denom


def iou_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    if union == 0:
        return 1.0
    return inter / union


def get_surface(mask):
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = binary_erosion(mask.astype(bool))
    surface = np.logical_xor(mask.astype(bool), eroded)
    return surface


def surface_distances(mask_gt, mask_pred, spacing=(1.0, 1.0)):
    """
    GT ve prediction yüzey noktaları arasındaki karşılıklı mesafeler.
    """
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    if mask_gt.sum() == 0 and mask_pred.sum() == 0:
        return np.array([0.0]), np.array([0.0])

    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return None, None

    gt_surface = get_surface(mask_gt)
    pred_surface = get_surface(mask_pred)

    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]

    return dist_gt_to_pred, dist_pred_to_gt


def hd95(mask_gt, mask_pred, spacing=(1.0, 1.0)):
    d1, d2 = surface_distances(mask_gt, mask_pred, spacing=spacing)
    if d1 is None or d2 is None:
        return np.nan

    all_dists = np.concatenate([d1, d2])
    if len(all_dists) == 0:
        return 0.0
    return float(np.percentile(all_dists, 95))


def assd(mask_gt, mask_pred, spacing=(1.0, 1.0)):
    d1, d2 = surface_distances(mask_gt, mask_pred, spacing=spacing)
    if d1 is None or d2 is None:
        return np.nan

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return float((d1.mean() + d2.mean()) / 2.0)


def write_nnunet_inputs(pairs, nnunet_input_dir):
    """
    nnU-Net natural image input format:
    caseid_0000.png

    Burada görüntüleri ZORLA tek kanallı grayscale olarak kaydediyoruz.
    """
    ensure_dir(nnunet_input_dir)

    manifest = []
    for item in pairs:
        case_id = item["case_id"]
        src = item["image_path"]
        dst = os.path.join(nnunet_input_dir, f"{case_id}_0000.png")

        # doğrudan grayscale oku
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {src}")

        # garanti olsun: 2D değilse griye çevir
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # uint8 garanti
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        ok = cv2.imwrite(dst, img)
        if not ok:
            raise ValueError(f"nnU-Net input yazılamadı: {dst}")

        manifest.append({
            "case_id": case_id,
            "input_image": dst,
            "original_image": src,
            "gt_mask": item["mask_path"],
            "written_shape": tuple(img.shape),
            "written_dtype": str(img.dtype),
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(nnunet_input_dir, "input_manifest.csv"), index=False)
    return manifest_df

def run_nnunet_predict(
    input_dir,
    output_dir,
    dataset_id,
    model_folder,
    folds="0",
    checkpoint="checkpoint_best.pth",
    device="cuda",
    disable_tta=False,
):
    ensure_dir(output_dir)

    model_folder = str(Path(model_folder).resolve())

    # Önce console script'i dene
    cmd = [
        "nnUNetv2_predict_from_modelfolder",
        "-i", input_dir,
        "-o", output_dir,
        "-m", model_folder,
        "-f", str(folds),
        "-chk", checkpoint,
        "-device", device,
        "--disable_progress_bar"
    ]

    if disable_tta:
        cmd.append("--disable_tta")

    env = os.environ.copy()

    print("\n[INFO] Çalıştırılan komut:")
    print(" ".join(cmd))
    print("[INFO] model_folder =", model_folder)

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except FileNotFoundError:
        # Console script yoksa module üzerinden fallback
        cmd = [
          "nnUNetv2_predict_from_modelfolder",
          "-i", input_dir,
          "-o", output_dir,
          "-m", model_folder,
          "-f", str(folds),
          "-chk", checkpoint,
          "-device", device,
          "--disable_progress_bar"
        ]
        print("\n[INFO] Fallback komutu:")
        print(" ".join(cmd))
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    print("\n[STDOUT]\n", result.stdout)
    print("\n[STDERR]\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"nnUNet prediction başarısız oldu. returncode={result.returncode}")


def find_prediction_file(output_dir, case_id):
    candidates = [
        os.path.join(output_dir, f"{case_id}.png"),
        os.path.join(output_dir, f"{case_id}.jpg"),
        os.path.join(output_dir, f"{case_id}.jpeg"),
        os.path.join(output_dir, f"{case_id}.tif"),
        os.path.join(output_dir, f"{case_id}.tiff"),
        os.path.join(output_dir, f"{case_id}.bmp"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    # bazen farklı isimlendirme olabilir, gevşek arama yap
    globbed = glob.glob(os.path.join(output_dir, f"{case_id}*"))
    for g in globbed:
        if os.path.isfile(g) and is_image_file(g):
            return g

    return None


def evaluate_pairs(pairs, pred_dir, spacing=(1.0, 1.0)):
    rows = []

    for item in pairs:
        case_id = item["case_id"]
        gt_path = item["mask_path"]
        pred_path = find_prediction_file(pred_dir, case_id)

        if pred_path is None:
            rows.append({
                "case_id": case_id,
                "gt_mask_path": gt_path,
                "pred_mask_path": None,
                "dice": np.nan,
                "iou": np.nan,
                "hd95": np.nan,
                "assd": np.nan,
                "status": "prediction_missing",
            })
            continue

        gt = load_mask_as_binary(gt_path)
        pred = load_pred_as_binary(pred_path)

        if gt.shape != pred.shape:
            pred = cv2.resize(
                pred.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        rows.append({
            "case_id": case_id,
            "gt_mask_path": gt_path,
            "pred_mask_path": pred_path,
            "dice": float(dice_score(gt, pred)),
            "iou": float(iou_score(gt, pred)),
            "hd95": float(hd95(gt, pred, spacing=spacing)) if not np.isnan(hd95(gt, pred, spacing=spacing)) else np.nan,
            "assd": float(assd(gt, pred, spacing=spacing)) if not np.isnan(assd(gt, pred, spacing=spacing)) else np.nan,
            "status": "ok",
        })

    df = pd.DataFrame(rows)
    return df


def summarize_metrics(df):
    ok_df = df[df["status"] == "ok"].copy()

    summary = {
        "num_total": int(len(df)),
        "num_ok": int(len(ok_df)),
        "num_missing_prediction": int((df["status"] == "prediction_missing").sum()),
        "dice_mean": float(ok_df["dice"].mean()) if len(ok_df) else None,
        "dice_std": float(ok_df["dice"].std()) if len(ok_df) else None,
        "iou_mean": float(ok_df["iou"].mean()) if len(ok_df) else None,
        "iou_std": float(ok_df["iou"].std()) if len(ok_df) else None,
        "hd95_mean": float(ok_df["hd95"].mean()) if len(ok_df) else None,
        "hd95_std": float(ok_df["hd95"].std()) if len(ok_df) else None,
        "assd_mean": float(ok_df["assd"].mean()) if len(ok_df) else None,
        "assd_std": float(ok_df["assd"].std()) if len(ok_df) else None,
    }
    return summary


def save_overlay_examples(df, out_dir, max_examples=20):
    ensure_dir(out_dir)

    ok_df = df[df["status"] == "ok"].copy()
    if len(ok_df) == 0:
        return

    # En kötü Dice örnekleri
    worst_df = ok_df.sort_values("dice", ascending=True).head(max_examples)

    for _, row in worst_df.iterrows():
        case_id = row["case_id"]
        gt = load_mask_as_binary(row["gt_mask_path"])
        pred = load_pred_as_binary(row["pred_mask_path"])

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # görüntüyü tahmin dosyasından değil, gt mask ile aynı stem'de bulma yerine
        # bu basit versiyonda overlay mask-only üretelim
        h, w = gt.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # GT yeşil
        canvas[gt > 0, 1] = 255
        # Pred kırmızı
        canvas[pred > 0, 2] = 255

        save_path = os.path.join(out_dir, f"{case_id}_overlay.png")
        cv2.imwrite(save_path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    parser.add_argument("--folds", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable_tta", action="store_true")

    parser.add_argument("--spacing_y", type=float, default=1.0)
    parser.add_argument("--spacing_x", type=float, default=1.0)
    parser.add_argument("--skip_prediction", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("\n=== GT-mask'li CXR veri kümesinde nnU-Net evaluation başlıyor ===")
    print("image_root :", args.image_root)
    print("mask_root  :", args.mask_root)
    print("output_dir :", args.output_dir)

    pairs, only_images, only_masks = build_pairs(args.image_root, args.mask_root)

    pair_csv = os.path.join(args.output_dir, "matched_pairs.csv")
    pd.DataFrame(pairs).to_csv(pair_csv, index=False)

    pd.DataFrame({"image_without_mask": only_images}).to_csv(
        os.path.join(args.output_dir, "images_without_mask.csv"), index=False
    )
    pd.DataFrame({"mask_without_image": only_masks}).to_csv(
        os.path.join(args.output_dir, "masks_without_image.csv"), index=False
    )

    print(f"[INFO] Eşleşen çift sayısı: {len(pairs)}")
    print(f"[INFO] Mask bulunamayan image sayısı: {len(only_images)}")
    print(f"[INFO] Image bulunamayan mask sayısı: {len(only_masks)}")

    if len(pairs) == 0:
        raise RuntimeError("Hiç image-mask eşleşmesi bulunamadı. Dosya isimlerini kontrol et.")

    nnunet_input_dir = os.path.join(args.output_dir, "nnunet_input")
    pred_dir = os.path.join(args.output_dir, "predictions")
    overlay_dir = os.path.join(args.output_dir, "worst_overlays")

    write_nnunet_inputs(pairs, nnunet_input_dir)

    if not args.skip_prediction:
        run_nnunet_predict(
            input_dir=nnunet_input_dir,
            output_dir=pred_dir,
            dataset_id=args.dataset_id,
            model_folder=args.model_folder,
            folds=args.folds,
            checkpoint=args.checkpoint,
            device=args.device,
            disable_tta=args.disable_tta,
        )
    else:
        print("[INFO] --skip_prediction verildi, mevcut prediction klasörü kullanılacak.")

    metrics_df = evaluate_pairs(
        pairs=pairs,
        pred_dir=pred_dir,
        spacing=(args.spacing_y, args.spacing_x),
    )

    metrics_csv = os.path.join(args.output_dir, "segmentation_metrics_per_case.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    summary = summarize_metrics(metrics_df)
    summary_json = os.path.join(args.output_dir, "segmentation_metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_overlay_examples(metrics_df, overlay_dir, max_examples=20)

    print("\n=== Evaluation tamamlandı ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[INFO] Per-case CSV : {metrics_csv}")
    print(f"[INFO] Summary JSON : {summary_json}")
    print(f"[INFO] Overlay dir  : {overlay_dir}")


if __name__ == "__main__":
    main()