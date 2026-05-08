import os
import re
import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from PIL import Image


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


DEFAULT_NNUNET_RAW = "/content/drive/MyDrive/Spring Semester/medical image analysis project/data/nnUNet_raw"
DEFAULT_NNUNET_PREPROCESSED = "/content/drive/MyDrive/Spring Semester/medical image analysis project/data/nnUNet_preprocessed"
DEFAULT_NNUNET_RESULTS = "/content/drive/MyDrive/Spring Semester/medical image analysis project/data/nnUNet_results"

os.environ["nnUNet_raw"] = DEFAULT_NNUNET_RAW
os.environ["nnUNet_preprocessed"] = DEFAULT_NNUNET_PREPROCESSED
os.environ["nnUNet_results"] = DEFAULT_NNUNET_RESULTS

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

    df = df.rename(columns=rename_map)

    required = ["image_id", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}. Mevcut sütunlar: {df.columns.tolist()}")

    df["image_id"] = df["image_id"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return df


def ensure_grayscale_copy(src_path, dst_path):
    img = Image.open(src_path).convert("L")
    img.save(dst_path)


def prepare_nnunet_inputs(df, temp_input_dir):
    os.makedirs(temp_input_dir, exist_ok=True)

    kept = []
    missing = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing {temp_input_dir}"):
        image_id = str(row["image_id"])
        image_path = str(row["image_path"])

        if not os.path.exists(image_path):
            missing.append((image_id, image_path))
            continue

        dst_name = f"{image_id}_0000.png"
        dst_path = os.path.join(temp_input_dir, dst_name)
        ensure_grayscale_copy(image_path, dst_path)
        kept.append(image_id)

    return kept, missing


def require_env_paths():
    needed = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    vals = {}
    missing = []
    for k in needed:
        v = os.environ.get(k)
        vals[k] = v
        if not v:
            missing.append(k)

    if missing:
        raise EnvironmentError(
            f"nnU-Net environment değişkenleri eksik: {missing}. "
            "Önce nnUNet_raw, nnUNet_preprocessed, nnUNet_results tanımlanmalı."
        )

    for k, v in vals.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"{k} yolu bulunamadı: {v}")

    return vals


def list_available_datasets(results_dir):
    results_path = Path(results_dir)
    return sorted([p.name for p in results_path.glob("Dataset*") if p.is_dir()])


def resolve_dataset_name(dataset_id_or_name, results_dir):
    """
    '501' -> 'Dataset501_LungSeg' gibi gerçek klasör adını çözer.
    'Dataset501_LungSeg' verilirse onu doğrular.
    """
    results_path = Path(results_dir)
    requested = str(dataset_id_or_name).strip()

    if requested.startswith("Dataset"):
        ds_path = results_path / requested
        if ds_path.exists():
            return requested
        raise RuntimeError(
            f"İstenen dataset klasörü bulunamadı: {requested}\n"
            f"Mevcut datasetler: {list_available_datasets(results_dir)}"
        )

    # Numeric ID gibi davran
    if requested.isdigit():
        pattern = f"Dataset{int(requested):03d}_*"
        matches = sorted(results_path.glob(pattern))
        if len(matches) == 1:
            return matches[0].name
        if len(matches) > 1:
            raise RuntimeError(
                f"Dataset ID {requested} için birden fazla eşleşme bulundu: "
                f"{[m.name for m in matches]}"
            )
        raise RuntimeError(
            f"Dataset ID {requested} için sonuç klasörü bulunamadı.\n"
            f"Mevcut datasetler: {list_available_datasets(results_dir)}"
        )

    # serbest string arama
    exact = results_path / requested
    if exact.exists():
        return requested

    candidates = [d for d in list_available_datasets(results_dir) if requested.lower() in d.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"'{requested}' için birden fazla dataset eşleşti: {candidates}"
        )

    raise RuntimeError(
        f"'{requested}' için dataset bulunamadı.\n"
        f"Mevcut datasetler: {list_available_datasets(results_dir)}"
    )


def verify_model_exists(dataset_name, trainer, configuration, folds, results_dir):
    """
    nnUNet_results/DatasetXXX_Name altında istenen trainer/config/fold mevcut mu?
    """
    ds_root = Path(results_dir) / dataset_name
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset klasörü yok: {ds_root}")

    trainer_config_dirs = sorted(ds_root.glob(f"{trainer}__*__{configuration}"))
    if not trainer_config_dirs:
        all_dirs = [p.name for p in ds_root.iterdir() if p.is_dir()]
        raise RuntimeError(
            f"{dataset_name} altında trainer/config eşleşmesi bulunamadı.\n"
            f"Aranan: {trainer}__*__{configuration}\n"
            f"Mevcut klasörler: {all_dirs}"
        )

    chosen = trainer_config_dirs[0]

    missing_folds = []
    for f in folds:
        fold_dir = chosen / f"fold_{f}"
        if not fold_dir.exists():
            missing_folds.append(str(f))

    if missing_folds:
        existing = [p.name for p in chosen.iterdir() if p.is_dir() and p.name.startswith("fold_")]
        raise RuntimeError(
            f"Eksik fold klasörleri: {missing_folds}\n"
            f"Bulunan fold klasörleri: {existing}\n"
            f"Model yolu: {chosen}"
        )

    return chosen


def preflight_check(dataset_id_or_name, trainer, configuration, folds):
    envs = require_env_paths()
    dataset_name = resolve_dataset_name(dataset_id_or_name, envs["nnUNet_results"])
    model_dir = verify_model_exists(
        dataset_name=dataset_name,
        trainer=trainer,
        configuration=configuration,
        folds=folds,
        results_dir=envs["nnUNet_results"]
    )

    print("=" * 80)
    print("nnU-Net preflight OK")
    print("Resolved dataset :", dataset_name)
    print("Model dir        :", model_dir)
    print("Folds            :", folds)
    print("=" * 80)

    return dataset_name


def run_nnunet_predict(input_dir, output_dir, dataset_name, configuration, trainer, folds,
                       step_size=0.5, device="cuda", save_probabilities=False):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", dataset_name,
        "-c", configuration,
        "-tr", trainer,
        "-device", device,
        "--disable_tta"
    ]

    if folds:
        cmd += ["-f"] + [str(f) for f in folds]

    if step_size is not None:
        cmd += ["-step_size", str(step_size)]

    if save_probabilities:
        cmd += ["--save_probabilities"]

    print("Çalışan komut:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def move_predictions_to_final(df, nnunet_output_dir, final_mask_dir):
    os.makedirs(final_mask_dir, exist_ok=True)

    missing_preds = []
    saved = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Collecting preds -> {final_mask_dir}"):
        image_id = str(row["image_id"])
        pred_candidates = [
            os.path.join(nnunet_output_dir, f"{image_id}.png"),
            os.path.join(nnunet_output_dir, f"{image_id}.nii.gz"),
        ]

        found = None
        for p in pred_candidates:
            if os.path.exists(p):
                found = p
                break

        if found is None:
            missing_preds.append(image_id)
            continue

        dst_png = os.path.join(final_mask_dir, f"{image_id}.png")

        if found.endswith(".png"):
            shutil.copy2(found, dst_png)
        else:
            raise ValueError(
                f"{found} NIfTI formatında geldi. Bu pipeline png mask bekliyor. "
                "Ya output formatını png üret, ya da burada NIfTI->PNG dönüşümü ekle."
            )

        saved += 1

    return saved, missing_preds


def process_split(csv_path, split_name, base_temp_dir, base_output_dir,
                  dataset_name, configuration, trainer, folds, step_size, device,
                  save_probabilities=False):
    df = normalize_columns(read_csv_safe(csv_path))

    temp_input_dir = os.path.join(base_temp_dir, split_name, "inputs")
    temp_pred_dir = os.path.join(base_temp_dir, split_name, "preds")
    final_mask_dir = os.path.join(base_output_dir, split_name, "masks")

    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    if os.path.exists(temp_pred_dir):
        shutil.rmtree(temp_pred_dir)

    kept, missing_imgs = prepare_nnunet_inputs(df, temp_input_dir)
    print(f"[{split_name}] input hazırlandı. kept={len(kept)} missing={len(missing_imgs)}")

    run_nnunet_predict(
        input_dir=temp_input_dir,
        output_dir=temp_pred_dir,
        dataset_name=dataset_name,
        configuration=configuration,
        trainer=trainer,
        folds=folds,
        step_size=step_size,
        device=device,
        save_probabilities=save_probabilities
    )

    saved, missing_preds = move_predictions_to_final(df, temp_pred_dir, final_mask_dir)

    print("=" * 80)
    print(f"SPLIT: {split_name}")
    print(f"CSV            : {csv_path}")
    print(f"Final mask dir : {final_mask_dir}")
    print(f"Saved masks    : {saved}")
    print(f"Missing images : {len(missing_imgs)}")
    print(f"Missing preds  : {len(missing_preds)}")
    print("=" * 80)

    return {
        "split": split_name,
        "total": len(df),
        "saved": saved,
        "missing_images": len(missing_imgs),
        "missing_preds": len(missing_preds),
        "final_mask_dir": final_mask_dir
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--configuration", type=str, default="2d")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--folds", nargs="+", default=["0"])
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_probabilities", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)


    print("nnUNet_raw =", os.environ["nnUNet_raw"])
    print("nnUNet_preprocessed =", os.environ["nnUNet_preprocessed"])
    print("nnUNet_results =", os.environ["nnUNet_results"])

    dataset_name = preflight_check(
        dataset_id_or_name=args.dataset_id,
        trainer=args.trainer,
        configuration=args.configuration,
        folds=args.folds
    )

    summaries = []
    summaries.append(process_split(
        args.train_csv, "train", args.temp_dir, args.output_dir,
        dataset_name, args.configuration, args.trainer, args.folds,
        args.step_size, args.device, args.save_probabilities
    ))
    summaries.append(process_split(
        args.val_csv, "val", args.temp_dir, args.output_dir,
        dataset_name, args.configuration, args.trainer, args.folds,
        args.step_size, args.device, args.save_probabilities
    ))
    summaries.append(process_split(
        args.test_csv, "test", args.temp_dir, args.output_dir,
        dataset_name, args.configuration, args.trainer, args.folds,
        args.step_size, args.device, args.save_probabilities
    ))

    print("\nÖZET")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()