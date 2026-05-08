#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#n
"""
convert_dicom_to_png.py

Amaç
-----
RSNA Pneumonia Detection Challenge DICOM görüntülerini
proje pipeline'ında kullanılmak üzere PNG formatına dönüştürmek.

Özellikler
----------
1) .dcm dosyalarını okur
2) mümkünse VOI LUT uygular
3) MONOCHROME1 ise intensity inversion yapar
4) uint8 [0, 255] normalize eder
5) isteğe bağlı resize uygular
6) tüm klasörü veya seçili patient ID listesini dönüştürebilir
7) çıktı olarak <patient_id>.png üretir

Örnek kullanım
--------------
Tüm klasörü dönüştür:
    python scripts/convert_dicom_to_png.py \
        --dicom-dir dataset/Data/stage_2_train_images\ 2 \
        --output-dir dataset/processed/images_png

Seçili patient ID'leri dönüştür:
    python scripts/convert_dicom_to_png.py \
        --dicom-dir dataset/Data/stage_2_train_images\ 2 \
        --output-dir dataset/processed/images_png \
        --patient-ids-csv outputs/rsna_dataset/rsna_patient_metadata.csv \
        --patient-id-col patientId

Resize ile dönüştür:
    python scripts/convert_dicom_to_png.py \
        --dicom-dir dataset/Data/stage_2_train_images\ 2 \
        --output-dir dataset/processed/images_png \
        --resize-width 512 \
        --resize-height 512
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:
    apply_voi_lut = None


LOGGER = logging.getLogger("convert_dicom_to_png")


# =========================================================
# Logging
# =========================================================

def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Görüntüyü uint8 [0, 255] aralığına normalize eder.
    """
    img = np.asarray(img).astype(np.float32)

    min_val = float(img.min())
    max_val = float(img.max())

    img = img - min_val
    denom = max_val - min_val

    if denom > 1e-8:
        img = img / denom

    img = img * 255.0
    return img.astype(np.uint8)


def safe_apply_voi_lut(ds: pydicom.dataset.FileDataset, img: np.ndarray) -> np.ndarray:
    """
    Mümkünse VOI LUT uygular, aksi halde ham görüntüyü döndürür.
    """
    if apply_voi_lut is None:
        return img

    try:
        return apply_voi_lut(img, ds)
    except Exception:
        return img


def dicom_to_uint8(
    dicom_path: str,
    resize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Tek bir DICOM dosyasını uint8 grayscale PNG'ye uygun diziye çevirir.

    Parameters
    ----------
    dicom_path : str
        DICOM dosya yolu
    resize : Optional[Tuple[int, int]]
        OpenCV formatında (width, height)

    Returns
    -------
    np.ndarray
        uint8 grayscale görüntü
    """
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array

    img = safe_apply_voi_lut(ds, img)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    img = normalize_to_uint8(img)

    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    return img


def convert_single_dicom(
    dicom_path: str,
    output_png_path: str,
    resize: Optional[Tuple[int, int]] = None,
    overwrite: bool = False
) -> str:
    """
    Tek bir DICOM dosyasını PNG'ye dönüştürür.
    """
    if os.path.exists(output_png_path) and not overwrite:
        return output_png_path

    ensure_dir(os.path.dirname(output_png_path))

    img = dicom_to_uint8(dicom_path=dicom_path, resize=resize)

    success = cv2.imwrite(output_png_path, img)
    if not success:
        raise RuntimeError(f"PNG yazılamadı: {output_png_path}")

    return output_png_path


def list_dicom_files(dicom_dir: str) -> List[str]:
    """
    Verilen klasördeki .dcm dosyalarını listeler.
    """
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM klasörü bulunamadı: {dicom_dir}")

    files = [
        f for f in os.listdir(dicom_dir)
        if f.lower().endswith(".dcm")
    ]
    files.sort()
    return files


def read_patient_ids_from_csv(
    csv_path: str,
    patient_id_col: str
) -> List[str]:
    """
    CSV içinden patient ID listesini okur.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)

    if patient_id_col not in df.columns:
        raise ValueError(
            f"'{patient_id_col}' sütunu CSV içinde yok: {csv_path}"
        )

    patient_ids = (
        df[patient_id_col]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    patient_ids.sort()
    return patient_ids


def convert_directory(
    dicom_dir: str,
    output_dir: str,
    resize: Optional[Tuple[int, int]] = None,
    overwrite: bool = False
) -> dict:
    """
    Tüm DICOM klasörünü PNG'ye dönüştürür.
    """
    ensure_dir(output_dir)

    dicom_files = list_dicom_files(dicom_dir)

    converted = 0
    skipped_existing = 0
    failed = 0
    failed_files: List[str] = []

    for file_name in tqdm(dicom_files, desc="Converting DICOM -> PNG"):
        dicom_path = os.path.join(dicom_dir, file_name)
        patient_id = os.path.splitext(file_name)[0]
        output_png_path = os.path.join(output_dir, f"{patient_id}.png")

        try:
            existed_before = os.path.exists(output_png_path)

            convert_single_dicom(
                dicom_path=dicom_path,
                output_png_path=output_png_path,
                resize=resize,
                overwrite=overwrite
            )

            if existed_before and not overwrite:
                skipped_existing += 1
            else:
                converted += 1

        except Exception as exc:
            failed += 1
            failed_files.append(file_name)
            LOGGER.exception("Dönüştürme hatası | file=%s | error=%s", file_name, str(exc))

    summary = {
        "mode": "all",
        "dicom_dir": dicom_dir,
        "output_dir": output_dir,
        "num_input_dicoms": len(dicom_files),
        "num_converted": converted,
        "num_skipped_existing": skipped_existing,
        "num_failed": failed,
        "failed_files": failed_files[:100],
        "resize": None if resize is None else {"width": resize[0], "height": resize[1]},
        "overwrite": bool(overwrite),
    }

    return summary


def convert_selected_patients(
    dicom_dir: str,
    output_dir: str,
    patient_ids: Sequence[str],
    resize: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
    strict_missing: bool = False
) -> dict:
    """
    Yalnızca seçili patient ID'lerin DICOM dosyalarını PNG'ye dönüştürür.
    """
    ensure_dir(output_dir)

    converted = 0
    skipped_existing = 0
    missing = 0
    failed = 0

    missing_ids: List[str] = []
    failed_ids: List[str] = []

    for patient_id in tqdm(patient_ids, desc="Converting selected DICOMs"):
        dicom_path = os.path.join(dicom_dir, f"{patient_id}.dcm")
        output_png_path = os.path.join(output_dir, f"{patient_id}.png")

        if not os.path.exists(dicom_path):
            missing += 1
            missing_ids.append(str(patient_id))
            if strict_missing:
                raise FileNotFoundError(f"Eksik DICOM: {dicom_path}")
            continue

        try:
            existed_before = os.path.exists(output_png_path)

            convert_single_dicom(
                dicom_path=dicom_path,
                output_png_path=output_png_path,
                resize=resize,
                overwrite=overwrite
            )

            if existed_before and not overwrite:
                skipped_existing += 1
            else:
                converted += 1

        except Exception as exc:
            failed += 1
            failed_ids.append(str(patient_id))
            LOGGER.exception(
                "Dönüştürme hatası | patient_id=%s | error=%s",
                str(patient_id),
                str(exc)
            )

    summary = {
        "mode": "selected",
        "dicom_dir": dicom_dir,
        "output_dir": output_dir,
        "num_requested_ids": len(patient_ids),
        "num_converted": converted,
        "num_skipped_existing": skipped_existing,
        "num_missing_dicoms": missing,
        "num_failed": failed,
        "missing_ids": missing_ids[:100],
        "failed_ids": failed_ids[:100],
        "resize": None if resize is None else {"width": resize[0], "height": resize[1]},
        "overwrite": bool(overwrite),
        "strict_missing": bool(strict_missing),
    }

    return summary


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RSNA DICOM görüntülerini PNG formatına dönüştür."
    )

    parser.add_argument(
        "--dicom-dir",
        type=str,
        required=True,
        help="DICOM klasörü"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="PNG çıktı klasörü"
    )

    parser.add_argument(
        "--patient-ids-csv",
        type=str,
        default=None,
        help="Sadece seçili patientId'leri dönüştürmek için CSV yolu"
    )
    parser.add_argument(
        "--patient-id-col",
        type=str,
        default="patientId",
        help="patient id sütun adı"
    )

    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="İsteğe bağlı resize width"
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="İsteğe bağlı resize height"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut PNG dosyalarının üstüne yaz"
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Seçili ID modunda eksik DICOM varsa hata ver"
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Özet çıktı JSON dosya yolu"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging seviyesi"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    resize = None
    if args.resize_width is not None or args.resize_height is not None:
        if args.resize_width is None or args.resize_height is None:
            raise ValueError("Resize için hem --resize-width hem --resize-height verilmelidir.")
        if args.resize_width <= 0 or args.resize_height <= 0:
            raise ValueError("Resize değerleri pozitif olmalıdır.")
        resize = (int(args.resize_width), int(args.resize_height))

    LOGGER.info("Başlıyor: convert_dicom_to_png.py")
    LOGGER.info("dicom_dir=%s", args.dicom_dir)
    LOGGER.info("output_dir=%s", args.output_dir)
    LOGGER.info("resize=%s", str(resize))
    LOGGER.info("overwrite=%s", str(bool(args.overwrite)))

    if args.patient_ids_csv is not None:
        patient_ids = read_patient_ids_from_csv(
            csv_path=args.patient_ids_csv,
            patient_id_col=args.patient_id_col
        )
        LOGGER.info("Seçili patient ID sayısı: %d", len(patient_ids))

        summary = convert_selected_patients(
            dicom_dir=args.dicom_dir,
            output_dir=args.output_dir,
            patient_ids=patient_ids,
            resize=resize,
            overwrite=args.overwrite,
            strict_missing=args.strict_missing
        )
    else:
        summary = convert_directory(
            dicom_dir=args.dicom_dir,
            output_dir=args.output_dir,
            resize=resize,
            overwrite=args.overwrite
        )

    LOGGER.info("Bitti. Özet:")
    LOGGER.info(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_json:
        ensure_dir(os.path.dirname(args.summary_json))
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        LOGGER.info("Summary JSON yazıldı: %s", args.summary_json)


if __name__ == "__main__":
    main()