# otsu_segment.py
"""
Klasik Otsu-tabanlı akciğer segmentasyonu (CXR icin).
Hicbir pretrained model kullanmaz, saf goruntu isleme.

Algoritma:
  1) CLAHE ile lokal kontrast iyilestir
  2) Hafif Gaussian blur
  3) Otsu thresholding (bright=govde+kemik, dark=arkaplan+akciger)
  4) Floodfill ile govde silueti cikar (ic delikler dolar)
  5) Akciger = doldurulmus govde - orijinal bright bolge
  6) Morfolojik open + close
  7) En buyuk 2 connected component (sol + sag akciger)
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def otsu_lung_mask(image, clip_limit=2.0, tile_size=8,
                   morph_kernel=7, min_area_ratio=0.005):
    h, w = image.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    flood = otsu.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes_filled = cv2.bitwise_not(flood)
    body_filled = cv2.bitwise_or(otsu, holes_filled)

    lung = cv2.subtract(body_filled, otsu)

    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    lung = cv2.morphologyEx(lung, cv2.MORPH_OPEN, kernel, iterations=2)
    lung = cv2.morphologyEx(lung, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lung, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(image, dtype=np.uint8)

    min_area = int(min_area_ratio * h * w)
    candidates = [
        (i, int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if len(candidates) == 0:
        return np.zeros_like(image, dtype=np.uint8)

    candidates.sort(key=lambda x: x[1], reverse=True)
    keep_ids = [cid for cid, _ in candidates[:2]]

    final = np.zeros_like(image, dtype=np.uint8)
    for cid in keep_ids:
        final[labels == cid] = 255
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Orijinal CXR PNG klasoru")
    parser.add_argument("--output_dir", required=True,
                        help="Otsu maskelerinin kaydedilecegi klasor")
    parser.add_argument("--clip_limit", type=float, default=2.0)
    parser.add_argument("--tile_size", type=int, default=8)
    parser.add_argument("--morph_kernel", type=int, default=7)
    parser.add_argument("--min_area_ratio", type=float, default=0.005,
                        help="Minimum component area / image area")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.input_dir)
                    if f.lower().endswith(".png")])
    print(f"[INFO] Found {len(files)} PNG files in {args.input_dir}")

    fail_read = 0
    fail_empty = 0
    skipped = 0
    written = 0

    for fname in tqdm(files):
        img_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            fail_read += 1
            continue

        mask = otsu_lung_mask(
            img,
            clip_limit=args.clip_limit,
            tile_size=args.tile_size,
            morph_kernel=args.morph_kernel,
            min_area_ratio=args.min_area_ratio,
        )

        if mask.sum() == 0:
            fail_empty += 1

        cv2.imwrite(out_path, mask)
        written += 1

    print(f"\n[OK] Done.")
    print(f"  Written         : {written}")
    print(f"  Skipped (exists): {skipped}")
    print(f"  Read failures   : {fail_read}")
    print(f"  Empty masks     : {fail_empty}")


if __name__ == "__main__":
    main()