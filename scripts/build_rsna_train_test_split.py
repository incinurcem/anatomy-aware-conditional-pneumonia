# build_rsna_train_test_split.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


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
        elif lc in ["target", "class", "pneumonia", "label"]:
            rename_map[c] = "label"
        elif lc in ["image", "img_path", "path", "image_path"]:
            rename_map[c] = "image_path"
    df = df.rename(columns=rename_map)

    required = ["image_id", "label", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}. Mevcut sütunlar: {df.columns.tolist()}")

    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["image_path"] = df["image_path"].astype(str)

    df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_labeled_csv", type=str, required=True,
                        help="RSNA labeled train verisinin tamamı")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Elindeki mevcut labeled validation csv. Aynen korunacak.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_df = normalize_columns(read_csv_safe(args.all_labeled_csv))
    val_df = normalize_columns(read_csv_safe(args.val_csv))

    val_ids = set(val_df["image_id"].astype(str).tolist())

    # val'de olanları labeled havuzdan çıkar
    remain_df = all_df[~all_df["image_id"].isin(val_ids)].copy().reset_index(drop=True)

    if len(remain_df) == 0:
        raise ValueError("Val çıkarıldıktan sonra kalan veri 0 oldu.")

    train_df, test_df = train_test_split(
        remain_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=remain_df["label"]
    )

    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_out = os.path.join(args.output_dir, "train_classifier.csv")
    val_out = os.path.join(args.output_dir, "val_classifier.csv")
    test_out = os.path.join(args.output_dir, "test_classifier.csv")

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("=" * 80)
    print("RSNA SPLIT OLUŞTURULDU")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")
    print("-" * 80)
    print("Train label dağılımı:")
    print(train_df["label"].value_counts(dropna=False))
    print("-" * 80)
    print("Val label dağılımı:")
    print(val_df["label"].value_counts(dropna=False))
    print("-" * 80)
    print("Test label dağılımı:")
    print(test_df["label"].value_counts(dropna=False))
    print("=" * 80)
    print("Kaydedildi:")
    print(train_out)
    print(val_out)
    print(test_out)


if __name__ == "__main__":
    main()