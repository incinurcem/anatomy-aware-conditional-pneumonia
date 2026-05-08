import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = [
    "image_id",
    "split",
    "label",
    "image_path",
    "mask_path",
    "roi_path",
    "masked_roi_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create internal labeled classifier-ready train/val/test splits "
                    "from existing ready CSV files without rerunning preprocessing."
    )

    parser.add_argument(
        "--train_ready_csv",
        type=str,
        required=True,
        help="Path to existing train_classifier_ready.csv"
    )
    parser.add_argument(
        "--val_ready_csv",
        type=str,
        required=True,
        help="Path to existing val_classifier_ready.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save new internal ready CSV files"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Train ratio for final split"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation ratio for final split"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test ratio for final split"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--drop_missing_files",
        action="store_true",
        help="If set, rows with missing image/mask/roi/masked_roi files are removed"
    )

    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting output CSV files if they already exist"
    )

    return parser.parse_args()


def check_ratio_sum(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )


def ensure_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def validate_columns(df, csv_path):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def load_ready_csv(csv_path):
    ensure_exists(csv_path, "CSV")
    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)

    print("=" * 100)
    print(f"[LOAD] {csv_path}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    return df


def clean_dataframe(df, source_name):
    df = df.copy()

    # label boşları at
    before = len(df)
    df = df[df["label"].notna()].copy()
    after_label = len(df)

    # label int yap
    df["label"] = df["label"].astype(int)

    # image_id boşları at
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df = df[df["image_id"] != ""].copy()

    # path kolonlarını string yap
    for col in ["image_path", "mask_path", "roi_path", "masked_roi_path"]:
        df[col] = df[col].astype(str).str.strip()

    print("-" * 100)
    print(f"[CLEAN] {source_name}")
    print(f"rows before clean: {before}")
    print(f"rows after label filter: {after_label}")
    print(f"rows after image_id clean: {len(df)}")
    print("label distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())

    return df


def optionally_filter_missing_files(df, source_name, drop_missing_files=False):
    df = df.copy()

    for col in ["image_path", "mask_path", "roi_path", "masked_roi_path"]:
        df[f"exists__{col}"] = df[col].apply(os.path.exists)

    print("-" * 100)
    print(f"[FILE CHECK] {source_name}")
    for col in ["image_path", "mask_path", "roi_path", "masked_roi_path"]:
        ok = int(df[f"exists__{col}"].sum())
        total = len(df)
        print(f"{col}: {ok} / {total} exists")

    if drop_missing_files:
        before = len(df)
        valid_mask = (
            df["exists__image_path"] &
            df["exists__mask_path"] &
            df["exists__roi_path"] &
            df["exists__masked_roi_path"]
        )
        df = df[valid_mask].copy()
        after = len(df)
        print(f"[DROP MISSING FILES] {source_name}: {before} -> {after}")

    drop_cols = [c for c in df.columns if c.startswith("exists__")]
    df.drop(columns=drop_cols, inplace=True)

    return df


def deduplicate_by_image_id(df):
    df = df.copy()

    before = len(df)
    dup_count = int(df.duplicated(subset=["image_id"]).sum())

    # Öncelik sırası:
    # 1) split train/val fark etmez, önce gelen kalsın
    # concat sırası train sonra val olacağı için train öncelikli kalır
    df = df.drop_duplicates(subset=["image_id"], keep="first").copy()

    after = len(df)

    print("-" * 100)
    print("[DEDUPLICATION]")
    print(f"duplicate image_id rows removed: {dup_count}")
    print(f"rows: {before} -> {after}")
    print("final label distribution:")
    print(df["label"].value_counts().sort_index())

    return df


def stratified_split(df, train_ratio, val_ratio, test_ratio, random_state):
    df = df.copy()

    # önce train ve temp ayır
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        stratify=df["label"]
    )

    # temp -> val + test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=temp_df["label"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def report_split(name, df):
    print("-" * 100)
    print(f"[{name}]")
    print(f"rows: {len(df)}")
    print("label distribution:")
    print(df["label"].value_counts().sort_index())

    for col in ["image_path", "mask_path", "roi_path", "masked_roi_path"]:
        exists_count = int(df[col].apply(os.path.exists).sum())
        print(f"{col} exists: {exists_count} / {len(df)}")


def save_csv(df, path, allow_overwrite=False):
    if os.path.exists(path) and not allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            f"Use --allow_overwrite if you want to overwrite it."
        )
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    check_ratio_sum(args.train_ratio, args.val_ratio, args.test_ratio)

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = load_ready_csv(args.train_ready_csv)
    val_df = load_ready_csv(args.val_ready_csv)

    train_df = clean_dataframe(train_df, "train_ready")
    val_df = clean_dataframe(val_df, "val_ready")

    train_df = optionally_filter_missing_files(
        train_df, "train_ready", drop_missing_files=args.drop_missing_files
    )
    val_df = optionally_filter_missing_files(
        val_df, "val_ready", drop_missing_files=args.drop_missing_files
    )

    # train öncelikli olacak şekilde concat
    merged_df = pd.concat([train_df, val_df], ignore_index=True)

    print("=" * 100)
    print("[MERGED]")
    print(f"merged rows: {len(merged_df)}")
    print("merged label distribution:")
    print(merged_df["label"].value_counts().sort_index())

    merged_df = deduplicate_by_image_id(merged_df)

    # minimum sınıf örnek sayısı kontrolü
    label_counts = merged_df["label"].value_counts()
    if label_counts.min() < 3:
        raise ValueError(
            f"Not enough samples in one of the classes for train/val/test split:\n{label_counts}"
        )

    train_internal, val_internal, test_internal = stratified_split(
        merged_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )

    # düzenli kolon sırası
    train_internal = train_internal[REQUIRED_COLUMNS].copy()
    val_internal = val_internal[REQUIRED_COLUMNS].copy()
    test_internal = test_internal[REQUIRED_COLUMNS].copy()

    report_split("TRAIN_INTERNAL", train_internal)
    report_split("VAL_INTERNAL", val_internal)
    report_split("TEST_INTERNAL", test_internal)

    train_out = os.path.join(args.output_dir, "train_classifier_ready.csv")
    val_out = os.path.join(args.output_dir, "val_classifier_ready.csv")
    test_out = os.path.join(args.output_dir, "test_classifier_ready.csv")

    save_csv(train_internal, train_out, allow_overwrite=args.allow_overwrite)
    save_csv(val_internal, val_out, allow_overwrite=args.allow_overwrite)
    save_csv(test_internal, test_out, allow_overwrite=args.allow_overwrite)

    # ek rapor
    report_rows = []
    for split_name, df in [
        ("train", train_internal),
        ("val", val_internal),
        ("test", test_internal),
    ]:
        row = {
            "split": split_name,
            "rows": len(df),
            "label_0": int((df["label"] == 0).sum()),
            "label_1": int((df["label"] == 1).sum()),
            "image_exists": int(df["image_path"].apply(os.path.exists).sum()),
            "mask_exists": int(df["mask_path"].apply(os.path.exists).sum()),
            "roi_exists": int(df["roi_path"].apply(os.path.exists).sum()),
            "masked_roi_exists": int(df["masked_roi_path"].apply(os.path.exists).sum()),
        }
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    report_out = os.path.join(args.output_dir, "internal_split_report.csv")
    report_df.to_csv(report_out, index=False)

    print("=" * 100)
    print("[DONE]")
    print(f"Saved train: {train_out}")
    print(f"Saved val  : {val_out}")
    print(f"Saved test : {test_out}")
    print(f"Saved report: {report_out}")
    print("=" * 100)


if __name__ == "__main__":
    main()