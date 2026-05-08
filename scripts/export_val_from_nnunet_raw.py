import os
import json
import shutil
import argparse


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet_raw_dataset", type=str, required=True,
                        help="Örn: /content/drive/MyDrive/Spring Semester/medical image analysis project/data/nnUNet_raw/Dataset502_LungMasksFT")
    parser.add_argument("--nnunet_preprocessed_dataset", type=str, required=True,
                        help="Örn: /content/drive/MyDrive/Spring Semester/medical image analysis project/data/nnUNet_preprocessed/Dataset502_LungMasksFT")
    parser.add_argument("--out_images", type=str, required=True)
    parser.add_argument("--out_labels", type=str, required=True)
    args = parser.parse_args()

    splits_path = os.path.join(args.nnunet_preprocessed_dataset, "splits_final.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_cases = splits[0]["val"]

    safe_mkdir(args.out_images)
    safe_mkdir(args.out_labels)

    for case_id in val_cases:
        src_img = os.path.join(args.nnunet_raw_dataset, "imagesTr", f"{case_id}_0000.png")
        src_lbl = os.path.join(args.nnunet_raw_dataset, "labelsTr", f"{case_id}.png")

        dst_img = os.path.join(args.out_images, f"{case_id}_0000.png")
        dst_lbl = os.path.join(args.out_labels, f"{case_id}.png")

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    print(f"[DONE] Exported val cases: {len(val_cases)}")


if __name__ == "__main__":
    main()