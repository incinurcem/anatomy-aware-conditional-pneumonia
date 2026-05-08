import os
import shutil
import argparse
from glob import glob
from tqdm import tqdm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def list_images(input_dir):
    files = []
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if os.path.isfile(full) and os.path.splitext(f.lower())[1] in IMG_EXTS:
            files.append(full)
    files.sort()
    return files


def build_nnunet_input_folder(src_dir, nnunet_input_dir):
    safe_mkdir(nnunet_input_dir)
    image_paths = list_images(src_dir)
    mapping = {}

    for img_path in tqdm(image_paths, desc="Preparing nnU-Net input"):
        base = os.path.basename(img_path)
        stem, ext = os.path.splitext(base)
        dst_name = f"{stem}_0000{ext.lower()}"
        dst_path = os.path.join(nnunet_input_dir, dst_name)
        shutil.copy2(img_path, dst_path)
        mapping[stem] = {"src": img_path, "nnunet_input": dst_path}

    return mapping


def rename_outputs(raw_pred_dir, final_mask_dir):
    safe_mkdir(final_mask_dir)
    pred_files = []
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        pred_files.extend(glob(os.path.join(raw_pred_dir, f"*{ext}")))
    pred_files.sort()

    for p in tqdm(pred_files, desc="Renaming outputs"):
        name = os.path.basename(p)
        stem, _ = os.path.splitext(name)
        if stem.endswith("_0000"):
            stem = stem[:-5]
        dst = os.path.join(final_mask_dir, f"{stem}.png")
        shutil.copy2(p, dst)


def run_prediction(model_folder, input_dir, output_dir, checkpoint_name="checkpoint_best.pth", device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_folder,
        use_folds=(0,),
        checkpoint_name=checkpoint_name
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_dir,
        output_folder=output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--final_mask_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    nnunet_input_dir = os.path.join(args.work_dir, "nnunet_input")
    raw_pred_dir = os.path.join(args.work_dir, "nnunet_raw_preds")

    safe_mkdir(args.work_dir)
    safe_mkdir(args.final_mask_dir)

    build_nnunet_input_folder(args.src_dir, nnunet_input_dir)
    run_prediction(args.model_folder, nnunet_input_dir, raw_pred_dir,
                   checkpoint_name=args.checkpoint_name, device=args.device)
    rename_outputs(raw_pred_dir, args.final_mask_dir)

    print("[DONE] RSNA masks ready.")


if __name__ == "__main__":
    main()