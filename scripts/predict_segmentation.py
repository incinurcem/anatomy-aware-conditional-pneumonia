"""
predict_segmentation.py
#d
Purpose
-------
Run lung segmentation inference on chest X-ray images and save:
- binary masks
- overlay visualizations
- optional probability maps

Supported usage
---------------
1) Single image inference
2) Directory inference

Expected Inputs
---------------
images_dir/
    <patient_id>.png

or

single image path:
    demo/sample.png

Outputs
-------
pred_masks/
    <patient_id>.png

pred_overlays/
    <patient_id>.png

optional:
pred_probs/
    <patient_id>.png
"""

import os
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# =========================================================
# Utility functions
# =========================================================

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def load_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def save_image(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image -= image.min()
    denom = image.max() + 1e-8
    image = image / denom
    image = image * 255.0
    return image.astype(np.uint8)


def clahe_preprocess(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    return clahe.apply(image)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def collect_image_paths(input_path: str) -> List[str]:
    """
    If input_path is a file, return [file].
    If input_path is a directory, collect images inside it.
    """
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(valid_exts)
    ]
    paths.sort()
    return paths


def get_base_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# =========================================================
# Fallback segmentation model
# Replace with your real trained model loader
# =========================================================

class DummySegmentationModel:
    """
    Fallback heuristic segmentation model.
    This is only for keeping inference runnable before
    real model integration.
    """

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        img = clahe_preprocess(image)
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Heuristic center-focused thresholding
        thresh = np.percentile(blur, 40)
        prob = blur.astype(np.float32) / 255.0
        mask_seed = (blur > thresh).astype(np.float32)

        h, w = prob.shape
        center_prior = np.zeros_like(prob, dtype=np.float32)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        y1 = int(h * 0.05)
        y2 = int(h * 0.95)
        center_prior[y1:y2, x1:x2] = 1.0

        prob = 0.6 * prob + 0.4 * mask_seed
        prob = prob * center_prior
        prob = cv2.GaussianBlur(prob, (0, 0), sigmaX=5, sigmaY=5)
        prob = np.clip(prob, 0.0, 1.0)

        return prob


# =========================================================
# Real model integration helpers
# =========================================================

def load_segmentation_model(
    model_name: str = "unet",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
):
    """
    Replace this with your real model selection logic.

    Example:
        from pneumo_pipeline.seg.models.unet import UNet
        model = UNet(...)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.to(device)
        return model
    """
    # Fallback
    return DummySegmentationModel()


def is_torch_model(model: Any) -> bool:
    return TORCH_AVAILABLE and isinstance(model, nn.Module)


# =========================================================
# Torch inference helpers
# =========================================================

def prepare_torch_input(image: np.ndarray, image_size: int) -> np.ndarray:
    """
    image: HxW uint8 grayscale
    return normalized float32 tensor-like np array (1, 1, H, W)
    """
    img = clahe_preprocess(image)
    img = resize_image(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # C,H,W
    img = np.expand_dims(img, axis=0)  # B,C,H,W
    return img


def predict_proba_torch(
    model: Any,
    image: np.ndarray,
    image_size: int = 512,
    device: str = "cpu"
) -> np.ndarray:
    """
    Standard binary segmentation inference helper.
    Assumes output shape compatible with:
    - [B,1,H,W] logits
    - [B,H,W] logits
    """
    inp = prepare_torch_input(image, image_size=image_size)

    x = torch.from_numpy(inp).float().to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)

        if isinstance(out, (list, tuple)):
            out = out[0]

        if out.ndim == 4 and out.shape[1] == 1:
            out = torch.sigmoid(out)
            out = out[0, 0].detach().cpu().numpy()
        elif out.ndim == 3:
            out = torch.sigmoid(out)
            out = out[0].detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model output shape: {tuple(out.shape)}")

    out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.clip(out, 0.0, 1.0)
    return out


# =========================================================
# Mask postprocessing
# =========================================================

def keep_largest_components(mask: np.ndarray, top_k: int = 2) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return mask_u8

    comps = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comps.append((idx, area))

    comps.sort(key=lambda x: x[1], reverse=True)

    cleaned = np.zeros_like(mask_u8)
    for idx, _ in comps[:top_k]:
        cleaned[labels == idx] = 1

    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape

    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, flood_inv)

    return (filled > 0).astype(np.uint8)


def postprocess_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    keep_top_k: int = 2,
    fill_internal_holes: bool = True
) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    mask_u8 = keep_largest_components(mask_u8, top_k=keep_top_k)

    if fill_internal_holes:
        mask_u8 = fill_holes(mask_u8)

    return (mask_u8 > 0).astype(np.uint8)


# =========================================================
# Visualization
# =========================================================

def create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Draw mask contour on grayscale image.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def create_soft_overlay(image: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    """
    Probability heat overlay.
    """
    heat = normalize_to_uint8(prob_map)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.65, heat, 0.35, 0)
    return overlay


# =========================================================
# Core prediction logic
# =========================================================

def predict_probability_map(
    model: Any,
    image: np.ndarray,
    image_size: int = 512,
    device: str = "cpu"
) -> np.ndarray:
    if is_torch_model(model):
        return predict_proba_torch(model, image, image_size=image_size, device=device)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(image)

    raise ValueError("Model does not support probability prediction.")


def predict_mask_from_image(
    model: Any,
    image: np.ndarray,
    image_size: int = 512,
    threshold: float = 0.5,
    device: str = "cpu",
    postprocess: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    prob_map = predict_probability_map(
        model=model,
        image=image,
        image_size=image_size,
        device=device
    )

    mask = (prob_map >= threshold).astype(np.uint8)

    if postprocess:
        mask = postprocess_mask(mask)

    return prob_map, mask


def process_single_image(
    image_path: str,
    model: Any,
    masks_dir: str,
    overlays_dir: str,
    probs_dir: Optional[str],
    image_size: int = 512,
    threshold: float = 0.5,
    device: str = "cpu",
    save_prob_map: bool = False,
    save_soft_overlay: bool = False,
    soft_overlays_dir: Optional[str] = None,
    postprocess: bool = True
) -> Dict[str, Any]:
    image = load_grayscale_image(image_path)
    prob_map, mask = predict_mask_from_image(
        model=model,
        image=image,
        image_size=image_size,
        threshold=threshold,
        device=device,
        postprocess=postprocess
    )

    base_name = get_base_name(image_path)

    mask_path = os.path.join(masks_dir, f"{base_name}.png")
    overlay_path = os.path.join(overlays_dir, f"{base_name}.png")

    save_image(mask_path, (mask * 255).astype(np.uint8))
    overlay = create_overlay(image, mask)
    save_image(overlay_path, overlay)

    prob_path = None
    soft_overlay_path = None

    if save_prob_map and probs_dir is not None:
        prob_path = os.path.join(probs_dir, f"{base_name}.png")
        save_image(prob_path, normalize_to_uint8(prob_map))

    if save_soft_overlay and soft_overlays_dir is not None:
        soft_overlay_path = os.path.join(soft_overlays_dir, f"{base_name}.png")
        soft_overlay = create_soft_overlay(image, prob_map)
        save_image(soft_overlay_path, soft_overlay)

    mask_area = int(mask.sum())
    image_area = int(mask.shape[0] * mask.shape[1])
    mask_ratio = float(mask_area / max(1, image_area))

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "prob_path": prob_path,
        "soft_overlay_path": soft_overlay_path,
        "mask_area": mask_area,
        "mask_ratio": mask_ratio,
    }


# =========================================================
# Batch driver
# =========================================================

def run_inference(
    input_path: str,
    model_name: str,
    checkpoint_path: Optional[str],
    masks_dir: str,
    overlays_dir: str,
    probs_dir: Optional[str] = None,
    soft_overlays_dir: Optional[str] = None,
    image_size: int = 512,
    threshold: float = 0.5,
    device: str = "cpu",
    save_prob_map: bool = False,
    save_soft_overlay: bool = False,
    postprocess: bool = True
) -> List[Dict[str, Any]]:
    ensure_dir(masks_dir)
    ensure_dir(overlays_dir)

    if save_prob_map and probs_dir is not None:
        ensure_dir(probs_dir)

    if save_soft_overlay and soft_overlays_dir is not None:
        ensure_dir(soft_overlays_dir)

    model = load_segmentation_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    if is_torch_model(model):
        model.to(device)
        model.eval()

    image_paths = collect_image_paths(input_path)
    results = []

    for image_path in tqdm(image_paths, desc="Predicting segmentation"):
        try:
            result = process_single_image(
                image_path=image_path,
                model=model,
                masks_dir=masks_dir,
                overlays_dir=overlays_dir,
                probs_dir=probs_dir,
                image_size=image_size,
                threshold=threshold,
                device=device,
                save_prob_map=save_prob_map,
                save_soft_overlay=save_soft_overlay,
                soft_overlays_dir=soft_overlays_dir,
                postprocess=postprocess
            )
            results.append(result)
        except Exception as e:
            print(f"[WARN] Failed on {image_path}: {e}")

    return results


# =========================================================
# Summary
# =========================================================

def print_summary(results: List[Dict[str, Any]]) -> None:
    if len(results) == 0:
        print("No predictions generated.")
        return

    ratios = [r["mask_ratio"] for r in results]
    areas = [r["mask_area"] for r in results]

    print("\n======= SEGMENTATION PREDICTION SUMMARY =======")
    print(f"Number of images      : {len(results)}")
    print(f"Mean mask area        : {np.mean(areas):.2f}")
    print(f"Mean mask ratio       : {np.mean(ratios):.4f}")
    print(f"Min mask ratio        : {np.min(ratios):.4f}")
    print(f"Max mask ratio        : {np.max(ratios):.4f}")
    print("===============================================\n")


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run lung segmentation prediction on chest X-ray images.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input image path or directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet",
        help="Segmentation model name (e.g. unet, attnunet, unetpp, transunet, deeplabv3, nnunet)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="pred_masks",
        help="Directory to save predicted binary masks"
    )
    parser.add_argument(
        "--overlays_dir",
        type=str,
        default="pred_overlays",
        help="Directory to save contour overlay images"
    )
    parser.add_argument(
        "--probs_dir",
        type=str,
        default="pred_probs",
        help="Directory to save probability maps"
    )
    parser.add_argument(
        "--soft_overlays_dir",
        type=str,
        default="pred_soft_overlays",
        help="Directory to save soft probability overlays"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Inference resize size"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary mask"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda"
    )
    parser.add_argument(
        "--save_prob_map",
        action="store_true",
        help="Save grayscale probability maps"
    )
    parser.add_argument(
        "--save_soft_overlay",
        action="store_true",
        help="Save colored probability overlays"
    )
    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="Disable mask postprocessing"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = run_inference(
        input_path=args.input_path,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        masks_dir=args.masks_dir,
        overlays_dir=args.overlays_dir,
        probs_dir=args.probs_dir,
        soft_overlays_dir=args.soft_overlays_dir,
        image_size=args.image_size,
        threshold=args.threshold,
        device=args.device,
        save_prob_map=args.save_prob_map,
        save_soft_overlay=args.save_soft_overlay,
        postprocess=not args.no_postprocess
    )

    print_summary(results)


if __name__ == "__main__":
    main()