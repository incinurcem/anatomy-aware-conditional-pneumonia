"""
run_gradcam.py
#d
Purpose
-------
Generate Grad-CAM visualizations for binary pneumonia classification.

Supported modes
---------------
1) Single image
2) Directory of images

Outputs
-------
outputs/gradcam/
    heatmaps/<id>.png
    overlays/<id>.png
    predictions.csv

Notes
-----
- This script is written to be integration-friendly.
- Replace `load_classifier_model(...)` and `infer_logits(...)`
  with your real project-specific classifier logic.
"""

import os
import csv
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def collect_image_paths(input_path: str) -> List[str]:
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


def load_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def clahe_preprocess(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )
    return clahe.apply(image)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    x /= (x.max() + 1e-8)
    x = x * 255.0
    return x.astype(np.uint8)


def sigmoid_np(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def save_image(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)


# =========================================================
# Fallback classifier
# Replace with your real model later
# =========================================================

class DummyClassifierModel:
    """
    A tiny fallback CNN-like placeholder is not useful for real Grad-CAM.
    Instead, this class only provides heuristic logits and a pseudo-heatmap.

    For real Grad-CAM, use a real torch model and a real target layer.
    """
    def predict_probability(self, image: np.ndarray) -> Tuple[float, float]:
        x = image.astype(np.float32) / 255.0
        score = 4.0 * (0.55 - float(np.mean(x))) + 2.0 * float(np.std(x))
        prob = sigmoid_np(score)
        return float(prob), float(score)

    def pseudo_heatmap(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        lap = cv2.Laplacian(img, cv2.CV_32F)
        heat = np.abs(lap)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9, sigmaY=9)
        heat = heat.astype(np.float32)
        heat -= heat.min()
        heat /= (heat.max() + 1e-8)
        return heat


# =========================================================
# Model loading
# =========================================================

def load_classifier_model(
    model_name: str = "resnet18",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
):
    """
    Replace this with your true classifier loader.

    Example:
        from pneumo_pipeline.cls.models.resnet import PneumoClassifier
        model = PneumoClassifier(...)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model
    """
    return DummyClassifierModel()


def is_torch_model(model: Any) -> bool:
    return TORCH_AVAILABLE and isinstance(model, nn.Module)


# =========================================================
# Torch preprocessing
# =========================================================

def prepare_torch_input(image: np.ndarray, image_size: int = 512) -> np.ndarray:
    img = clahe_preprocess(image)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)   # C,H,W
    img = np.expand_dims(img, axis=0)   # B,C,H,W
    return img


# =========================================================
# Grad-CAM core class
# =========================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available for Grad-CAM.")

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cam: HxW float in [0,1]
            probs/logits raw output as numpy
        """
        self.model.zero_grad(set_to_none=True)

        output = self.model(x)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Binary classification handling
        # Cases:
        #   [B,1] logit
        #   [B] logit
        #   [B,2] logits
        if output.ndim == 2 and output.shape[1] == 1:
            logit = output[:, 0]
            prob = torch.sigmoid(logit)
            if class_idx is None:
                class_idx = int((prob[0] >= 0.5).item())
            target = logit[0] if class_idx == 1 else -logit[0]

        elif output.ndim == 1:
            logit = output
            prob = torch.sigmoid(logit)
            if class_idx is None:
                class_idx = int((prob[0] >= 0.5).item())
            target = logit[0] if class_idx == 1 else -logit[0]

        elif output.ndim == 2 and output.shape[1] == 2:
            probs = torch.softmax(output, dim=1)
            if class_idx is None:
                class_idx = int(torch.argmax(probs[0]).item())
            target = output[0, class_idx]

        else:
            raise ValueError(f"Unsupported classifier output shape: {tuple(output.shape)}")

        target.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations: [B,C,H,W]
        # gradients  : [B,C,H,W]
        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)   # [C,1,1]
        cam = (weights * acts).sum(dim=0)                # [H,W]
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam, output.detach().cpu().numpy()


# =========================================================
# Target layer finder
# =========================================================

def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(
            f"Layer '{module_name}' not found. "
            f"Available example names: {list(modules.keys())[:20]}"
        )
    return modules[module_name]


def auto_find_last_conv_layer(model: nn.Module) -> nn.Module:
    candidate = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            candidate = module
    if candidate is None:
        raise ValueError("No Conv2d layer found in model for Grad-CAM.")
    return candidate


# =========================================================
# Inference helpers
# =========================================================

def infer_logits_torch(model: nn.Module, image: np.ndarray, image_size: int, device: str) -> Tuple[float, float, int]:
    inp = prepare_torch_input(image, image_size=image_size)
    x = torch.from_numpy(inp).float().to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim == 2 and out.shape[1] == 1:
            logit = float(out[0, 0].item())
            prob = float(torch.sigmoid(out[0, 0]).item())
            pred = int(prob >= 0.5)

        elif out.ndim == 1:
            logit = float(out[0].item())
            prob = float(torch.sigmoid(out[0]).item())
            pred = int(prob >= 0.5)

        elif out.ndim == 2 and out.shape[1] == 2:
            probs = torch.softmax(out, dim=1)
            pred = int(torch.argmax(probs[0]).item())
            prob = float(probs[0, 1].item()) if out.shape[1] > 1 else float(probs[0, pred].item())
            logit = float(out[0, pred].item())

        else:
            raise ValueError(f"Unsupported output shape: {tuple(out.shape)}")

    return prob, logit, pred


def infer_logits_dummy(model: DummyClassifierModel, image: np.ndarray) -> Tuple[float, float, int]:
    prob, logit = model.predict_probability(image)
    pred = int(prob >= 0.5)
    return float(prob), float(logit), int(pred)


# =========================================================
# Visualization helpers
# =========================================================

def resize_cam_to_image(cam: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = np.clip(cam, 0.0, 1.0)
    return cam


def create_heatmap_image(cam: np.ndarray) -> np.ndarray:
    cam_u8 = normalize_to_uint8(cam)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    return heatmap


def create_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    heatmap = create_heatmap_image(cam)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


# =========================================================
# Single-image processing
# =========================================================

def process_single_image_torch(
    image_path: str,
    model: nn.Module,
    target_layer: nn.Module,
    image_size: int,
    target_class: Optional[int],
    device: str,
    heatmaps_dir: str,
    overlays_dir: str
) -> Dict[str, Any]:
    image = load_grayscale_image(image_path)
    base_name = get_base_name(image_path)

    prob, logit, pred = infer_logits_torch(model, image, image_size=image_size, device=device)

    inp = prepare_torch_input(image, image_size=image_size)
    x = torch.from_numpy(inp).float().to(device)

    gradcam = GradCAM(model, target_layer)
    try:
        cam, _ = gradcam(x, class_idx=target_class)
    finally:
        gradcam.remove_hooks()

    cam = resize_cam_to_image(cam, image.shape)
    heatmap_img = create_heatmap_image(cam)
    overlay_img = create_overlay(image, cam, alpha=0.35)

    heatmap_path = os.path.join(heatmaps_dir, f"{base_name}.png")
    overlay_path = os.path.join(overlays_dir, f"{base_name}.png")

    save_image(heatmap_path, heatmap_img)
    save_image(overlay_path, overlay_img)

    return {
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "probability": float(prob),
        "logit": float(logit),
        "prediction": int(pred),
        "target_class_used": int(pred if target_class is None else target_class),
        "label_name": "pneumonia" if int(pred) == 1 else "normal",
    }


def process_single_image_dummy(
    image_path: str,
    model: DummyClassifierModel,
    heatmaps_dir: str,
    overlays_dir: str
) -> Dict[str, Any]:
    image = load_grayscale_image(image_path)
    base_name = get_base_name(image_path)

    prob, logit, pred = infer_logits_dummy(model, image)
    cam = model.pseudo_heatmap(image)
    cam = resize_cam_to_image(cam, image.shape)

    heatmap_img = create_heatmap_image(cam)
    overlay_img = create_overlay(image, cam, alpha=0.35)

    heatmap_path = os.path.join(heatmaps_dir, f"{base_name}.png")
    overlay_path = os.path.join(overlays_dir, f"{base_name}.png")

    save_image(heatmap_path, heatmap_img)
    save_image(overlay_path, overlay_img)

    return {
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "probability": float(prob),
        "logit": float(logit),
        "prediction": int(pred),
        "target_class_used": int(pred),
        "label_name": "pneumonia" if int(pred) == 1 else "normal",
    }


# =========================================================
# Batch pipeline
# =========================================================

def save_predictions_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    ensure_dir(os.path.dirname(csv_path))
    fieldnames = [
        "image_path",
        "heatmap_path",
        "overlay_path",
        "probability",
        "logit",
        "prediction",
        "target_class_used",
        "label_name",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_gradcam_pipeline(
    input_path: str,
    model_name: str,
    checkpoint_path: Optional[str],
    output_dir: str,
    image_size: int = 512,
    target_layer_name: Optional[str] = None,
    target_class: Optional[int] = None,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    image_paths = collect_image_paths(input_path)

    heatmaps_dir = os.path.join(output_dir, "heatmaps")
    overlays_dir = os.path.join(output_dir, "overlays")
    csv_path = os.path.join(output_dir, "predictions.csv")

    ensure_dir(heatmaps_dir)
    ensure_dir(overlays_dir)

    model = load_classifier_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    results = []

    if is_torch_model(model):
        model.to(device)
        model.eval()

        if target_layer_name is not None:
            target_layer = get_module_by_name(model, target_layer_name)
        else:
            target_layer = auto_find_last_conv_layer(model)

        for image_path in tqdm(image_paths, desc="Running Grad-CAM"):
            try:
                res = process_single_image_torch(
                    image_path=image_path,
                    model=model,
                    target_layer=target_layer,
                    image_size=image_size,
                    target_class=target_class,
                    device=device,
                    heatmaps_dir=heatmaps_dir,
                    overlays_dir=overlays_dir
                )
                results.append(res)
            except Exception as e:
                print(f"[WARN] Failed on {image_path}: {e}")

    else:
        for image_path in tqdm(image_paths, desc="Running pseudo Grad-CAM"):
            try:
                res = process_single_image_dummy(
                    image_path=image_path,
                    model=model,
                    heatmaps_dir=heatmaps_dir,
                    overlays_dir=overlays_dir
                )
                results.append(res)
            except Exception as e:
                print(f"[WARN] Failed on {image_path}: {e}")

    save_predictions_csv(results, csv_path)
    return results


# =========================================================
# Summary
# =========================================================

def print_summary(results: List[Dict[str, Any]]) -> None:
    if len(results) == 0:
        print("No Grad-CAM results generated.")
        return

    probs = np.array([r["probability"] for r in results], dtype=np.float32)
    preds = np.array([r["prediction"] for r in results], dtype=np.int32)

    print("\n=============== GRAD-CAM SUMMARY ===============")
    print(f"Number of images      : {len(results)}")
    print(f"Mean probability      : {float(np.mean(probs)):.6f}")
    print(f"Std probability       : {float(np.std(probs)):.6f}")
    print(f"Predicted pneumonia   : {int((preds == 1).sum())}")
    print(f"Predicted normal      : {int((preds == 0).sum())}")
    print("================================================\n")


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for classifier predictions.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input image path or directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Classifier model name"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to trained classifier checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gradcam",
        help="Directory to save Grad-CAM outputs"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Classifier inference image size"
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default=None,
        help="Optional module name for Grad-CAM target layer"
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Optional class index to explain (0=normal, 1=pneumonia)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = run_gradcam_pipeline(
        input_path=args.input_path,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        target_layer_name=args.target_layer,
        target_class=args.target_class,
        device=args.device
    )

    print_summary(results)


if __name__ == "__main__":
    main()