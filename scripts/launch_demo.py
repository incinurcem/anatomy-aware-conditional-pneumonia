"""
launch_demo.py
#d
Purpose
-------
Local end-to-end demo launcher for the pneumonia analysis pipeline.

Pipeline
--------
Input CXR
    -> optional segmentation
    -> lung mask
    -> ROI crop
    -> classifier prediction
    -> optional burden estimation
    -> optional radiomics extraction
    -> optional uncertainty estimation
    -> optional Grad-CAM
    -> final JSON-like result

This script is intentionally modular so it can be adapted
to your project-specific model classes with minimal changes.

Expected Usage
--------------
python launch_demo.py \
    --image_path demo/sample.png \
    --seg_checkpoint checkpoints/seg/best.pt \
    --clf_checkpoint checkpoints/cls/best.pt \
    --output_dir demo_outputs

Notes
-----
- By default this script uses simple fallback models if real project
  classes are unavailable.
- Replace DummySegmentationModel and DummyClassifierModel with your
  actual implementations when integrating into the full project.
"""

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# Optional torch import
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    T = None


# =========================================================
# General utilities
# =========================================================

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def load_grayscale_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


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


# =========================================================
# Basic image preprocessing
# =========================================================

def clahe_preprocess(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(image)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# =========================================================
# Simple fallback ROI helpers
# =========================================================

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    padding_ratio: float = 0.05,
    min_padding_px: int = 8
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = image_shape

    bw = x2 - x1
    bh = y2 - y1

    pad_x = max(min_padding_px, int(round(bw * padding_ratio)))
    pad_y = max(min_padding_px, int(round(bh * padding_ratio)))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return x1, y1, x2, y2


def crop_with_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


# =========================================================
# Demo result data structures
# =========================================================

@dataclass
class SegmentationResult:
    success: bool
    mask_path: Optional[str]
    overlay_path: Optional[str]
    roi_path: Optional[str]
    roi_bbox: Optional[Tuple[int, int, int, int]]
    mask_area: int
    lung_area_ratio: float


@dataclass
class ClassificationResult:
    success: bool
    probability: float
    prediction: int
    label_name: str
    logits: Optional[float]
    confidence: float


@dataclass
class BurdenResult:
    success: bool
    estimated_burden_ratio: float
    estimated_burden_level: str


@dataclass
class RadiomicsResult:
    success: bool
    features: Dict[str, float]


@dataclass
class UncertaintyResult:
    success: bool
    mean_probability: float
    std_probability: float
    predictive_entropy: float
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass
class GradCAMResult:
    success: bool
    heatmap_path: Optional[str]
    overlay_path: Optional[str]


@dataclass
class FinalDemoResult:
    image_path: str
    segmentation: Dict[str, Any]
    classification: Dict[str, Any]
    burden: Dict[str, Any]
    radiomics: Dict[str, Any]
    uncertainty: Dict[str, Any]
    gradcam: Dict[str, Any]


# =========================================================
# Fallback / dummy models
# Replace these with project-specific models later
# =========================================================

class DummySegmentationModel:
    """
    Fallback lung region estimator using simple thresholding heuristics.
    This is NOT a real medical segmentation model.
    It only keeps the demo script runnable before integration.
    """

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        img = clahe_preprocess(image)
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Heuristic threshold
        thresh_val = np.percentile(blur, 40)
        mask = (blur > thresh_val).astype(np.uint8)

        # Focus center region more strongly
        h, w = mask.shape
        center_mask = np.zeros_like(mask, dtype=np.uint8)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        y1 = int(h * 0.05)
        y2 = int(h * 0.95)
        center_mask[y1:y2, x1:x2] = 1
        mask = mask * center_mask

        # Morphological cleanup
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Keep largest 2 components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask.astype(np.uint8)

        component_areas = []
        for idx in range(1, num_labels):
            component_areas.append((idx, stats[idx, cv2.CC_STAT_AREA]))
        component_areas.sort(key=lambda x: x[1], reverse=True)

        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for idx, _ in component_areas[:2]:
            cleaned[labels == idx] = 1

        return cleaned.astype(np.uint8)


class DummyClassifierModel:
    """
    Fallback classifier using handcrafted intensity heuristic.
    This is not a learned model; it only provides a runnable demo.
    """

    def predict_probability(self, image: np.ndarray) -> Tuple[float, float]:
        image = image.astype(np.float32) / 255.0
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))
        score = 4.0 * (0.55 - mean_intensity) + 2.0 * std_intensity
        prob = sigmoid(score)
        logit = score
        return prob, logit


# =========================================================
# Optional torch helpers
# =========================================================

def default_torch_transform(image_size: int = 512):
    if not TORCH_AVAILABLE:
        return None
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


# =========================================================
# Model loading stubs
# Replace with your actual project imports/checkpoint logic
# =========================================================

def load_segmentation_model(checkpoint_path: Optional[str] = None):
    """
    Replace this with your true segmentation model loading logic.
    """
    # Example placeholder:
    # from pneumo_pipeline.seg.models.unet import UNet
    # model = UNet(...)
    # state = torch.load(checkpoint_path, map_location="cpu")
    # model.load_state_dict(state["model"])
    # model.eval()
    return DummySegmentationModel()


def load_classifier_model(checkpoint_path: Optional[str] = None):
    """
    Replace this with your true classifier model loading logic.
    """
    # Example placeholder:
    # from pneumo_pipeline.cls.models.resnet import PneumoClassifier
    # model = PneumoClassifier(...)
    # state = torch.load(checkpoint_path, map_location="cpu")
    # model.load_state_dict(state["model"])
    # model.eval()
    return DummyClassifierModel()


# =========================================================
# Segmentation step
# =========================================================

def create_segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Saveable grayscale overlay with mask contour.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image_rgb.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def run_segmentation_step(
    image: np.ndarray,
    seg_model: Any,
    output_dir: str,
    base_name: str
) -> SegmentationResult:
    mask = seg_model.predict_mask(image).astype(np.uint8)

    bbox = mask_to_bbox(mask)
    if bbox is None:
        return SegmentationResult(
            success=False,
            mask_path=None,
            overlay_path=None,
            roi_path=None,
            roi_bbox=None,
            mask_area=0,
            lung_area_ratio=0.0,
        )

    bbox = expand_bbox(bbox, image.shape, padding_ratio=0.05, min_padding_px=8)
    roi = crop_with_bbox(image, bbox)

    mask_path = os.path.join(output_dir, f"{base_name}_lung_mask.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_seg_overlay.png")
    roi_path = os.path.join(output_dir, f"{base_name}_roi.png")

    save_image(mask_path, (mask * 255).astype(np.uint8))
    save_image(roi_path, roi)

    overlay = create_segmentation_overlay(image, mask)
    save_image(overlay_path, overlay)

    mask_area = int(mask.sum())
    lung_area_ratio = float(mask_area / (image.shape[0] * image.shape[1]))

    return SegmentationResult(
        success=True,
        mask_path=mask_path,
        overlay_path=overlay_path,
        roi_path=roi_path,
        roi_bbox=bbox,
        mask_area=mask_area,
        lung_area_ratio=lung_area_ratio,
    )


# =========================================================
# Classification step
# =========================================================

def classify_image(
    image: np.ndarray,
    clf_model: Any
) -> ClassificationResult:
    prob, logit = clf_model.predict_probability(image)
    pred = int(prob >= 0.5)
    label_name = "pneumonia" if pred == 1 else "normal"
    confidence = prob if pred == 1 else (1.0 - prob)

    return ClassificationResult(
        success=True,
        probability=float(prob),
        prediction=pred,
        label_name=label_name,
        logits=float(logit),
        confidence=float(confidence),
    )


# =========================================================
# Burden estimation
# =========================================================

def estimate_burden_from_image_and_mask(
    image: np.ndarray,
    lung_mask: Optional[np.ndarray],
    cls_result: ClassificationResult
) -> BurdenResult:
    if cls_result.prediction == 0:
        return BurdenResult(
            success=True,
            estimated_burden_ratio=0.0,
            estimated_burden_level="none"
        )

    img = image.astype(np.float32)
    if lung_mask is None or lung_mask.sum() == 0:
        region = img
    else:
        region = img[lung_mask > 0]

    if region.size == 0:
        burden_ratio = 0.0
    else:
        threshold = np.percentile(region, 20)
        suspicious = (img < threshold).astype(np.uint8)
        if lung_mask is not None:
            suspicious = suspicious * (lung_mask > 0).astype(np.uint8)
        burden_ratio = float(suspicious.sum() / max(1, (lung_mask > 0).sum() if lung_mask is not None else suspicious.size))

    # scale by predicted probability
    burden_ratio = float(np.clip(burden_ratio * cls_result.probability * 1.5, 0.0, 1.0))

    if burden_ratio < 0.10:
        level = "low"
    elif burden_ratio < 0.25:
        level = "moderate"
    else:
        level = "high"

    return BurdenResult(
        success=True,
        estimated_burden_ratio=burden_ratio,
        estimated_burden_level=level
    )


# =========================================================
# Radiomics-like extraction
# =========================================================

def extract_simple_radiomics(image: np.ndarray, mask: Optional[np.ndarray]) -> RadiomicsResult:
    if mask is not None and mask.sum() > 0:
        values = image[mask > 0].astype(np.float32)
    else:
        values = image.reshape(-1).astype(np.float32)

    if values.size == 0:
        feats = {
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "min_intensity": 0.0,
            "max_intensity": 0.0,
            "median_intensity": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "entropy_16bin": 0.0,
        }
        return RadiomicsResult(success=True, features=feats)

    hist, _ = np.histogram(values, bins=16, range=(0, 256), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log2(hist))

    feats = {
        "mean_intensity": float(np.mean(values)),
        "std_intensity": float(np.std(values)),
        "min_intensity": float(np.min(values)),
        "max_intensity": float(np.max(values)),
        "median_intensity": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "entropy_16bin": float(entropy),
    }

    return RadiomicsResult(success=True, features=feats)


# =========================================================
# Uncertainty estimation
# =========================================================

def predictive_entropy_from_prob(prob: float) -> float:
    p = float(np.clip(prob, 1e-8, 1 - 1e-8))
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def run_uncertainty_estimation(
    image: np.ndarray,
    clf_model: Any,
    num_samples: int = 10
) -> UncertaintyResult:
    """
    For real MC Dropout models, you would run stochastic forward passes.
    Here we simulate mild perturbations for demo robustness.
    """
    probs = []

    for _ in range(num_samples):
        noise = np.random.normal(0, 3.0, size=image.shape).astype(np.float32)
        aug = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        prob, _ = clf_model.predict_probability(aug)
        probs.append(prob)

    probs = np.array(probs, dtype=np.float32)

    mean_p = float(np.mean(probs))
    std_p = float(np.std(probs))
    entropy = predictive_entropy_from_prob(mean_p)

    ci_low = float(np.clip(mean_p - 1.96 * std_p, 0.0, 1.0))
    ci_high = float(np.clip(mean_p + 1.96 * std_p, 0.0, 1.0))

    return UncertaintyResult(
        success=True,
        mean_probability=mean_p,
        std_probability=std_p,
        predictive_entropy=entropy,
        confidence_interval_low=ci_low,
        confidence_interval_high=ci_high,
    )


# =========================================================
# Grad-CAM-like visualization
# =========================================================

def generate_fake_gradcam(
    image: np.ndarray,
    output_dir: str,
    base_name: str
) -> GradCAMResult:
    """
    Placeholder Grad-CAM-like heatmap using Laplacian + blur.
    Replace with true Grad-CAM from your classifier.
    """
    img_f = image.astype(np.float32)
    lap = cv2.Laplacian(img_f, cv2.CV_32F)
    heat = np.abs(lap)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9, sigmaY=9)
    heat = normalize_to_uint8(heat)

    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.65, heat_color, 0.35, 0)

    heatmap_path = os.path.join(output_dir, f"{base_name}_gradcam_heatmap.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_gradcam_overlay.png")

    save_image(heatmap_path, heat)
    save_image(overlay_path, overlay)

    return GradCAMResult(
        success=True,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
    )


# =========================================================
# Final interpretation
# =========================================================

def build_text_summary(
    cls_result: ClassificationResult,
    burden_result: BurdenResult,
    uncertainty_result: UncertaintyResult
) -> str:
    label = cls_result.label_name
    prob = cls_result.probability
    burden = burden_result.estimated_burden_level
    unc = uncertainty_result.std_probability

    if label == "normal":
        return (
            f"Model prediction suggests no pneumonia "
            f"(probability={prob:.4f}, uncertainty_std={unc:.4f})."
        )

    return (
        f"Model prediction suggests pneumonia "
        f"(probability={prob:.4f}, burden={burden}, uncertainty_std={unc:.4f})."
    )


# =========================================================
# Full pipeline runner
# =========================================================

def run_demo_pipeline(
    image_path: str,
    output_dir: str,
    seg_checkpoint: Optional[str] = None,
    clf_checkpoint: Optional[str] = None,
    run_segmentation: bool = True,
    run_radiomics: bool = True,
    run_uncertainty: bool = True,
    run_gradcam: bool = True
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    image = load_grayscale_image(image_path)
    image = clahe_preprocess(image)

    seg_model = load_segmentation_model(seg_checkpoint) if run_segmentation else None
    clf_model = load_classifier_model(clf_checkpoint)

    # Segmentation
    if run_segmentation and seg_model is not None:
        seg_result = run_segmentation_step(image, seg_model, output_dir, base_name)
        lung_mask = None
        roi_image = image

        if seg_result.success and seg_result.mask_path is not None:
            lung_mask = cv2.imread(seg_result.mask_path, cv2.IMREAD_GRAYSCALE)
            lung_mask = (lung_mask > 0).astype(np.uint8) if lung_mask is not None else None
        else:
            lung_mask = None

        if seg_result.success and seg_result.roi_path is not None:
            roi_image = load_grayscale_image(seg_result.roi_path)
        else:
            roi_image = image
    else:
        seg_result = SegmentationResult(
            success=False,
            mask_path=None,
            overlay_path=None,
            roi_path=None,
            roi_bbox=None,
            mask_area=0,
            lung_area_ratio=0.0,
        )
        lung_mask = None
        roi_image = image

    # Classification
    cls_result = classify_image(roi_image, clf_model)

    # Burden
    burden_result = estimate_burden_from_image_and_mask(image, lung_mask, cls_result)

    # Radiomics
    if run_radiomics:
        radiomics_result = extract_simple_radiomics(image, lung_mask)
    else:
        radiomics_result = RadiomicsResult(success=False, features={})

    # Uncertainty
    if run_uncertainty:
        uncertainty_result = run_uncertainty_estimation(roi_image, clf_model, num_samples=10)
    else:
        uncertainty_result = UncertaintyResult(
            success=False,
            mean_probability=cls_result.probability,
            std_probability=0.0,
            predictive_entropy=0.0,
            confidence_interval_low=cls_result.probability,
            confidence_interval_high=cls_result.probability,
        )

    # Grad-CAM
    if run_gradcam:
        gradcam_result = generate_fake_gradcam(roi_image, output_dir, base_name)
    else:
        gradcam_result = GradCAMResult(success=False, heatmap_path=None, overlay_path=None)

    final = FinalDemoResult(
        image_path=image_path,
        segmentation=asdict(seg_result),
        classification=asdict(cls_result),
        burden=asdict(burden_result),
        radiomics=asdict(radiomics_result),
        uncertainty=asdict(uncertainty_result),
        gradcam=asdict(gradcam_result),
    )

    result_dict = asdict(final)
    result_dict["summary_text"] = build_text_summary(cls_result, burden_result, uncertainty_result)

    return result_dict


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Launch local pneumonia analysis demo.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input CXR image"
    )
    parser.add_argument(
        "--seg_checkpoint",
        type=str,
        default=None,
        help="Path to segmentation checkpoint"
    )
    parser.add_argument(
        "--clf_checkpoint",
        type=str,
        default=None,
        help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_outputs",
        help="Directory to save demo outputs"
    )
    parser.add_argument(
        "--no_segmentation",
        action="store_true",
        help="Disable segmentation step"
    )
    parser.add_argument(
        "--no_radiomics",
        action="store_true",
        help="Disable radiomics extraction"
    )
    parser.add_argument(
        "--no_uncertainty",
        action="store_true",
        help="Disable uncertainty estimation"
    )
    parser.add_argument(
        "--no_gradcam",
        action="store_true",
        help="Disable Grad-CAM generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result = run_demo_pipeline(
        image_path=args.image_path,
        output_dir=args.output_dir,
        seg_checkpoint=args.seg_checkpoint,
        clf_checkpoint=args.clf_checkpoint,
        run_segmentation=not args.no_segmentation,
        run_radiomics=not args.no_radiomics,
        run_uncertainty=not args.no_uncertainty,
        run_gradcam=not args.no_gradcam,
    )

    ensure_dir(args.output_dir)

    json_path = os.path.join(args.output_dir, "demo_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n========== DEMO RESULT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=================================\n")
    print(f"Saved final result JSON to: {json_path}")


if __name__ == "__main__":
    main()