"""On-the-fly Otsu segmentation + prediction helpers for uploaded images."""
import io
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

_IMAGENET = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_image_from_upload(uploaded_file):
    """Handle PNG/JPG/JPEG/DICOM uploads. Returns (PIL RGB, PIL grayscale)."""
    name = uploaded_file.name.lower()
    if name.endswith((".dcm", ".dicom")):
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
                arr = arr.max() - arr  # invert
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            pil_gray = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        except ImportError:
            raise RuntimeError("Install pydicom for DICOM: pip install pydicom")
    else:
        pil_gray = Image.open(uploaded_file).convert("L")
    return pil_gray.convert("RGB"), pil_gray


def otsu_lung_mask(pil_gray, kernel_size=15):
    """Otsu threshold + morphology + 2 largest components."""
    arr = np.array(pil_gray)
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw  # lungs are darker
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return bw
    areas = stats[1:, cv2.CC_STAT_AREA]
    top2 = np.argsort(areas)[-2:] + 1
    mask = np.zeros_like(bw)
    for k_ in top2:
        mask[labels == k_] = 255
    return mask


def extract_roi(rgb_arr, mask, padding=15):
    """Tight bbox crop."""
    ys, xs = np.where(mask > 0)
    H, W = rgb_arr.shape[:2]
    if len(xs) == 0:
        return rgb_arr, (0, 0, W, H)
    x0 = max(0, xs.min() - padding); x1 = min(W, xs.max() + padding)
    y0 = max(0, ys.min() - padding); y1 = min(H, ys.max() + padding)
    return rgb_arr[y0:y1, x0:x1], (x0, y0, x1, y1)


def soft_mask_apply(rgb_arr, mask, alpha=0.1):
    m = mask.astype(np.float32) / 255.0
    m = alpha + (1 - alpha) * m
    return (rgb_arr.astype(np.float32) * m[..., None]).clip(0, 255).astype(np.uint8)


def make_overlay(rgb_arr, mask, color=(46, 134, 171), alpha=0.35):
    """Draw mask outline on the original image for display."""
    overlay = rgb_arr.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    tint = np.zeros_like(overlay)
    tint[mask > 0] = color
    return cv2.addWeighted(overlay, 1.0, tint, alpha, 0)


def preprocess_for_model(rgb_arr, img_size=224):
    """Standard ImageNet preprocessing."""
    pil = Image.fromarray(rgb_arr).convert("RGB").resize((img_size, img_size))
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    t = _IMAGENET(t).unsqueeze(0)
    return t, pil


@torch.no_grad()
def predict_proba(model, x, device):
    return torch.sigmoid(model(x.to(device)).squeeze()).item()


def severity_label(p):
    if p < 0.30:  return "Low",      "#06A77D"
    if p < 0.60:  return "Moderate", "#F4A261"
    if p < 0.80:  return "High",     "#E76F51"
    return            "Very High",   "#D62246"


"""Conditional feature extraction for uploaded images."""
import pandas as pd
from pathlib import Path

try:
    from skimage.feature import graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def _feat(name, gray_arr, mask, roi_gray):
    """Compute a single feature by name. Returns 0 if unknown."""
    H, W = gray_arr.shape
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    has_mask = len(xs) > 0

    # ---- Mask geometry ----
    if name in ("mask_area_ratio", "mask_area"):
        return m.sum() / (H * W)
    if name in ("mask_bbox_w", "mask_bbox_width"):
        return (xs.max() - xs.min() + 1) / W if has_mask else 0.0
    if name in ("mask_bbox_h", "mask_bbox_height"):
        return (ys.max() - ys.min() + 1) / H if has_mask else 0.0
    if name == "mask_aspect_ratio":
        if not has_mask: return 0.0
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        return bw / max(bh, 1)
    if name == "mask_centroid_y":
        return ys.mean() / H if has_mask else 0.0
    if name == "mask_centroid_x":
        return xs.mean() / W if has_mask else 0.0
    if name == "mask_bbox_fill_ratio":
        if not has_mask: return 0.0
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        return m.sum() / max(bw * bh, 1)
    if name == "mask_asymmetry":
        if not has_mask: return 0.0
        left = m[:, :W//2].sum()
        right = m[:, W//2:].sum()
        return abs(left - right) / max(left + right, 1)
    if name in ("mask_l_area_ratio", "mask_left_area_ratio"):
        return m[:, :W//2].sum() / (H * W // 2)
    if name in ("mask_r_area_ratio", "mask_right_area_ratio"):
        return m[:, W//2:].sum() / (H * W // 2)

    # ---- ROI intensity stats ----
    roi_pix = gray_arr[m > 0] if has_mask and m.sum() > 0 else gray_arr.flatten()
    if name == "roi_intensity_mean":   return float(roi_pix.mean())
    if name == "roi_intensity_std":    return float(roi_pix.std())
    if name == "roi_intensity_median": return float(np.median(roi_pix))
    if name == "roi_intensity_q1":     return float(np.percentile(roi_pix, 25))
    if name == "roi_intensity_q3":     return float(np.percentile(roi_pix, 75))
    if name == "roi_intensity_iqr":
        return float(np.percentile(roi_pix, 75) - np.percentile(roi_pix, 25))
    if name == "roi_intensity_min":    return float(roi_pix.min())
    if name == "roi_intensity_max":    return float(roi_pix.max())
    if name == "roi_intensity_entropy":
        hist, _ = np.histogram(roi_pix, bins=64, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-10)))

    # ---- Edge / gradient ----
    sx = cv2.Sobel(gray_arr, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_arr, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx**2 + sy**2)
    if name in ("roi_edge_sobel_density", "sobel_density", "roi_sobel_density"):
        return float((grad > grad.mean()).sum() / (H * W))
    if name in ("roi_laplacian_var", "roi_edge_laplacian_var", "laplacian_var"):
        return float(cv2.Laplacian(gray_arr, cv2.CV_32F).var())
    if name in ("roi_gradient_mean", "roi_edge_grad_mean", "gradient_mean"):
        return float(grad.mean())
    if name in ("roi_gradient_std", "roi_edge_grad_std", "gradient_std"):
        return float(grad.std())
    if name in ("roi_foreground_ratio", "roi_edge_foreground_ratio", "foreground_ratio"):
        return float(m.sum() / (H * W))
    if name in ("roi_edge_diff", "roi_lr_diff"):
        lm = m.copy(); lm[:, W//2:] = 0
        rm = m.copy(); rm[:, :W//2] = 0
        l = gray_arr[lm > 0].mean() if lm.sum() > 0 else 0
        r = gray_arr[rm > 0].mean() if rm.sum() > 0 else 0
        return float(abs(l - r))
    if name in ("roi_ud_diff", "roi_upper_lower_diff"):
        um = m.copy(); um[H//2:, :] = 0
        dm = m.copy(); dm[:H//2, :] = 0
        u = gray_arr[um > 0].mean() if um.sum() > 0 else 0
        d = gray_arr[dm > 0].mean() if dm.sum() > 0 else 0
        return float(abs(u - d))

    # ---- GLCM texture ----
    if name.startswith("glcm_") and _HAS_SKIMAGE:
        src = roi_gray if (roi_gray is not None and roi_gray.size > 0) else gray_arr
        g8 = (src // 32).astype(np.uint8)
        try:
            glcm = graycomatrix(g8, distances=[1], angles=[0],
                                levels=8, symmetric=True, normed=True)
            prop_map = {
                "glcm_contrast": "contrast",
                "glcm_dissimilarity": "dissimilarity",
                "glcm_homogeneity": "homogeneity",
                "glcm_energy": "energy",
                "glcm_correlation": "correlation",
                "glcm_ASM": "ASM",
            }
            if name in prop_map:
                return float(graycoprops(glcm, prop_map[name])[0, 0])
        except Exception:
            return 0.0

    return 0.0


def compute_conditional_features(gray_arr, mask, roi_gray, feature_names, csv_path=None):
    """
    Compute the 22-dim (or whatever cond_dim) conditional vector for an uploaded image.
    Optionally normalize using statistics from the training/test CSV.

    Args:
        gray_arr: full grayscale image (H, W) uint8
        mask: lung mask (H, W) uint8 (0/255)
        roi_gray: cropped grayscale ROI (h, w) uint8 — for GLCM
        feature_names: list of column names from CSV (defines order)
        csv_path: path to test/train CSV; used for z-score normalization stats

    Returns:
        numpy array shape (len(feature_names),) — normalized features.
    """
    raw = np.array([_feat(n, gray_arr, mask, roi_gray) for n in feature_names],
                   dtype=np.float32)

    # Apply z-score normalization using CSV column statistics
    if csv_path and Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path)
            for i, name in enumerate(feature_names):
                if name in df.columns:
                    col = df[name].dropna()
                    if len(col) > 1:
                        mu, sd = float(col.mean()), float(col.std())
                        if sd > 0:
                            raw[i] = (raw[i] - mu) / sd
        except Exception:
            pass

    return raw