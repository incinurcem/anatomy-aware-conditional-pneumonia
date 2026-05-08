"""Mirror training-time dataset preprocessing for inference."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


_IMAGENET = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prepare_input(row, mode, img_size=224, normalize_mode="imagenet",
                  soft_mask_alpha=0.1, corner_mask_size=0.15):
    """row: pandas Series with image_path / roi_path / mask_crop_path columns."""
    if mode == "plain":
        path = row["image_path"]
    elif mode == "roi":
        path = row.get("roi_path") or row["image_path"]
    elif mode == "masked_roi":
        path = row.get("masked_roi_path") or row.get("roi_path") or row["image_path"]
    else:
        path = row["image_path"]

    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    pil_display = img.copy()
    arr = np.array(img).astype(np.float32) / 255.0  # HWC

    # Plain mode: corner masking (only training applied this in plain)
    if mode == "plain" and corner_mask_size > 0:
        cs = int(img_size * corner_mask_size)
        for ci, cj in [(0, 0), (0, img_size - cs),
                       (img_size - cs, 0), (img_size - cs, img_size - cs)]:
            arr[ci:ci+cs, cj:cj+cs, :] = 0.0

    # Masked-ROI: apply soft mask
    if mode == "masked_roi":
        mp = row.get("mask_crop_path")
        if isinstance(mp, str):
            try:
                mask = Image.open(mp).convert("L").resize((img_size, img_size))
                m = np.array(mask).astype(np.float32) / 255.0
                m = soft_mask_alpha + (1.0 - soft_mask_alpha) * m
                arr = arr * m[..., None]
            except Exception:
                pass

    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    if normalize_mode == "imagenet":
        t = _IMAGENET(t)
    return t.unsqueeze(0), pil_display