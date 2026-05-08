#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_uncertainty.py

MC Dropout based uncertainty estimation for pneumonia classifier.

Updated
-------
- Reads default paths from configs/paths.yaml
- Uses preprocess train/test PNG directories
- Loads checkpoints saved by train_classifier.py
  (supports model_state_dict, state_dict, or raw state dict)
- Supports the same model names as train_classifier.py
- Aligned with current project YAML keys from access audit
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import yaml
except ImportError:
    yaml = None


# =========================================================
# YAML / PATH HELPERS
# =========================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(yaml_path: Optional[str]) -> Dict[str, Any]:
    if yaml_path is None:
        return {}

    yaml_file = Path(yaml_path)
    if not yaml_file.is_absolute():
        yaml_file = get_project_root() / yaml_file

    if not yaml_file.exists():
        return {}

    if yaml is None:
        raise ImportError("PyYAML is required to load paths.yaml")

    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(d: Dict[str, Any], keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_default_image_dir(paths_cfg: Dict[str, Any], split: str) -> str:
    if split == "train":
        return deep_get(
            paths_cfg,
            ["data", "train_png_dir"],
            "data/processed_pre/train/images_png",
        )
    return deep_get(
        paths_cfg,
        ["data", "test_png_dir"],
        "data/processed_pre/test/images_png",
    )


def resolve_default_roi_dir(paths_cfg: Dict[str, Any]) -> Optional[str]:
    candidates = [
        deep_get(paths_cfg, ["roi", "nnunet_dir"]),
        deep_get(paths_cfg, ["roi", "roi_dir"]),
        deep_get(paths_cfg, ["outputs", "roi_dir"]),
        deep_get(paths_cfg, ["data", "roi_dir"]),
        deep_get(paths_cfg, ["data", "classifier_roi_dir"]),
    ]
    for c in candidates:
        if c:
            return c
    return None


def resolve_default_checkpoint(paths_cfg: Dict[str, Any]) -> str:
    return (
        deep_get(paths_cfg, ["classifier", "best_checkpoint"])
        or deep_get(paths_cfg, ["classifier", "checkpoint"])
        or "checkpoints/classifier/best.pt"
    )


def resolve_default_output_csv(paths_cfg: Dict[str, Any], split: str) -> str:
    explicit = deep_get(paths_cfg, ["uncertainty", "output_csv"])
    if explicit:
        explicit = str(explicit)
        if "{split}" in explicit:
            return explicit.format(split=split)

        root, ext = os.path.splitext(explicit)
        if split not in os.path.basename(root):
            return f"{root}_{split}{ext or '.csv'}"
        return explicit

    output_dir = deep_get(paths_cfg, ["uncertainty", "output_dir"], "outputs/uncertainty")
    return os.path.join(output_dir, f"uncertainty_results_{split}.csv")


# =========================================================
# REPRODUCIBILITY
# =========================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run MC Dropout uncertainty inference")
    parser.add_argument("--paths_yaml", type=str, default="configs/paths.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])

    temp_args, _ = parser.parse_known_args()
    paths_cfg = load_yaml_config(temp_args.paths_yaml)

    default_image_dir = resolve_default_image_dir(paths_cfg, temp_args.split)
    default_csv = deep_get(paths_cfg, ["data", "train_labels_csv"], "data/rsna/stage_2_train_labels.csv")
    default_roi = resolve_default_roi_dir(paths_cfg)
    default_ckpt = resolve_default_checkpoint(paths_cfg)
    default_out = resolve_default_output_csv(paths_cfg, temp_args.split)

    parser.add_argument("--image_dir", type=str, default=default_image_dir)
    parser.add_argument("--roi_dir", type=str, default=default_roi)
    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--output_csv", type=str, default=default_out)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_roi", action="store_true")

    parser.add_argument("--mean", type=float, default=0.485)
    parser.add_argument("--std", type=float, default=0.229)

    parser.add_argument("--positive_col", type=str, default="Target")
    parser.add_argument("--id_col", type=str, default="patientId")
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=["simplecnn", "resnet18", "resnet34", "efficientnet_b0"],
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--mi_threshold", type=float, default=0.05)
    parser.add_argument("--var_threshold", type=float, default=0.02)

    return parser.parse_args()


# =========================================================
# IMAGE TRANSFORMS
# =========================================================

def build_transforms(img_size: int, mean: float, std: float):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])


# =========================================================
# DATASET
# =========================================================

class UncertaintyDataset(Dataset):
    def __init__(
        self,
        image_ids: List[str],
        image_dir: str,
        roi_dir: Optional[str],
        transform=None,
        labels_map: Optional[Dict[str, int]] = None,
        use_roi: bool = False,
        ext: str = ".png",
    ):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.roi_dir = roi_dir
        self.transform = transform
        self.labels_map = labels_map if labels_map is not None else {}
        self.use_roi = use_roi
        self.ext = ext

    def __len__(self):
        return len(self.image_ids)

    def _resolve_path(self, image_id: str) -> Tuple[str, str]:
        filename = f"{image_id}{self.ext}"

        if self.use_roi and self.roi_dir is not None:
            roi_path = os.path.join(self.roi_dir, filename)
            if os.path.exists(roi_path):
                return roi_path, "roi"

        fallback_path = os.path.join(self.image_dir, filename)
        return fallback_path, "image_dir"

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path, source_type = self._resolve_path(image_id)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image could not be read: {image_path}")

        if self.transform is not None:
            img_tensor = self.transform(image=img)["image"]
        else:
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            img_tensor = torch.tensor(img, dtype=torch.float32)

        label = self.labels_map.get(image_id, -1)

        return {
            "image_id": image_id,
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_path": image_path,
            "source_type": source_type,
        }


# =========================================================
# MODELS
# =========================================================

class SimpleClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(model_name: str, dropout: float = 0.3) -> nn.Module:
    if model_name == "simplecnn":
        return SimpleClassifier(in_channels=1, dropout_p=dropout)

    try:
        import torchvision.models as models
    except Exception as e:
        raise ImportError("torchvision is required for resnet/efficientnet models") from e

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


# =========================================================
# CHECKPOINT LOADING
# =========================================================

def _normalize_saved_args(saved_args: Any) -> Dict[str, Any]:
    if saved_args is None:
        return {}
    if isinstance(saved_args, dict):
        return saved_args
    if hasattr(saved_args, "__dict__"):
        return vars(saved_args)
    return {}


def load_model(checkpoint_path: str, device: torch.device, model_name: str, dropout: float) -> Tuple[nn.Module, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    saved_args = {}
    if isinstance(checkpoint, dict):
        saved_args = _normalize_saved_args(checkpoint.get("args", {}))

    actual_model_name = saved_args.get("model_name", model_name)
    actual_dropout = float(saved_args.get("dropout", dropout))

    model = build_model(actual_model_name, actual_dropout)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("module.", "")] = v

    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}")

    meta = {
        "model_name": actual_model_name,
        "dropout": actual_dropout,
        "saved_args": saved_args,
    }
    return model, meta


# =========================================================
# DROPOUT ACTIVATION
# =========================================================

def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


# =========================================================
# LABEL CSV PARSING
# =========================================================

def build_labels_map(csv_path: str, id_col: str = "patientId", positive_col: str = "Target") -> Dict[str, int]:
    if not csv_path or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if id_col not in df.columns or positive_col not in df.columns:
        return {}

    grouped = df.groupby(id_col)[positive_col].max().reset_index()
    return {str(row[id_col]): int(row[positive_col]) for _, row in grouped.iterrows()}


def collect_image_ids(image_dir: str, ext: str = ".png") -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_ids = []
    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.startswith("."):
            continue
        if fname.lower().endswith(ext.lower()):
            image_ids.append(os.path.splitext(fname)[0])

    return sorted(image_ids)


# =========================================================
# UNCERTAINTY METRICS
# =========================================================

def binary_entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_uncertainty_from_mc_probs(mc_probs: np.ndarray) -> Dict[str, np.ndarray]:
    mean_prob = np