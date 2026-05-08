"""Paths, constants, and styling for the pneumonia dashboard."""
import os

BASE = os.environ.get(
    "PROJECT_BASE",
    "/content/drive/MyDrive/Spring Semester/medical image analysis project",
)
LOCAL = os.environ.get("LOCAL_DATA", "/content/data_local")

DEFAULTS = {
    "nnUNet_root":       f"{BASE}/outputs_seeds_nnUNet",
    "Otsu_root":         f"{BASE}/outputs_seeds_otsu",
    "nnUNet_test_csv":   f"{LOCAL}/conditional_v3/test/test_conditional_safe.csv",
    "Otsu_test_csv":     f"{LOCAL}/conditional_v3_otsu/test/test_conditional_safe.csv",
    "external_summary":  f"{BASE}/external_eval/external_summary.csv",
    "external_per_seed": f"{BASE}/external_eval/external_per_seed.csv",
}

EXPERIMENTS = [
    "classifier_plain",
    "classifier_roi",
    "classifier_masked_roi",
    "conditional_plain",
    "conditional_roi",
    "conditional_masked_roi",
]

SEEDS = [42, 123, 7]
PIPELINES = ["nnUNet", "Otsu"]

EXP_LABELS = {
    "classifier_plain":       "Plain",
    "classifier_roi":         "ROI",
    "classifier_masked_roi":  "Masked-ROI",
    "conditional_plain":      "Plain + Cond",
    "conditional_roi":        "ROI + Cond",
    "conditional_masked_roi": "Masked-ROI + Cond",
}

COLORS = {
    "nnUNet":  "#2E86AB",
    "Otsu":    "#E63946",
    "primary": "#264653",
    "accent":  "#F4A261",
    "good":    "#06A77D",
    "bad":     "#D62246",
}