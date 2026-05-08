import os
import json
import math
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss, log_loss
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class ROITestDataset(Dataset):
    def __init__(self, csv_path, image_col="masked_roi_path", label_col="label", image_size=224):
        self.df = pd.read_csv(csv_path).copy()
        self.image_col = image_col
        self.label_col = label_col
        self.has_label = label_col in self.df.columns

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        image = Image.open(img_path).convert("L")
        image_t = self.transform(image)

        item = {
            "image": image_t,
            "image_path": img_path,
            "image_name": row["image_name"] if "image_name" in row.index else os.path.basename(img_path)
        }

        if self.has_label:
            item["label"] = torch.tensor(float(row[self.label_col]), dtype=torch.float32)

        return item


class PlainClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=False):
        super().__init__()

        if backbone_name == "resnet18":
            model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
            self.target_layer = self.encoder.layer4[-1]
        elif backbone_name == "resnet34":
            model = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
            self.target_layer = self.encoder.layer4[-1]
        elif backbone_name == "resnet50":
            model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
            self.target_layer = self.encoder.layer4[-1]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return logits.squeeze(1)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "nll": float(log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1]))
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
    return out


def compute_calibration(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    records = []

    for b in range(n_bins):
        mask = (bin_ids == b)
        if np.sum(mask) == 0:
            records.append({
                "bin_idx": b,
                "bin_start": float(bins[b]),
                "bin_end": float(bins[b+1]),
                "count": 0,
                "confidence": 0.0,
                "accuracy": 0.0,
                "gap": 0.0
            })
            continue

        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        gap = abs(acc - conf)

        ece += gap * (np.sum(mask) / len(y_true))
        mce = max(mce, gap)

        records.append({
            "bin_idx": b,
            "bin_start": float(bins[b]),
            "bin_end": float(bins[b+1]),
            "count": int(np.sum(mask)),
            "confidence": conf,
            "accuracy": acc,
            "gap": float(gap)
        })

    return float(ece), float(mce), pd.DataFrame(records)


def plot_reliability_diagram(calib_df, save_path):
    plt.figure(figsize=(6, 6))
    xs = calib_df["confidence"].values
    ys = calib_df["accuracy"].values
    counts = calib_df["count"].values

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.bar(xs, ys, width=0.08, alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, images, mc_runs=20):
    probs_list = []
    model.eval()
    enable_dropout(model)

    with torch.no_grad():
        for _ in range(mc_runs):
            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.unsqueeze(0))

    probs_all = torch.cat(probs_list, dim=0)
    mean_probs = probs_all.mean(dim=0)
    var_probs = probs_all.var(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8) + (1 - mean_probs) * torch.log(1 - mean_probs + 1e-8))
    return mean_probs, var_probs, entropy


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, input_tensor):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0]
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def overlay_cam_on_image(gray_image, cam, save_path):
    gray = np.array(gray_image.resize((224, 224))).astype(np.uint8)
    cam_resized = np.uint8(255 * cam)
    cam_resized = np.array(Image.fromarray(cam_resized).resize((224, 224)))

    heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
    gray_3 = np.stack([gray, gray, gray], axis=-1) / 255.0
    overlay = (0.5 * gray_3 + 0.5 * heatmap)
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="masked_roi_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mc_runs", type=int, default=20)
    parser.add_argument("--gradcam_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "gradcam"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ROITestDataset(
        csv_path=args.test_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        image_size=args.image_size
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = PlainClassifier(backbone_name=args.backbone, pretrained=False).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_probs = []
    all_labels = []
    all_names = []
    all_paths = []
    all_var = []
    all_entropy = []

    for batch in loader:
        images = batch["image"].to(device)
        mean_probs, var_probs, entropy = mc_dropout_predict(model, images, mc_runs=args.mc_runs)

        all_probs.extend(mean_probs.cpu().numpy().tolist())
        all_var.extend(var_probs.cpu().numpy().tolist())
        all_entropy.extend(entropy.cpu().numpy().tolist())
        all_names.extend(batch["image_name"])
        all_paths.extend(batch["image_path"])

        if "label" in batch:
            all_labels.extend(batch["label"].cpu().numpy().tolist())

    pred_df = pd.DataFrame({
        "image_name": all_names,
        "image_path": all_paths,
        "pred_prob": all_probs,
        "mc_var": all_var,
        "predictive_entropy": all_entropy
    })

    has_label = len(all_labels) == len(all_probs)

    if has_label:
        y_true = np.array(all_labels).astype(int)
        y_prob = np.array(all_probs)

        pred_df["true_label"] = y_true
        pred_df["pred_label"] = (y_prob >= 0.5).astype(int)
        pred_df["correct"] = (pred_df["true_label"] == pred_df["pred_label"]).astype(int)

        metrics = compute_metrics(y_true, y_prob)
        ece, mce, calib_df = compute_calibration(y_true, y_prob, n_bins=10)
        metrics["ece"] = ece
        metrics["mce"] = mce

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        calib_df.to_csv(os.path.join(args.output_dir, "calibration_bins.csv"), index=False)
        plot_reliability_diagram(calib_df, os.path.join(args.output_dir, "reliability_diagram.png"))

        cm = confusion_matrix(y_true, pred_df["pred_label"].values)
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            os.path.join(args.output_dir, "confusion_matrix.csv")
        )

        correct_mask = pred_df["correct"].values == 1
        incorrect_mask = pred_df["correct"].values == 0

        uncertainty_summary = {
            "correct_mean_mc_var": float(pred_df.loc[correct_mask, "mc_var"].mean()) if correct_mask.sum() > 0 else None,
            "incorrect_mean_mc_var": float(pred_df.loc[incorrect_mask, "mc_var"].mean()) if incorrect_mask.sum() > 0 else None,
            "correct_mean_entropy": float(pred_df.loc[correct_mask, "predictive_entropy"].mean()) if correct_mask.sum() > 0 else None,
            "incorrect_mean_entropy": float(pred_df.loc[incorrect_mask, "predictive_entropy"].mean()) if incorrect_mask.sum() > 0 else None,
        }
        with open(os.path.join(args.output_dir, "uncertainty_summary.json"), "w") as f:
            json.dump(uncertainty_summary, f, indent=2)

    pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    gradcam = GradCAM(model, model.target_layer)
    gradcam_df = pred_df.copy()

    if has_label:
        gradcam_df["error_score"] = np.abs(gradcam_df["true_label"] - gradcam_df["pred_prob"])
        gradcam_df = gradcam_df.sort_values("error_score", ascending=False)
    else:
        gradcam_df = gradcam_df.sort_values("predictive_entropy", ascending=False)

    sample_df = gradcam_df.head(args.gradcam_samples)

    infer_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for _, row in sample_df.iterrows():
        img_path = row["image_path"]
        image_pil = Image.open(img_path).convert("L")
        x = infer_tf(image_pil).unsqueeze(0).to(device)

        cam = gradcam.generate(x)
        save_name = os.path.splitext(os.path.basename(img_path))[0] + "_gradcam.png"
        save_path = os.path.join(args.output_dir, "gradcam", save_name)
        overlay_cam_on_image(image_pil, cam, save_path)

    gradcam.remove()

    print(f"Predictions saved to: {os.path.join(args.output_dir, 'test_predictions.csv')}")
    if has_label:
        print(f"Metrics saved to: {os.path.join(args.output_dir, 'metrics.json')}")
        print(f"Reliability diagram saved to: {os.path.join(args.output_dir, 'reliability_diagram.png')}")
    print(f"Grad-CAM folder: {os.path.join(args.output_dir, 'gradcam')}")


if __name__ == "__main__":
    main()