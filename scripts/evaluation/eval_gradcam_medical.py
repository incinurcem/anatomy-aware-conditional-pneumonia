import os
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common_eval_utils import build_model, load_model_weights


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        self.hook_handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        score = logits.sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-12)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        return cam, prob


def resolve_target_layer(model, model_name: str):
    name = model_name.lower()
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name == "densenet121":
        return model.features[-1]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def preprocess_image(path: str, image_size: int, mean: float, std: float):
    img = Image.open(path).convert("L").resize((image_size, image_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - mean) / std
    return img_np, tensor


def overlay_heatmap(image_gray: np.ndarray, cam: np.ndarray):
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_u8 = (image_gray * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
    return heatmap, overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="image_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mean", type=float, default=0.485)
    parser.add_argument("--std", type=float, default=0.229)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--positive_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "overlays"), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.test_csv).copy()
    if args.positive_only and args.label_col in df.columns:
        df = df[df[args.label_col] == 1].copy()
    if len(df) > args.num_samples:
        df = df.sample(args.num_samples, random_state=42)

    model = build_model(args.model_name, in_channels=1, pretrained=False)
    model = load_model_weights(model, args.model_path, device)
    model.eval()

    target_layer = resolve_target_layer(model, args.model_name)
    gradcam = GradCAM(model, target_layer)

    rows = []
    try:
        for idx, row in df.reset_index(drop=True).iterrows():
            image_path = row[args.image_col]
            label = int(row[args.label_col]) if args.label_col in row else -1
            image_gray, tensor = preprocess_image(image_path, args.image_size, args.mean, args.std)
            tensor = tensor.to(device)
            cam, prob = gradcam(tensor)
            heatmap, overlay = overlay_heatmap(image_gray, cam)

            basename = os.path.splitext(os.path.basename(image_path))[0]
            heatmap_path = os.path.join(args.output_dir, "heatmaps", f"{basename}_heatmap.png")
            overlay_path = os.path.join(args.output_dir, "overlays", f"{basename}_overlay.png")
            cv2.imwrite(heatmap_path, heatmap)
            cv2.imwrite(overlay_path, overlay)

            rows.append({
                "image_path": image_path,
                "label": label,
                "pred_prob": float(prob[0]),
                "heatmap_path": heatmap_path,
                "overlay_path": overlay_path,
            })
    finally:
        gradcam.remove_hooks()

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "gradcam_index.csv"), index=False)
    print("===== GRAD-CAM GENERATION COMPLETE =====")


if __name__ == "__main__":
    main()
