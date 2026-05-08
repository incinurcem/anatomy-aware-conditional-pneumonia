import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from common_eval_utils import build_model, create_loader, load_model_weights, save_json


def enable_dropout(m):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


def predictive_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def variation_ratio(samples_binary: np.ndarray) -> np.ndarray:
    mode_prob = np.maximum(samples_binary.mean(axis=0), 1.0 - samples_binary.mean(axis=0))
    return 1.0 - mode_prob


def apply_tta_tensor(x: torch.Tensor, tta_id: int) -> torch.Tensor:
    if tta_id == 0:
        return x
    if tta_id == 1:
        return torch.flip(x, dims=[3])
    if tta_id == 2:
        return torch.flip(x, dims=[2])
    if tta_id == 3:
        return torch.rot90(x, 1, dims=[2, 3])
    if tta_id == 4:
        return torch.rot90(x, 3, dims=[2, 3])
    return x


def inverse_tta_prob(prob: torch.Tensor, tta_id: int) -> torch.Tensor:
    return prob


def area_under_risk_coverage_curve(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    order = np.argsort(uncertainty)
    correct_sorted = correct[order]
    risks = []
    coverages = []
    n = len(correct_sorted)
    for k in range(1, n + 1):
        cov = k / n
        risk = 1.0 - np.mean(correct_sorted[:k])
        coverages.append(cov)
        risks.append(risk)
    return float(np.trapz(risks, coverages))


def excess_aurc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    aurc = area_under_risk_coverage_curve(correct, uncertainty)
    err = 1.0 - np.mean(correct)
    optimal = err + (1.0 - err) * np.log(max(1.0 - err, 1e-12))
    return float(aurc - optimal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="image_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mean", type=float, default=0.485)
    parser.add_argument("--std", type=float, default=0.229)
    parser.add_argument("--mc_runs", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--tta_runs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader = create_loader(
        csv_path=args.test_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=args.mean,
        std=args.std,
    )
    model = build_model(args.model_name, in_channels=1, pretrained=False)
    model = load_model_weights(model, args.model_path, device)
    model.eval()
    model.apply(enable_dropout)

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.uint8)
            image_paths = batch["image_path"]

            mc_probs = []
            for _ in range(args.mc_runs):
                if args.use_tta:
                    tta_probs = []
                    for tta_id in range(args.tta_runs):
                        aug_images = apply_tta_tensor(images, tta_id)
                        logits = model(aug_images).squeeze(1)
                        probs = torch.sigmoid(logits)
                        probs = inverse_tta_prob(probs, tta_id)
                        tta_probs.append(probs.cpu().numpy())
                    probs_np = np.mean(np.stack(tta_probs, axis=0), axis=0)
                else:
                    logits = model(images).squeeze(1)
                    probs_np = torch.sigmoid(logits).cpu().numpy()
                mc_probs.append(probs_np)

            mc_probs = np.stack(mc_probs, axis=0)
            mean_prob = mc_probs.mean(axis=0)
            var_prob = mc_probs.var(axis=0)
            pred_entropy = predictive_entropy(mean_prob)
            expected_entropy = bernoulli_entropy(mc_probs).mean(axis=0)
            mutual_info = pred_entropy - expected_entropy
            sample_preds = (mc_probs >= args.threshold).astype(np.uint8)
            var_ratio = variation_ratio(sample_preds)
            y_pred = (mean_prob >= args.threshold).astype(np.uint8)
            correct = (y_pred == labels).astype(np.uint8)

            for i in range(len(labels)):
                rows.append({
                    "image_path": image_paths[i],
                    "y_true": int(labels[i]),
                    "mean_prob": float(mean_prob[i]),
                    "var_prob": float(var_prob[i]),
                    "predictive_entropy": float(pred_entropy[i]),
                    "expected_entropy": float(expected_entropy[i]),
                    "mutual_information": float(mutual_info[i]),
                    "variation_ratio": float(var_ratio[i]),
                    "y_pred": int(y_pred[i]),
                    "correct": int(correct[i]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "uncertainty_predictions.csv"), index=False)

    summary = {
        "mean_entropy_correct": float(df.loc[df["correct"] == 1, "predictive_entropy"].mean()),
        "mean_entropy_incorrect": float(df.loc[df["correct"] == 0, "predictive_entropy"].mean()),
        "mean_variance_correct": float(df.loc[df["correct"] == 1, "var_prob"].mean()),
        "mean_variance_incorrect": float(df.loc[df["correct"] == 0, "var_prob"].mean()),
        "mean_mutual_info_correct": float(df.loc[df["correct"] == 1, "mutual_information"].mean()),
        "mean_mutual_info_incorrect": float(df.loc[df["correct"] == 0, "mutual_information"].mean()),
        "mean_variation_ratio_correct": float(df.loc[df["correct"] == 1, "variation_ratio"].mean()),
        "mean_variation_ratio_incorrect": float(df.loc[df["correct"] == 0, "variation_ratio"].mean()),
        "aurc_entropy": area_under_risk_coverage_curve(df["correct"].values, df["predictive_entropy"].values),
        "eaurc_entropy": excess_aurc(df["correct"].values, df["predictive_entropy"].values),
        "aurc_variance": area_under_risk_coverage_curve(df["correct"].values, df["var_prob"].values),
        "eaurc_variance": excess_aurc(df["correct"].values, df["var_prob"].values),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "uncertainty_summary.csv"), index=False)
    save_json(os.path.join(args.output_dir, "uncertainty_summary.json"), summary)

    for col, fname, title in [
        ("predictive_entropy", "entropy_histogram.png", "Predictive Entropy"),
        ("var_prob", "variance_histogram.png", "Probability Variance"),
        ("mutual_information", "mutual_information_histogram.png", "Mutual Information"),
        ("variation_ratio", "variation_ratio_histogram.png", "Variation Ratio"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.hist(df.loc[df["correct"] == 1, col], bins=30, alpha=0.7, label="Correct")
        plt.hist(df.loc[df["correct"] == 0, col], bins=30, alpha=0.7, label="Incorrect")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname), dpi=300)
        plt.close()

    print("===== UNCERTAINTY EVALUATION COMPLETE =====")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
