import os
import json
import copy
import math
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

try:
    from torch.amp import autocast, GradScaler
    AMP_NEW = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW = False


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class ROIDataset(Dataset):
    def __init__(self, csv_path, image_col="masked_roi_path", label_col="label", transform=None):
        self.df = pd.read_csv(csv_path).copy()
        if label_col not in self.df.columns:
            raise ValueError(f"{label_col} column not found in {csv_path}")
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        label = float(row[self.label_col])

        image = Image.open(img_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_path": img_path
        }


def get_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class PlainClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True):
        super().__init__()

        if backbone_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
        elif backbone_name == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
        elif backbone_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
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


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")

    return metrics


def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True, use_amp=False):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if AMP_NEW:
                with autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                with autocast(enabled=(use_amp and device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)

            if train:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_binary_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = float(epoch_loss)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_col", type=str, default="masked_roi_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ROIDataset(
        csv_path=args.train_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        transform=get_transforms(args.image_size, is_train=True)
    )
    val_ds = ROIDataset(
        csv_path=args.val_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        transform=get_transforms(args.image_size, is_train=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = PlainClassifier(backbone_name=args.backbone, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if (args.use_amp and device.type == "cuda") else None

    history = []
    best_state = None
    best_score = -1.0

    for epoch in range(args.epochs):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            use_amp=args.use_amp
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None,
            train=False,
            use_amp=args.use_amp
        )

        scheduler.step()

        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(row)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        current_score = val_metrics["auroc"] if not math.isnan(val_metrics["auroc"]) else val_metrics["f1"]
        if current_score > best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "train_history.csv"), index=False)

    config = vars(args)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print(f"Training history saved to: {os.path.join(args.output_dir, 'train_history.csv')}")


if __name__ == "__main__":
    main()