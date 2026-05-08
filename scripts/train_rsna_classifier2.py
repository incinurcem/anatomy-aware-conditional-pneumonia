# train_rsna_classifier.py
import os
import json
import math
import copy
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, roc_curve
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def log_print(message, log_file=None):
    print(message, flush=True)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


class RSNAClassifierDataset(Dataset):
    """
    input_mode:
      - plain      -> uses plain_path (full CXR resized 224)
      - roi        -> uses roi_path (cropped lung 224)
      - masked_roi -> uses roi_path + mask_crop_path with soft mask
                       (alpha=0 means hard mask, alpha=1 means no masking)
    """
    def __init__(self, csv_path, input_mode="plain", img_size=224,
                 is_train=False, soft_mask_alpha=0.1,
                 corner_mask_size=0.15):
        self.df = pd.read_csv(csv_path)
        self.input_mode = input_mode
        self.img_size = img_size
        self.is_train = is_train
        self.soft_mask_alpha = soft_mask_alpha
        self.corner_mask_size = corner_mask_size

        self.cond_features = [
            c for c in self.df.columns
            if c.startswith(("roi_", "glcm_", "mask_")) and not c.endswith("_path")
        ]

        if input_mode == "plain":
            self.path_col = "plain_path" if "plain_path" in self.df.columns else "image_path"
        elif input_mode in ("roi", "masked_roi"):
            self.path_col = "roi_path"
        else:
            raise ValueError(f"unknown input_mode: {input_mode}")

        required = [self.path_col, "label"]
        if input_mode == "masked_roi":
            required.append("mask_crop_path")
        self.df = self.df.dropna(subset=required).reset_index(drop=True)

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.df)

    def apply_corner_masking(self, img_tensor, size=0.15):
        _, h, w = img_tensor.shape
        mh, mw = int(h * size), int(w * size)
        img_tensor[:, 0:mh, 0:mw] = 0
        img_tensor[:, 0:mh, w - mw:w] = 0
        img_tensor[:, h - mh:h, 0:mw] = 0
        img_tensor[:, h - mh:h, w - mw:w] = 0
        return img_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row.get("image_id", idx))
        path = str(row[self.path_col])
        label = int(row["label"])

        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)

        if self.input_mode == "plain" and self.corner_mask_size > 0:
            img_tensor = self.apply_corner_masking(img_tensor, size=self.corner_mask_size)

        if self.input_mode == "masked_roi":
            mask_path = str(row["mask_crop_path"])
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)  # 1xHxW, 0..1
            alpha = float(self.soft_mask_alpha)
            img_tensor = img_tensor * mask_tensor + img_tensor * (1.0 - mask_tensor) * alpha

        img_tensor = self.normalize(img_tensor)

        if len(self.cond_features) > 0:
            cond_vec = torch.tensor(
                row[self.cond_features].values.astype(np.float32)
            )
        else:
            cond_vec = torch.zeros(0, dtype=torch.float32)

        return {
            "image": img_tensor,
            "cond": cond_vec,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_id": image_id,
        }


class ConditionalResNet50(nn.Module):
    def __init__(self, cond_dim, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        self.num_ftrs = base.fc.in_features
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.cond_projection = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs + 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, img, cond):
        f = self.backbone(img).view(img.size(0), -1)
        c = self.cond_projection(cond)
        return self.classifier(torch.cat([f, c], dim=1))


def build_model(model_name="resnet50", pretrained=True, dropout=0.3,
                is_conditional=False, cond_dim=0):
    if is_conditional:
        assert cond_dim > 0, "is_conditional=True requires cond_dim > 0"
        return ConditionalResNet50(cond_dim=cond_dim, dropout=dropout,
                                   pretrained=pretrained)

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
        return m
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=weights)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
        return m
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
        return m
    raise ValueError(f"Unsupported model: {model_name}")


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if not math.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not math.isnan(pr_auc) else None,
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(bal),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def youden_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def run_one_epoch(model, loader, criterion, optimizer, device,
                  scaler=None, train=False, is_conditional=False):
    model.train() if train else model.eval()
    total_loss, all_probs, all_labels, all_ids = 0.0, [], [], []
    pbar = tqdm(loader, desc="Train" if train else "Eval", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        conds = batch["cond"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)
        image_ids = batch["image_id"]

        with torch.set_grad_enabled(train):
            if train and scaler is not None and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits = model(images, conds) if is_conditional else model(images)
                    loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, conds) if is_conditional else model(images)
                loss = criterion(logits, labels)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        total_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
        all_ids.extend(list(image_ids))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = float(epoch_loss)

    pred_df = pd.DataFrame({
        "image_id": all_ids,
        "y_true": np.asarray(all_labels).astype(int),
        "y_prob": np.asarray(all_probs),
        "y_pred": (np.asarray(all_probs) >= 0.5).astype(int),
    })
    return metrics, pred_df


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_auc, history, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_auc": best_val_auc,
        "history": history,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--input_mode", default="plain",
                        choices=["plain", "roi", "masked_roi"])
    parser.add_argument("--model_name", default="resnet50",
                        choices=["resnet50", "densenet121", "efficientnet_b0"])
    parser.add_argument("--is_conditional", action="store_true",
                        help="If set, use ConditionalResNet50 with cond features.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--soft_mask_alpha", type=float, default=0.1,
                        help="masked_roi background visibility (0=hard mask, 1=no mask)")
    parser.add_argument("--corner_mask_size", type=float, default=0.15,
                        help="corner suppression size for plain mode (0 to disable)")
    parser.add_argument("--use_weighted_sampler", action="store_true")

    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--save_every_epoch", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Training started at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"[INFO] Device: {device}", log_file)
    log_print(f"[INFO] Args: {vars(args)}", log_file)

    train_ds = RSNAClassifierDataset(args.train_csv, args.input_mode, args.img_size,
                                     is_train=True,
                                     soft_mask_alpha=args.soft_mask_alpha,
                                     corner_mask_size=args.corner_mask_size)
    val_ds = RSNAClassifierDataset(args.val_csv, args.input_mode, args.img_size,
                                   is_train=False,
                                   soft_mask_alpha=args.soft_mask_alpha,
                                   corner_mask_size=args.corner_mask_size)
    test_ds = RSNAClassifierDataset(args.test_csv, args.input_mode, args.img_size,
                                    is_train=False,
                                    soft_mask_alpha=args.soft_mask_alpha,
                                    corner_mask_size=args.corner_mask_size)

    log_print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}",
              log_file)

    cond_dim = len(train_ds.cond_features) if args.is_conditional else 0
    log_print(f"[INFO] is_conditional={args.is_conditional} | cond_dim={cond_dim}",
              log_file)
    if args.is_conditional and cond_dim == 0:
        raise ValueError("is_conditional set but no cond features in CSV.")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if args.use_weighted_sampler:
        labels = train_ds.df["label"].astype(int).values
        class_counts = np.bincount(labels, minlength=2)
        class_w = 1.0 / np.maximum(class_counts, 1)
        sample_w = class_w[labels]
        sampler = WeightedRandomSampler(weights=sample_w.tolist(),
                                        num_samples=len(sample_w),
                                        replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = build_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        dropout=args.dropout,
        is_conditional=args.is_conditional,
        cond_dim=cond_dim,
    ).to(device)

    pos = int((train_ds.df["label"] == 1).sum())
    neg = int((train_ds.df["label"] == 0).sum())
    pos_weight_value = neg / max(pos, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    log_print(f"[INFO] pos={pos} neg={neg} pos_weight={pos_weight_value:.4f}", log_file)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(args.amp and device.type == "cuda"))

    best_val_auc = -1.0
    best_state = None
    best_val_pred = None
    history = []
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, train=True, is_conditional=args.is_conditional
        )
        val_metrics, val_pred = run_one_epoch(
            model, val_loader, criterion, None, device,
            scaler=None, train=False, is_conditional=args.is_conditional
        )

        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        scheduler.step(val_auc)
        cur_lr = float(optimizer.param_groups[0]["lr"])

        is_best = val_auc > best_val_auc
        history.append({
            "epoch": epoch, "lr": cur_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"),
                                     index=False)

        if is_best:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_val_pred = val_pred.copy()
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pth"))
            save_checkpoint(os.path.join(args.output_dir, "best_checkpoint.pth"),
                            model, optimizer, scheduler, epoch,
                            best_val_auc, history, args)
            best_val_pred.to_csv(
                os.path.join(args.output_dir, "best_val_predictions.csv"),
                index=False
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint(os.path.join(args.output_dir, "last_checkpoint.pth"),
                        model, optimizer, scheduler, epoch,
                        best_val_auc, history, args)
        if args.save_every_epoch:
            save_checkpoint(os.path.join(args.output_dir,
                                         f"checkpoint_epoch_{epoch:03d}.pth"),
                            model, optimizer, scheduler, epoch,
                            best_val_auc, history, args)

        log_print(
            f"[EP {epoch}/{args.epochs}] "
            f"tr_loss={train_metrics['loss']:.4f} "
            f"tr_auc={(train_metrics['roc_auc'] or 0):.4f} | "
            f"vl_loss={val_metrics['loss']:.4f} "
            f"vl_auc={val_auc:.4f} vl_f1={val_metrics['f1']:.4f} | "
            f"lr={cur_lr:.2e} time={time.time()-t0:.1f}s "
            f"best={'Y' if is_best else 'N'} no_imp={epochs_no_improve}",
            log_file,
        )

        if epochs_no_improve >= args.early_stop_patience:
            log_print(f"[EARLY STOP] no improvement for "
                      f"{args.early_stop_patience} epochs.", log_file)
            break

    if best_state is None:
        best_state = model.state_dict()
        best_val_pred = val_pred
    model.load_state_dict(best_state)

    best_thr = youden_threshold(best_val_pred["y_true"].values,
                                best_val_pred["y_prob"].values)
    log_print(f"[INFO] Youden threshold on val = {best_thr:.4f}", log_file)

    _, test_pred = run_one_epoch(
        model, test_loader, criterion, None, device,
        scaler=None, train=False, is_conditional=args.is_conditional
    )
    test_metrics_05 = compute_metrics(test_pred["y_true"].values,
                                      test_pred["y_prob"].values, threshold=0.5)
    test_metrics_youden = compute_metrics(test_pred["y_true"].values,
                                          test_pred["y_prob"].values,
                                          threshold=best_thr)

    test_pred["y_pred_youden"] = (test_pred["y_prob"].values >= best_thr).astype(int)
    test_pred.to_csv(os.path.join(args.output_dir, "test_predictions.csv"),
                     index=False)

    save_json({
        "test_metrics_threshold_0.5": test_metrics_05,
        "test_metrics_threshold_youden": test_metrics_youden,
        "youden_threshold_from_val": best_thr,
        "best_val_auc": best_val_auc,
    }, os.path.join(args.output_dir, "test_metrics.json"))

    log_print("FINAL TEST (thr=0.5)   -> " + json.dumps(test_metrics_05,
                                                         ensure_ascii=False),
              log_file)
    log_print(f"FINAL TEST (thr={best_thr:.3f}) -> " +
              json.dumps(test_metrics_youden, ensure_ascii=False), log_file)


if __name__ == "__main__":
    main()