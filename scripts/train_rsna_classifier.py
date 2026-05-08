
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
from tqdm import tqdm #
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_print(message, log_file=None):
    print(message, flush=True)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


class RSNAClassifierDataset(Dataset):
    def __init__(self, csv_path, input_mode="plain", img_size=224, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.input_mode = input_mode
        self.img_size = img_size
        self.is_train = is_train # Train/Test ayrımı için saklıyoruz
        
        # Condition kolonlarını otomatik seç (27 özellik)
# Yeni hali: İçinde "path" geçen sütunları dahil etmiyoruz!
        self.cond_features = [c for c in self.df.columns if c.startswith(('roi_', 'glcm_', 'mask_')) and not c.endswith('_path')]
        
        # Input moduna göre sütun seçimi
        if input_mode == "plain": 
            self.path_col = "image_path"
        elif input_mode == "roi": 
            self.path_col = "roi_path"
        elif input_mode == "masked_roi": 
            self.path_col = "masked_roi_path"
        
        # Eksik veri temizliği
        self.df = self.df.dropna(subset=[self.path_col, "label"]).reset_index(drop=True)

        # --- GÜNCELLEME: Eğitim ve Test Transformlarını Ayırıyoruz ---
        if is_train:
            # Eğitimde modelin "sağlam" öğrenmesi için küçük değişimler ekliyoruz
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5), # Göğüs kafesi simetrik olduğu için güvenli
                transforms.RandomRotation(degrees=7),   # Küçük duruş bozukluklarını simüle eder
                transforms.ToTensor(), # Pikselleri [0, 1] arasına çeker
            ])
        else:
            # Test ve Validasyon setinde görüntüye dokunmuyoruz, sadece hazırlıyoruz
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        
        # Normalizasyonu ayrı tutuyoruz çünkü __getitem__ içinde 
        # masking işlemlerinden SONRA elle uygulayacağız.
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    def __len__(self): return len(self.df)

    def apply_corner_masking(self, img_tensor, size=0.15):
        """Dört köşeyi siyah kutuyla kapatır (Shortcut Suppression)"""
        _, h, w = img_tensor.shape
        mh, mw = int(h * size), int(w * size)
        img_tensor[:, 0:mh, 0:mw] = 0        # Sol Üst
        img_tensor[:, 0:mh, w-mw:w] = 0      # Sağ Üst (L harfi)
        img_tensor[:, h-mh:h, 0:mw] = 0      # Sol Alt
        img_tensor[:, h-mh:h, w-mw:w] = 0    # Sağ Alt
        return img_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row.get("image_id", idx))
        path = str(row[self.path_col])
        label = int(row["label"])

        # 1. Görüntüyü yükle
        img = Image.open(path).convert("RGB")
        
        # 2. Temel Transformları uygula (Resize, Augmentation ve ToTensor)
        # Not: ToTensor() pikselleri [0, 1] arasına çeker.
        img_tensor = self.transform(img) 

        # 3. GÜNCELLEME 1: Tüm modeller için köşeleri kapat (Shortcut Suppression)
        # Modelin 'L' harfine bakmasını engellemek için pikseller hala [0,1] iken yapıyoruz.
        img_tensor = self.apply_corner_masking(img_tensor, size=0.15)

        # 4. GÜNCELLEME 2: Masked ROI için 'Soft Masking'
        if self.input_mode == "masked_roi" and "mask_path" in row:
            mask_path = str(row["mask_path"])
            if os.path.exists(mask_path):
                # Maskeyi yükle ve görüntüyle aynı boyuta getir
                mask = Image.open(mask_path).convert("L")
                mask = transforms.Resize((self.img_size, self.img_size))(mask)
                mask_tensor = transforms.ToTensor()(mask)
                
                # Soft Masking: Dışarıyı tamamen siyah (0) yapmak yerine %10 (0.1) görünür bırak
                # Bu, modelin akciğer sınırlarını ve anatomik konumunu anlamasını sağlar.
                img_tensor = img_tensor * mask_tensor + (img_tensor * (1 - mask_tensor) * 0.1)

        # 5. GÜNCELLEME: Normalizasyonu en son uygula
        # Maskelemeler bittikten sonra mean/std değerlerini uyguluyoruz.
        img_tensor = self.normalize(img_tensor)
        
        # 6. Condition vektörünü al (27 özellik)
        cond_vec = torch.tensor(row[self.cond_features].values.astype(np.float32))

        return {
            "image": img_tensor,
            "cond": cond_vec,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_id": image_id
        }

class ConditionalResNet50(nn.Module):
    def __init__(self, model_name="resnet50", cond_dim=26, dropout=0.3):
        super().__init__()
        # Backbone seçimi
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_ftrs = base_model.fc.in_features
        self.backbone = nn.Sequential(*list(base_model.children())[:-1]) # Son FC'yi at
        
        # GÜNCELLEME 3: Condition Vector'ün sesini yükselt (Projection)
        self.cond_projection = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512), # Vektörü 512 boyuta taşıyoruz
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final Classifier (2048 Image + 512 Cond = 2560)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_ftrs + 512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, img, cond):
        img_feats = self.backbone(img).view(img.size(0), -1) # 2048
        cond_feats = self.cond_projection(cond)             # 512
        
        combined = torch.cat((img_feats, cond_feats), dim=1) # 2560
        return self.classifier(combined)


def build_model(model_name="resnet50", pretrained=True, dropout=0.3, is_conditional=False, cond_dim=26):
    if is_conditional:
        # Eğer condition vektörü kullanıyorsak bizim yeni sınıfımızı oluştur
        return ConditionalResNet50(cond_dim=cond_dim, dropout=dropout)
    
    # Standart (sadece görüntü) modeller için eski mantık devam eder:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
        return model
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    return model


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if not math.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not math.isnan(pr_auc) else None,
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(bal_acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, train=False):
    if train: 
        model.train()
    else: 
        model.eval()

    total_loss, all_probs, all_labels, all_ids = 0.0, [], [], []

    pbar = tqdm(loader, desc="Training" if train else "Evaluating", leave=False)

    for batch in pbar:
        # Verileri cihaza taşı (non_blocking hızı artırır)
        images = batch["image"].to(device, non_blocking=True)
        conds = batch["cond"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)
        image_ids = batch["image_id"]

        with torch.set_grad_enabled(train):
            # AMP (Mixed Precision) Desteği - CUDA varsa aktif olur
            if train and scaler is not None and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    # --- MODEL TİPİNE GÖRE ÇAĞRI ---
                    if isinstance(model, ConditionalResNet50):
                        logits = model(images, conds)
                    else:
                        logits = model(images)
                    
                    loss = criterion(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # --- MODEL TİPİNE GÖRE ÇAĞRI ---
                if isinstance(model, ConditionalResNet50):
                    logits = model(images, conds)
                else:
                    logits = model(images)
                
                loss = criterion(logits, labels)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        # Tahminleri ve etiketleri CPU'ya çek

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        total_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
        all_ids.extend(list(image_ids))

    # Ortalama kaybı ve metrikleri hesapla
    epoch_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = float(epoch_loss)

    # Değerlendirme sırasında kullanılacak tahmin DataFrame'i
    pred_df = pd.DataFrame({
        "image_id": all_ids,
        "y_true": np.asarray(all_labels).astype(int),
        "y_prob": np.asarray(all_probs),
        "y_pred": (np.asarray(all_probs) >= 0.5).astype(int)
    })
    
    return metrics, pred_df

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_auc, history, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_auc": best_val_auc,
        "history": history,
        "args": vars(args)
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--input_mode", type=str, default="plain",
                        choices=["plain", "roi", "masked_roi"])
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "densenet121", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epoch", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "train_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"[INFO] Device: {device}", log_file=log_file)
    log_print(f"[INFO] Args: {vars(args)}", log_file=log_file)

    train_ds = RSNAClassifierDataset(args.train_csv, args.input_mode, args.img_size, is_train=True)
    val_ds = RSNAClassifierDataset(args.val_csv, args.input_mode, args.img_size, is_train=False)
    test_ds = RSNAClassifierDataset(args.test_csv, args.input_mode, args.img_size, is_train=False)

    log_print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", log_file=log_file)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda")
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # main() içinde modelin kurulduğu yer:
    # is_conditional bilgisini input_mode veya csv isminden anlayabilirsin.
    is_cond = "condition" in args.train_csv.lower() # Basit bir kontrol
    cond_dim = len(train_ds.cond_features) if is_cond else 0
    log_print(f"[INFO] Condition dimension: {cond_dim}")

    model = build_model(
      model_name=args.model_name, 
      pretrained=args.pretrained, 
      dropout=args.dropout,
      is_conditional=is_cond,
      cond_dim=cond_dim # Dinamik boyut gönderiliyor
    ).to(device)

    train_df = pd.read_csv(args.train_csv)
    pos_count = int((train_df["label"] == 1).sum())
    neg_count = int((train_df["label"] == 0).sum())
    pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    log_print(
        f"[INFO] Class balance | pos_count={pos_count} | neg_count={neg_count} | pos_weight={pos_weight_value:.6f}",
        log_file=log_file
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_val_auc = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics, train_pred = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True
        )

        val_metrics, val_pred = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None,
            train=False
        )

        val_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0
        scheduler.step(val_auc)
        current_lr = float(optimizer.param_groups[0]["lr"])

        is_best = val_auc > best_val_auc

        row = {
            "epoch": epoch,
            "lr": current_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)

        epoch_time = time.time() - epoch_start

        if is_best:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

            torch.save(best_state, os.path.join(args.output_dir, "best_model.pth"))
            save_checkpoint(
                path=os.path.join(args.output_dir, "best_checkpoint.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_auc=best_val_auc,
                history=history,
                args=args
            )

            train_pred.to_csv(os.path.join(args.output_dir, "best_train_predictions.csv"), index=False)
            val_pred.to_csv(os.path.join(args.output_dir, "best_val_predictions.csv"), index=False)

        if args.save_every_epoch:
            save_checkpoint(
                path=os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:03d}.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_auc=best_val_auc,
                history=history,
                args=args
            )

        save_checkpoint(
            path=os.path.join(args.output_dir, "last_checkpoint.pth"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_auc=best_val_auc,
            history=history,
            args=args
        )

        epoch_msg = (
            f"[EPOCH {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"train_auc={(train_metrics['roc_auc'] if train_metrics['roc_auc'] is not None else 0.0):.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_auc={(val_metrics['roc_auc'] if val_metrics['roc_auc'] is not None else 0.0):.6f} | "
            f"val_f1={val_metrics['f1']:.6f} | "
            f"lr={current_lr:.8f} | "
            f"time={epoch_time:.2f}s | "
            f"best={'yes' if is_best else 'no'}"
        )
        log_print(epoch_msg, log_file=log_file)

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)

    test_metrics, test_pred = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        scaler=None,
        train=False
    )

    test_pred.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))

    log_print("FINAL TEST METRICS -> " + json.dumps(test_metrics, ensure_ascii=False), log_file=log_file)


if __name__ == "__main__":
    main()