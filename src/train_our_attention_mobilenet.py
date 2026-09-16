import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.attention_mobilenet import build_model
from src.utils.leaf_isolation import isolate_leaf_roi
from src.utils.dataset import TransformedSubset

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LR = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test-Time Augmentation (TTA) Helper Function for Attention-MobileNetV3
def predict_with_tta(model, image_tensor):
    """
    5-Crop & Flip Test-Time Augmentation (TTA) Inference Engine.
    Averages predictions across 5 transforms (Original, H-Flip, V-Flip, Rot15, Rot-15)
    to smooth out smartphone glare, shadows, and angle noise.
    """
    model.eval()
    tensors = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),  # Horizontal Flip
        torch.flip(image_tensor, dims=[2]),  # Vertical Flip
        transforms.functional.rotate(image_tensor, 15),
        transforms.functional.rotate(image_tensor, -15)
    ]
    all_probs = []
    with torch.no_grad():
        for t in tensors:
            outputs = model(t.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs

def train_and_test_our_model():
    print("==================================================")
    print(f" HIGH-PRECISION FIELD-ROBUST TRAINING FOR ATTENTION-MOBILENET (Device: {device})")
    print("==================================================")

    pv_dir = os.path.join("data", "plantvillage")
    kashmiri_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")

    if not os.path.exists(pv_dir):
        print(f"[ERROR] Data directory '{pv_dir}' not found.")
        return

    # Advanced Augmentations (Cutout + Jitter + Resized Crops)
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pv_dataset = datasets.ImageFolder(root=pv_dir)
    class_names = pv_dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(pv_dataset))
    val_size = len(pv_dataset) - train_size
    train_sub, val_sub = random_split(pv_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(TransformedSubset(train_sub, train_transform), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(TransformedSubset(val_sub, val_transform), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    print(f"Dataset split: {train_size} training images, {val_size} validation images across {num_classes} classes.")
    print("Initializing Attention-MobileNetV3 (CBAM) Model...")

    model = build_model(num_classes=num_classes, model_type="attention_mobilenet", pretrained=False)
    
    pth_path = os.path.join("models", "plantvillage_38class_attention_mobilenet.pth")
    if os.path.exists(pth_path):
        ckpt = torch.load(pth_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded existing weights from: {pth_path}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    start_t = time.time()

    print(f"\nLaunching {EPOCHS}-Epoch High-Precision Training on CUDA GPU...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        print(f"[ATTENTION_MOBILENET] Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, pth_path)

    elapsed_m = (time.time() - start_t) / 60
    print(f"\n[DONE] Attention-MobileNetV3 Training Complete in {elapsed_m:.2f} mins. Peak Accuracy: {best_val_acc*100:.2f}%")
    print(f"[SAVED] Updated model checkpoint saved to: {pth_path}")

    # =========================================================
    # TESTING ON UNSEEN KASHMIRI FIELD DATASET WITH 5-CROP TTA
    # =========================================================
    if os.path.exists(kashmiri_dir):
        print("\n==================================================")
        print(" TESTING ATTENTION-MOBILENET ON UNSEEN FIELD DATASET (WITH 5-CROP TTA)")
        print("==================================================")

        raw_kashmiri = datasets.ImageFolder(root=kashmiri_dir)
        CLASS_MAP = {
            "APPLE ROT LEAVES": "Apple___Black_rot",
            "HEALTHY LEAVES": "Apple___healthy",
            "LEAF BLOTCH": "Apple___Cedar_apple_rust",
            "SCAB LEAVES": "Apple___Apple_scab"
        }

        folder_to_model_idx = {}
        for f_idx, folder_name in enumerate(raw_kashmiri.classes):
            target_name = CLASS_MAP[folder_name]
            folder_to_model_idx[f_idx] = class_names.index(target_name)

        model.load_state_dict(torch.load(pth_path, map_location=device)["model_state_dict"])
        model.eval()

        y_true, y_pred_standard, y_pred_tta = [], [], []

        for img_path, label in raw_kashmiri.imgs:
            pil_img = Image.open(img_path).convert("RGB")
            # Apply Foliar Leaf Isolation
            isolated_img = isolate_leaf_roi(pil_img)
            tensor_img = val_transform(isolated_img).unsqueeze(0)

            # Standard Single Pass
            with torch.no_grad():
                out_std = model(tensor_img.to(device))
                pred_std = out_std.argmax(dim=1).item()

            # 5-Crop TTA Pass
            probs_tta = predict_with_tta(model, tensor_img)
            pred_tta = probs_tta.argmax(dim=1).item()

            target_idx = folder_to_model_idx[label]
            y_true.append(target_idx)
            y_pred_standard.append(pred_std)
            y_pred_tta.append(pred_tta)

        y_true = np.array(y_true)
        acc_std = accuracy_score(y_true, y_pred_standard)
        acc_tta = accuracy_score(y_true, y_pred_tta)
        f1_tta = f1_score(y_true, y_pred_tta, average="weighted", zero_division=0)

        print("\n" + "=" * 80)
        print("          UNSEEN FIELD DATASET EVALUATION RESULTS (OUR MODEL)")
        print("=" * 80)
        print(f"  - Standard Single-Pass Accuracy  : {acc_std * 100:.2f}%")
        print(f"  - 5-Crop Test-Time Aug (TTA) Acc : {acc_tta * 100:.2f}%")
        print(f"  - TTA Weighted F1-Score         : {f1_tta:.4f}")
        print("=" * 80)

        # Save Final Plot
        cm = confusion_matrix(y_true, y_pred_tta)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
        plt.title(f"Attention-MobileNetV3 (5-Crop TTA)\nField Accuracy: {acc_tta*100:.2f}% | F1: {f1_tta:.4f}")
        plt.xlabel("Predicted Class Index")
        plt.ylabel("True Class Index")
        plt.tight_layout()
        plt.savefig(os.path.join("notebooks", "our_model_tta_field_cm.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    train_and_test_our_model()
