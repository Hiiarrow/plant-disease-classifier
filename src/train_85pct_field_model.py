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
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.attention_mobilenet import build_model
from src.utils.leaf_isolation import isolate_leaf_roi

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_with_5crop_tta(model, image_tensor):
    model.eval()
    tensors = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),  # H-Flip
        torch.flip(image_tensor, dims=[2]),  # V-Flip
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

def run_85pct_field_training():
    print("==================================================")
    print(f" TARGETING 85%+ FIELD ACCURACY FOR ATTENTION-MOBILENET (Device: {device})")
    print("==================================================")

    kashmiri_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")
    if not os.path.exists(kashmiri_dir):
        print(f"[ERROR] Data directory '{kashmiri_dir}' not found.")
        return

    # Heavy Field Augmentation Pipeline
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(35),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.25))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_dataset = datasets.ImageFolder(root=kashmiri_dir)
    num_classes = len(raw_dataset.classes)
    class_names = raw_dataset.classes

    train_size = int(0.8 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    train_sub, val_sub = random_split(raw_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    class PreprocessedFieldDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            path, y = self.subset.dataset.imgs[self.subset.indices[idx]]
            pil_img = Image.open(path).convert("RGB")
            isolated_img = isolate_leaf_roi(pil_img)
            tensor_img = self.transform(isolated_img)
            return tensor_img, y
        def __len__(self):
            return len(self.subset)

    print(f"Pre-processing foliar leaf isolation on {train_size} training & {val_size} validation field images...")

    train_loader = DataLoader(PreprocessedFieldDataset(train_sub, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PreprocessedFieldDataset(val_sub, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_classes=num_classes, model_type="attention_mobilenet", pretrained=True)
    model = model.to(device)

    # Layer-wise Discriminative Learning Rates
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n and "cbam" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n or "cbam" in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-3},
        {'params': head_params, 'lr': 1e-3, 'weight_decay': 1e-3}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_val_acc = 0.0
    best_tta_acc = 0.0
    start_t = time.time()

    print(f"\nStarting {EPOCHS}-Epoch High-Accuracy Discriminative Fine-Tuning on CUDA GPU...")

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

        # Validation Pass (Standard + TTA)
        model.eval()
        val_correct, val_total = 0, 0
        y_true, y_pred_tta = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                for img, lbl in zip(images, labels):
                    probs_tta = predict_with_5crop_tta(model, img.unsqueeze(0))
                    pred = probs_tta.argmax(dim=1).item()
                    if pred == lbl.item():
                        val_correct += 1
                    val_total += 1
                    y_true.append(lbl.item())
                    y_pred_tta.append(pred)

        val_acc_tta = val_correct / val_total
        print(f"[ATTENTION_MOBILENET] Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | 5-Crop TTA Field Acc: {val_acc_tta*100:.2f}%")

        if val_acc_tta > best_tta_acc:
            best_tta_acc = val_acc_tta
            pth_path = os.path.join("models", "kashmiri_apple_85pct_attention_mobilenet.pth")
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, pth_path)

    elapsed_m = (time.time() - start_t) / 60
    print(f"\n" + "=" * 80)
    print(f" [COMPLETE] ATTENTION-MOBILENET FIELD FINE-TUNING FINISHED IN {elapsed_m:.2f} MINS")
    print(f" PEAK 5-CROP TTA FIELD ACCURACY: {best_tta_acc * 100:.2f}%")
    print(f" CHECKPOINT SAVED TO: models/kashmiri_apple_85pct_attention_mobilenet.pth")
    print("=" * 80)

if __name__ == "__main__":
    run_85pct_field_training()
