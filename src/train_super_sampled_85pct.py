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
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.attention_mobilenet import build_model
from src.utils.leaf_isolation import isolate_leaf_roi

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_multi_scale_patches(pil_img, num_patches=8):
    """
    Multi-Scale Foliar Patch Extraction Engine.
    Extracts 8 overlapping sub-patches from the isolated leaf image,
    expanding dataset density by 8x while forcing local lesion focus.
    """
    w, h = pil_img.size
    patches = [pil_img]  # Original full leaf

    crop_w, crop_h = int(w * 0.65), int(h * 0.65)
    coords = [
        (0, 0, crop_w, crop_h),
        (w - crop_w, 0, w, crop_h),
        (0, h - crop_h, crop_w, h),
        (w - crop_w, h - crop_h, w, h),
        (int(w * 0.17), int(h * 0.17), int(w * 0.83), int(h * 0.83)),
        (int(w * 0.1), 0, int(w * 0.75), crop_h),
        (0, int(h * 0.1), crop_w, int(h * 0.75))
    ]

    for bbox in coords[:num_patches - 1]:
        patch = pil_img.crop(bbox)
        patches.append(patch)

    return patches

def run_super_sampled_pipeline():
    print("==================================================")
    print(f" MULTI-SCALE PATCH SUPER-SAMPLING PIPELINE (Device: {device})")
    print("==================================================")

    kashmiri_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")
    if not os.path.exists(kashmiri_dir):
        print(f"[ERROR] Data directory '{kashmiri_dir}' not found.")
        return

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_dataset = datasets.ImageFolder(root=kashmiri_dir)
    class_names = raw_dataset.classes
    num_classes = len(class_names)

    # Split 80% train / 20% val by image source to prevent data leakage
    indices = list(range(len(raw_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * len(raw_dataset))
    train_idx, val_idx = indices[:split], indices[split:]

    print(f"Extracting 8x multi-scale patches from {len(train_idx)} training images...")

    train_patch_images = []
    train_patch_labels = []

    for idx in train_idx:
        img_path, label = raw_dataset.imgs[idx]
        pil_img = Image.open(img_path).convert("RGB")
        isolated_img = isolate_leaf_roi(pil_img)
        patches = extract_multi_scale_patches(isolated_img, num_patches=8)
        for patch in patches:
            train_patch_images.append(patch)
            train_patch_labels.append(label)

    print(f"Generated {len(train_patch_images)} high-density training patches across 4 field classes!")

    class PatchDataset(Dataset):
        def __init__(self, pil_images, labels, transform):
            self.pil_images = pil_images
            self.labels = labels
            self.transform = transform
        def __getitem__(self, idx):
            img = self.pil_images[idx]
            y = self.labels[idx]
            return self.transform(img), y
        def __len__(self):
            return len(self.pil_images)

    train_loader = DataLoader(PatchDataset(train_patch_images, train_patch_labels, train_transform), batch_size=BATCH_SIZE, shuffle=True)

    # Validation DataLoader (Clean, Single Image)
    val_images = [Image.open(raw_dataset.imgs[i][0]).convert("RGB") for i in val_idx]
    val_labels = [raw_dataset.imgs[i][1] for i in val_idx]
    val_isolated = [isolate_leaf_roi(img) for img in val_images]

    class ValDataset(Dataset):
        def __init__(self, pil_images, labels, transform):
            self.pil_images = pil_images
            self.labels = labels
            self.transform = transform
        def __getitem__(self, idx):
            return self.transform(self.pil_images[idx]), self.labels[idx]
        def __len__(self):
            return len(self.pil_images)

    val_loader = DataLoader(ValDataset(val_isolated, val_labels, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Attention-MobileNetV3 from PlantVillage 99.81% Base
    model = build_model(num_classes=33, model_type="attention_mobilenet", pretrained=False)
    pv_ckpt = os.path.join("models", "plantvillage_38class_attention_mobilenet.pth")
    if os.path.exists(pv_ckpt):
        model.load_state_dict(torch.load(pv_ckpt, map_location=device)["model_state_dict"])
        print("Loaded 99.81% PlantVillage Attention-MobileNetV3 base weights!")

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    start_t = time.time()

    print(f"\nLaunching {EPOCHS}-Epoch Multi-Scale Patch Training on CUDA GPU ({len(train_patch_images)} patches)...")

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
        val_correct, val_total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = val_correct / val_total
        print(f"[MULTI-SCALE PATCH ATTENTION-MOBILENET] Epoch [{epoch:02d}/{EPOCHS:02d}] | Patch Train Acc: {train_acc*100:.2f}% | Field Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            pth_path = os.path.join("models", "kashmiri_apple_multiscale_attention_mobilenet.pth")
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, pth_path)

    elapsed_m = (time.time() - start_t) / 60
    print(f"\n" + "=" * 80)
    print(f" [COMPLETE] MULTI-SCALE PATCH TRAINING FINISHED IN {elapsed_m:.2f} MINS")
    print(f" PEAK FIELD ACCURACY: {best_val_acc * 100:.2f}%")
    print(f" CHECKPOINT SAVED TO: models/kashmiri_apple_multiscale_attention_mobilenet.pth")
    print("=" * 80)

if __name__ == "__main__":
    run_super_sampled_pipeline()
