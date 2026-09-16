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
import pandas as pd
import numpy as np

from src.models.attention_mobilenet import build_model
from src.utils.dataset import get_default_transforms, TransformedSubset

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
LR = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_mixed_38class_fine_tuning():
    print("==================================================")
    print(f" JOINT 38-CLASS MIXED-DOMAIN FINE-TUNING PIPELINE (Device: {device})")
    print("==================================================")

    pv_dir = os.path.join("data", "plantvillage")
    kashmiri_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")

    if not os.path.exists(pv_dir):
        print(f"[ERROR] Data directory '{pv_dir}' not found.")
        return

    # Advanced Multi-Crop & Field Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pv_dataset = datasets.ImageFolder(root=pv_dir)
    class_names = pv_dataset.classes
    num_classes = len(class_names)

    print(f"Detected {num_classes} PlantVillage crop disease classes.")

    train_size = int(0.8 * len(pv_dataset))
    val_size = len(pv_dataset) - train_size
    train_sub, val_sub = random_split(pv_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(TransformedSubset(train_sub, train_transform), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(TransformedSubset(val_sub, val_transform), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    models_to_train = [
        ("Attention-MobileNetV3 (Ours)", "plantvillage_38class_attention_mobilenet.pth", "attention_mobilenet"),
        ("MobileNetV3 (Baseline)", "plantvillage_38class_mobilenet_v3.pth", "mobilenet_v3"),
        ("ResNet50 (Benchmark)", "plantvillage_38class_resnet50.pth", "resnet50")
    ]

    for label, fname, mtype in models_to_train:
        print(f"\n>>> Fine-Tuning Mixed-Domain Model: {label} <<<")
        pth = os.path.join("models", fname)
        model = build_model(num_classes=num_classes, model_type=mtype, pretrained=False)
        
        if os.path.exists(pth):
            ckpt = torch.load(pth, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(f"Loaded initial checkpoint from: {pth}")

        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_val_acc = 0.0
        start_t = time.time()

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
            print(f"[{mtype.UPPER() if hasattr(mtype, 'UPPER') else mtype.upper()}] Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, pth)

        elapsed_m = (time.time() - start_t) / 60
        print(f"[{mtype.upper()}] Fine-Tuning Complete in {elapsed_m:.2f} mins. Peak Accuracy: {best_val_acc*100:.2f}% | Checkpoint: {pth}")

if __name__ == "__main__":
    run_mixed_38class_fine_tuning()
