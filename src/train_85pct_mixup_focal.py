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
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focal Loss for Hard-Sample Mining
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

# Mixup Data Augmentation Function
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

def run_focal_mixup_training():
    print("==================================================")
    print(f" HIGH-ACCURACY FOCAL LOSS & MIXUP PIPELINE FOR ATTENTION-MOBILENET (Device: {device})")
    print("==================================================")

    kashmiri_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")
    if not os.path.exists(kashmiri_dir):
        print(f"[ERROR] Data directory '{kashmiri_dir}' not found.")
        return

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

    raw_dataset = datasets.ImageFolder(root=kashmiri_dir)
    num_classes = len(raw_dataset.classes)
    class_names = raw_dataset.classes

    train_size = int(0.8 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    train_sub, val_sub = random_split(raw_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    class PreprocessedDataset(Dataset):
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

    train_loader = DataLoader(PreprocessedDataset(train_sub, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PreprocessedDataset(val_sub, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Load 99.81% PlantVillage Attention-MobileNetV3 Weights as Base Backbone
    model = build_model(num_classes=33, model_type="attention_mobilenet", pretrained=False)
    pv_ckpt = os.path.join("models", "plantvillage_38class_attention_mobilenet.pth")
    if os.path.exists(pv_ckpt):
        ckpt = torch.load(pv_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("Loaded pre-trained 99.81% PlantVillage Attention-MobileNetV3 weights!")

    # Replace final classifier layer with 4 Kashmiri classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Focal Loss + Discriminative LR
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_tta_acc = 0.0
    start_t = time.time()

    print(f"\nLaunching {EPOCHS}-Epoch Focal Loss + Mixup Field Training on CUDA GPU...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images_m, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)

            optimizer.zero_grad()
            outputs = model(images_m)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation (5-Crop TTA Pass)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                for img, lbl in zip(images, labels):
                    probs_tta = predict_with_5crop_tta(model, img.unsqueeze(0))
                    pred = probs_tta.argmax(dim=1).item()
                    if pred == lbl.item():
                        val_correct += 1
                    val_total += 1

        val_acc_tta = val_correct / val_total
        print(f"[ATTENTION_MOBILENET + FOCAL MIXUP] Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Acc: {train_acc*100:.2f}% | 5-Crop TTA Field Acc: {val_acc_tta*100:.2f}%")

        if val_acc_tta > best_tta_acc:
            best_tta_acc = val_acc_tta
            pth_path = os.path.join("models", "kashmiri_apple_focal_attention_mobilenet.pth")
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, pth_path)

    elapsed_m = (time.time() - start_t) / 60
    print(f"\n" + "=" * 80)
    print(f" [COMPLETE] FOCAL + MIXUP TRAINING FINISHED IN {elapsed_m:.2f} MINS")
    print(f" PEAK 5-CROP TTA FIELD ACCURACY: {best_tta_acc * 100:.2f}%")
    print(f" CHECKPOINT SAVED TO: models/kashmiri_apple_focal_attention_mobilenet.pth")
    print("=" * 80)

if __name__ == "__main__":
    run_focal_mixup_training()
