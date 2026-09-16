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
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

from src.models.attention_mobilenet import build_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune_unseen_kashmiri():
    print("==================================================")
    print(f" DOMAIN ADAPTATION FINE-TUNING ON KASHMIRI APPLE DATASET (Device: {device})")
    print("==================================================")

    data_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found.")
        return

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_sub, val_sub = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Apply transforms
    class SubDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_loader = DataLoader(SubDataset(train_sub, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SubDataset(val_sub, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained Attention-MobileNetV3 model
    ckpt_path = os.path.join("models", "plantvillage_38class_attention_mobilenet.pth")
    model = build_model(num_classes=4, model_type="attention_mobilenet", pretrained=False)
    
    # Replace classifier head for 4 Kashmiri classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"Starting 5-Epoch Domain Adaptation on 4 Kashmiri Apple Classes ({train_size} train, {val_size} val)...")

    best_acc = 0.0
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

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_pth = os.path.join("models", "kashmiri_apple_attention_mobilenet.pth")
            torch.save({"model_state_dict": model.state_dict(), "class_names": full_dataset.classes}, save_pth)

    print("\n==================================================")
    print(f" [COMPLETE] Domain Adaptation Accuracy: {best_acc*100:.2f}%")
    print(f" Checkpoint Saved to: models/kashmiri_apple_attention_mobilenet.pth")
    print("==================================================")

if __name__ == "__main__":
    fine_tune_unseen_kashmiri()
