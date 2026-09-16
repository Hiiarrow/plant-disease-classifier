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
    w, h = pil_img.size
    patches = [pil_img]

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

def predict_with_5crop_tta(model, image_tensor):
    model.eval()
    tensors = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),
        torch.flip(image_tensor, dims=[2]),
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

def run_3model_field_pipeline():
    print("==================================================")
    print(f" 3-MODEL FIELD MULTI-SCALE PATCH & TTA BENCHMARK (Device: {device})")
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

    indices = list(range(len(raw_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * len(raw_dataset))
    train_idx, val_idx = indices[:split], indices[split:]

    print(f"Extracting 8x multi-scale patches from {len(train_idx)} field training images...")

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
            return self.transform(self.pil_images[idx]), self.labels[idx]
        def __len__(self):
            return len(self.pil_images)

    train_loader = DataLoader(PatchDataset(train_patch_images, train_patch_labels, train_transform), batch_size=BATCH_SIZE, shuffle=True)

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

    models_to_train = [
        ("MobileNetV3 (Baseline)", "plantvillage_38class_mobilenet_v3.pth", "mobilenet_v3"),
        ("Attention-MobileNetV3 (Ours)", "plantvillage_38class_attention_mobilenet.pth", "attention_mobilenet"),
        ("ResNet50 (Benchmark)", "plantvillage_38class_resnet50.pth", "resnet50")
    ]

    summary_results = []

    for label, base_fname, mtype in models_to_train:
        print(f"\n==================================================")
        print(f" Training Field Model: {label}")
        print("==================================================")

        model = build_model(num_classes=33, model_type=mtype, pretrained=False)
        base_pth = os.path.join("models", base_fname)
        if os.path.exists(base_pth):
            ckpt = torch.load(base_pth, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"Loaded base 99.8% weights from: {base_pth}")

        # Replace classifier head for 4 Kashmiri classes
        if mtype == "resnet50":
            in_f = model.resnet.fc[3].in_features
            model.resnet.fc[3] = nn.Linear(in_f, num_classes)
        else:
            in_f = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_f, num_classes)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        total_params = sum(p.numel() for p in model.parameters())

        best_val_acc = 0.0
        best_state = None
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
            print(f"[{mtype.upper()}] Epoch [{epoch:02d}/{EPOCHS:02d}] | Patch Train Acc: {train_acc*100:.2f}% | Field Val Acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

        elapsed_m = (time.time() - start_t) / 60
        print(f"[{mtype.upper()}] Patch Training Complete in {elapsed_m:.2f} mins. Peak Single-Pass Acc: {best_val_acc*100:.2f}%")

        # Save Checkpoint
        save_pth = os.path.join("models", f"multiscale_kashmiri_{mtype}.pth")
        torch.save({"model_state_dict": best_state, "class_names": class_names}, save_pth)

        # 5-Crop TTA Pass
        model.load_state_dict(best_state)
        model.eval()
        y_true, y_pred_tta = [], []
        with torch.no_grad():
            for img, lbl in zip(val_isolated, val_labels):
                tensor_img = val_transform(img).unsqueeze(0)
                probs_tta = predict_with_5crop_tta(model, tensor_img)
                pred = probs_tta.argmax(dim=1).item()
                y_true.append(lbl)
                y_pred_tta.append(pred)

        y_true = np.array(y_true)
        acc_tta = accuracy_score(y_true, y_pred_tta)
        prec_tta = precision_score(y_true, y_pred_tta, average="weighted", zero_division=0)
        rec_tta = recall_score(y_true, y_pred_tta, average="weighted", zero_division=0)
        f1_tta = f1_score(y_true, y_pred_tta, average="weighted", zero_division=0)

        summary_results.append({
            "Model Architecture": label,
            "Parameters": f"{total_params:,}",
            "Patch Train Accuracy": f"{train_acc * 100:.2f}%",
            "Single-Pass Field Acc": f"{best_val_acc * 100:.2f}%",
            "5-Crop TTA Field Acc": f"{acc_tta * 100:.2f}%",
            "Field Precision": f"{prec_tta:.4f}",
            "Field F1-Score": f"{f1_tta:.4f}"
        })

    print("\n" + "=" * 105)
    print("      FINAL 3-MODEL REAL-WORLD FIELD BENCHMARK (MULTI-SCALE PATCHES + FOLIAR ISOLATION + 5-CROP TTA)")
    print("=" * 105)
    df_res = pd.DataFrame(summary_results)
    print(df_res.to_string(index=False))

    csv_out = os.path.join("notebooks", "final_3model_field_benchmark.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\n[SAVED] Benchmark summary saved to: {csv_out}")

if __name__ == "__main__":
    run_3model_field_pipeline()
