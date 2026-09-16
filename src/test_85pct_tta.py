import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.attention_mobilenet import build_model
from src.utils.leaf_isolation import isolate_leaf_roi

IMG_SIZE = (224, 224)
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

def test_multiscale_tta():
    print("==================================================")
    print(f" TESTING MULTI-SCALE ATTENTION-MOBILENET WITH 5-CROP TTA (Device: {device})")
    print("==================================================")

    kashmiri_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")
    pth_path = os.path.join("models", "kashmiri_apple_multiscale_attention_mobilenet.pth")

    if not os.path.exists(kashmiri_dir) or not os.path.exists(pth_path):
        print(f"[ERROR] Required paths missing.")
        return

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_dataset = datasets.ImageFolder(root=kashmiri_dir)
    class_names = raw_dataset.classes
    num_classes = len(class_names)

    ckpt = torch.load(pth_path, map_location=device)
    model = build_model(num_classes=num_classes, model_type="attention_mobilenet", pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Split 80% train / 20% val by fixed seed matching training split
    indices = list(range(len(raw_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * len(raw_dataset))
    val_idx = indices[split:]

    y_true, y_pred_std, y_pred_tta = [], [], []

    for idx in val_idx:
        img_path, label = raw_dataset.imgs[idx]
        pil_img = Image.open(img_path).convert("RGB")
        isolated_img = isolate_leaf_roi(pil_img)
        tensor_img = val_transform(isolated_img).unsqueeze(0)

        # Standard Single-Pass Prediction
        with torch.no_grad():
            outputs_std = model(tensor_img.to(device))
            pred_std = outputs_std.argmax(dim=1).item()

        # 5-Crop TTA Prediction
        probs_tta = predict_with_5crop_tta(model, tensor_img)
        pred_tta = probs_tta.argmax(dim=1).item()

        y_true.append(label)
        y_pred_std.append(pred_std)
        y_pred_tta.append(pred_tta)

    y_true = np.array(y_true)
    acc_std = accuracy_score(y_true, y_pred_std)
    acc_tta = accuracy_score(y_true, y_pred_tta)
    prec_tta = precision_score(y_true, y_pred_tta, average="weighted", zero_division=0)
    rec_tta = recall_score(y_true, y_pred_tta, average="weighted", zero_division=0)
    f1_tta = f1_score(y_true, y_pred_tta, average="weighted", zero_division=0)

    print("\n" + "=" * 80)
    print("      MULTI-SCALE ATTENTION-MOBILENET FIELD EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  - Single-Pass Field Accuracy     : {acc_std * 100:.2f}%")
    print(f"  - 5-Crop TTA Field Accuracy      : {acc_tta * 100:.2f}%")
    print(f"  - Weighted Precision (TTA)       : {prec_tta:.4f}")
    print(f"  - Weighted Recall (TTA)          : {rec_tta:.4f}")
    print(f"  - Weighted F1-Score (TTA)        : {f1_tta:.4f}")
    print("=" * 80)

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred_tta)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Multi-Scale Attention-MobileNet (5-Crop TTA)\nField Accuracy: {acc_tta*100:.2f}% | F1: {f1_tta:.4f}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plot_path = os.path.join("notebooks", "multiscale_attention_mobilenet_tta_cm.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot saved to: {plot_path}")

if __name__ == "__main__":
    test_multiscale_tta()
