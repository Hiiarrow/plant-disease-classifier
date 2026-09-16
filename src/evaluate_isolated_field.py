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
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.attention_mobilenet import build_model
from src.utils.leaf_isolation import isolate_leaf_roi

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAP = {
    "APPLE ROT LEAVES": "Apple___Black_rot",
    "HEALTHY LEAVES": "Apple___healthy",
    "LEAF BLOTCH": "Apple___Cedar_apple_rust",
    "SCAB LEAVES": "Apple___Apple_scab"
}

def evaluate_isolated_field_dataset():
    print("==================================================")
    print(f" EVALUATING FIELD DATASET WITH FOLIAR ISOLATION (Device: {device})")
    print("==================================================")

    data_dir = os.path.join("data", "unseen_kashmiri_apple", "APPLE_DISEASE_DATASET")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found.")
        return

    # Transform incorporating Foliar Isolation Pre-Processing
    norm_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_dataset = datasets.ImageFolder(root=data_dir)
    print(f"Loaded {len(raw_dataset)} unseen Kashmiri field images across {len(raw_dataset.classes)} folders.")
    print("Applying Automated HSV Leaf Isolation & Background Soil Removal...")

    # Pre-process all images with leaf isolation
    processed_images = []
    processed_labels = []

    for img_path, label in raw_dataset.imgs:
        pil_img = Image.open(img_path).convert("RGB")
        isolated_img = isolate_leaf_roi(pil_img)
        tensor_img = norm_transform(isolated_img)
        processed_images.append(tensor_img)
        processed_labels.append(label)

    processed_tensor = torch.stack(processed_images)
    processed_labels = torch.tensor(processed_labels)

    class CustomTensorDataset(torch.utils.data.Dataset):
        def __init__(self, tensors, labels):
            self.tensors = tensors
            self.labels = labels
        def __getitem__(self, idx):
            return self.tensors[idx], self.labels[idx]
        def __len__(self):
            return len(self.tensors)

    loader = DataLoader(CustomTensorDataset(processed_tensor, processed_labels), batch_size=BATCH_SIZE, shuffle=False)

    models_to_evaluate = [
        ("MobileNetV3 (Baseline)", "plantvillage_38class_mobilenet_v3.pth", "mobilenet_v3"),
        ("Attention-MobileNetV3 (Ours)", "plantvillage_38class_attention_mobilenet.pth", "attention_mobilenet"),
        ("ResNet50 (Benchmark)", "plantvillage_38class_resnet50.pth", "resnet50")
    ]

    results = []

    for label, fname, mtype in models_to_evaluate:
        pth = os.path.join("models", fname)
        if not os.path.exists(pth):
            print(f"\n[SKIP] Checkpoint '{pth}' not found.")
            continue

        print(f"\nEvaluating Model on Isolated Field Leaves: {label}...")
        ckpt = torch.load(pth, map_location=device)
        class_names = ckpt.get("class_names", [])
        num_classes = len(class_names)

        folder_to_model_idx = {}
        for f_idx, folder_name in enumerate(raw_dataset.classes):
            target_name = CLASS_MAP[folder_name]
            folder_to_model_idx[f_idx] = class_names.index(target_name)

        state_dict = ckpt.get("model_state_dict", ckpt)
        model = build_model(num_classes=num_classes, model_type=mtype, pretrained=False)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())

        # Latency Benchmark
        dummy_input = torch.randn(1, 3, *IMG_SIZE).to(device)
        for _ in range(10): _ = model(dummy_input)

        latencies = []
        with torch.no_grad():
            for _ in range(50):
                start = time.perf_counter()
                _ = model(dummy_input)
                if device.type == "cuda": torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
        avg_latency_ms = np.mean(latencies)

        # Predict
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()

                for label_idx, pred_idx in zip(labels.numpy(), preds):
                    target_model_idx = folder_to_model_idx[label_idx]
                    y_true.append(target_model_idx)
                    y_pred.append(pred_idx)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model Architecture": label,
            "Parameters": f"{total_params:,}",
            "Latency": f"{avg_latency_ms:.2f} ms",
            "Isolated Field Accuracy": f"{acc * 100:.2f}%",
            "Weighted Precision": f"{prec:.4f}",
            "Weighted Recall": f"{rec:.4f}",
            "Weighted F1-Score": f"{f1:.4f}"
        })

        # Save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
        plt.title(f"Isolated Leaf Field Data: {label}\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}")
        plt.xlabel("Predicted Index")
        plt.ylabel("True Index")
        plot_path = os.path.join("notebooks", f"isolated_field_{mtype}_cm.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    if results:
        print("\n" + "=" * 95)
        print("    ISOLATED FOLIAR LEAF FIELD DATASET BENCHMARK (BACKGROUND SOIL REMOVED)")
        print("=" * 95)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))

        csv_out = os.path.join("notebooks", "isolated_field_kashmiri_benchmark.csv")
        df_res.to_csv(csv_out, index=False)
        print(f"\n[SAVED] Benchmark summary saved to: {csv_out}")

if __name__ == "__main__":
    evaluate_isolated_field_dataset()
