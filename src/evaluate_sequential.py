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
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.utils.dataset import TransformedSubset, get_default_transforms
from src.models.attention_mobilenet import build_model
from src.utils.explainability import GradCAM, overlay_heatmap_on_image

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sequential_evaluation():
    print("==================================================")
    print(f" SEQUENTIAL MODEL EVALUATION SUITE (Device: {device})")
    print("==================================================")

    data_dir = os.path.join("data", "plantvillage")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found.")
        return

    # Ingest validation dataset
    raw_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    _, val_sub = random_split(raw_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    _, val_transform = get_default_transforms(IMG_SIZE)
    val_dataset = TransformedSubset(val_sub, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

        print(f"\nEvaluating Model sequentially: {label}...")
        ckpt = torch.load(pth, map_location=device)
        class_names = ckpt.get("class_names", raw_dataset.classes)
        num_classes = len(class_names)

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

        # Validation Predictions
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                y_probs.extend(probs)
                y_pred.extend(preds)
                y_true.extend(labels.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Specificity
        cm = confusion_matrix(y_true, y_pred)
        specs = []
        for i in range(num_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specs.append(spec)
        macro_spec = np.mean(specs)

        # ROC-AUC
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="weighted")

        results.append({
            "Model Architecture": label,
            "Parameters": f"{total_params:,}",
            "Latency": f"{avg_latency_ms:.2f} ms",
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "Weighted F1": f"{f1:.4f}",
            "Macro Specificity": f"{macro_spec:.4f}",
            "ROC-AUC": f"{roc_auc:.4f}"
        })

        # Save Plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title(f"{label} - Confusion Matrix")

        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, f"Accuracy: {acc*100:.2f}%\nF1-Score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}",
                 ha="center", va="center", fontsize=14, fontweight="bold")
        plt.title(f"{label} - Key Indicators")
        plt.axis("off")

        plot_path = os.path.join("notebooks", f"{mtype}_metrics.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Save Grad-CAM visual attribution
        try:
            sample_img_path = raw_dataset.imgs[0][0]
            pil_img = Image.open(sample_img_path).convert("RGB")
            img_tensor = val_transform(pil_img).unsqueeze(0).to(device)

            grad_cam = GradCAM(model)
            heatmap, _ = grad_cam.generate_heatmap(img_tensor)
            cam_blend = overlay_heatmap_on_image(pil_img, heatmap)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(pil_img.resize(IMG_SIZE))
            ax1.set_title("Input Leaf Image")
            ax1.axis("off")

            ax2.imshow(cam_blend)
            ax2.set_title(f"Grad-CAM ({label})")
            ax2.axis("off")

            cam_out_file = os.path.join("notebooks", f"{mtype}_gradcam_sample.png")
            plt.tight_layout()
            plt.savefig(cam_out_file, dpi=300)
            plt.close()
        except Exception:
            pass

    if results:
        print("\n" + "=" * 85)
        print("          FINAL SEQUENTIAL EVALUATION COMPARATIVE SUMMARY TABLE")
        print("=" * 85)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_sequential_evaluation()
