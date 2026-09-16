import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)
from sklearn.preprocessing import label_binarize

from src.utils.dataset import TransformedSubset, get_default_transforms
from src.models.attention_mobilenet import build_model, build_legacy_mobilenet_v3
from src.utils.explainability import GradCAM, overlay_heatmap_on_image
from src.utils.ood_detector import OODDetector

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELS_TO_EVALUATE = [
    ("MobileNetV3 (Baseline)", "plantvillage_38class_mobilenet_v3.pth", "mobilenet_v3"),
    ("Attention-MobileNetV3 (Ours)", "plantvillage_38class_attention_mobilenet.pth", "attention_mobilenet"),
    ("ResNet50 (Deep Benchmark)", "plantvillage_38class_resnet50.pth", "resnet50")
]

os.makedirs("notebooks", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation suite on device: {device}")

def evaluate_model_checkpoint(model_label, model_filename, default_mtype, val_loader, raw_dataset):
    model_path = os.path.join("models", model_filename)
    if not os.path.exists(model_path):
        print(f"[INFO] Checkpoint {model_path} not found. Skipping evaluation for {model_label}.")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", raw_dataset.classes)
    num_classes = len(class_names)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    has_cbam = any("cbam" in k for k in state_dict.keys())
    is_legacy = any("classifier.3.0.weight" in k for k in state_dict.keys())

    if is_legacy:
        model = build_legacy_mobilenet_v3(num_classes=num_classes)
        m_type = "mobilenet_v3 (Legacy)"
    elif "resnet" in default_mtype.lower() or "resnet" in str(state_dict.keys()).lower():
        model = build_model(num_classes=num_classes, model_type="resnet50", pretrained=False)
        m_type = "resnet50"
    else:
        m_type = "attention_mobilenet" if has_cbam else "mobilenet_v3"
        model = build_model(num_classes=num_classes, model_type=m_type, pretrained=False)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Latency Benchmark
    dummy_input = torch.randn(1, 3, *IMG_SIZE).to(device)
    for _ in range(10): _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    avg_latency_ms = np.mean(latencies)

    # Predictions
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
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = np.mean(specificities)

    # ROC-AUC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    else:
        roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="weighted")

    print("\n" + "=" * 65)
    print(f"       MODEL EVALUATION: {model_label.upper()} ({num_classes} Classes)")
    print("=" * 65)
    print(f"• Total Parameters:       {total_params:,}")
    print(f"• CPU Latency:            {avg_latency_ms:.2f} ms / image")
    print(f"• Accuracy:               {acc * 100:.2f}%")
    print(f"• Precision (Weighted):   {precision:.4f}")
    print(f"• Recall (Weighted):      {recall:.4f}")
    print(f"• F1-Score (Weighted):    {f1:.4f}")
    print(f"• Specificity (Macro):    {macro_specificity:.4f}")
    print(f"• ROC-AUC (Weighted OvR): {roc_auc:.4f}")
    print("-" * 65)

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{model_label} - Confusion Matrix ({num_classes} Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f"Weighted OvR ROC-AUC: {roc_auc:.4f}\nAccuracy: {acc*100:.2f}%\nF1-Score: {f1:.4f}",
             ha="center", va="center", fontsize=14, fontweight="bold")
    plt.title(f"{model_label} - Summary Metrics")
    plt.axis("off")

    plot_file = os.path.join("notebooks", f"{default_mtype}_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()

    # Generate Grad-CAM Sample Visualization
    try:
        sample_img_path = raw_dataset.imgs[0][0]
        pil_img = Image.open(sample_img_path).convert("RGB")
        _, val_transform = get_default_transforms(IMG_SIZE)
        img_tensor = val_transform(pil_img).unsqueeze(0).to(device)

        grad_cam = GradCAM(model)
        heatmap, _ = grad_cam.generate_heatmap(img_tensor)
        cam_blend = overlay_heatmap_on_image(pil_img, heatmap)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(pil_img.resize(IMG_SIZE))
        ax1.set_title("Input Leaf Image")
        ax1.axis("off")

        ax2.imshow(cam_blend)
        ax2.set_title(f"Grad-CAM ({model_label})")
        ax2.axis("off")

        cam_out_file = os.path.join("notebooks", f"{default_mtype}_gradcam_sample.png")
        plt.tight_layout()
        plt.savefig(cam_out_file, dpi=300)
        plt.close()
    except Exception as e:
        pass

    return {
        "Model": model_label,
        "Parameters": f"{total_params:,}",
        "Latency": f"{avg_latency_ms:.2f} ms",
        "Accuracy": f"{acc * 100:.2f}%",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Specificity": f"{macro_specificity:.4f}",
        "ROC-AUC": f"{roc_auc:.4f}"
    }

def main():
    data_dir = os.path.join("data", "plantvillage")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found.")
        return

    raw_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    _, val_sub = random_split(raw_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    _, val_transform = get_default_transforms(IMG_SIZE)
    val_dataset = TransformedSubset(val_sub, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []
    for label, fname, mtype in MODELS_TO_EVALUATE:
        res = evaluate_model_checkpoint(label, fname, mtype, val_loader, raw_dataset)
        if res:
            results.append(res)

    if results:
        print("\n" + "=" * 80)
        print("               COMPARATIVE ABLATION SUMMARY TABLE (ALL 3 MODELS)")
        print("=" * 80)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()