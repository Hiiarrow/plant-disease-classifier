import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# -------------------------------------------------------------
# Configuration & Hardware Setup
# -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CROPS = ["potato", "apple"]
os.makedirs("notebooks", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on device: {device}")

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_crop(crop_name):
    data_dir = os.path.join("data", crop_name)
    model_path = os.path.join("models", f"{crop_name}_mobilenet_v3.pth")

    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file {model_path} not found! Please train {crop_name} first.")
        return None

    # 1. Load Validation Data Split (20%)
    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Reconstruct Model & Load Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Latency Benchmark (Single Image)
    dummy_input = torch.randn(1, 3, *IMG_SIZE).to(device)
    for _ in range(10):  # Warmup
        _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    avg_latency_ms = np.mean(latencies)

    # 4. Predictions on Validation Set
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

    # 5. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Specificity (One-vs-Rest Macro Average)
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = np.mean(specificities)

    # ROC-AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    else:
        roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="weighted")

    # 6. Terminal Output
    print("\n" + "=" * 60)
    print(f"       MODEL EVALUATION METRICS: {crop_name.upper()}")
    print("=" * 60)
    print(f"• Total Parameters:       {total_params:,}")
    print(f"• Trainable Parameters:   {trainable_params:,}")
    print(f"• Inference Time:         {avg_latency_ms:.2f} ms / image")
    print(f"• Accuracy:               {acc * 100:.2f}%")
    print(f"• Precision (Weighted):   {precision:.4f}")
    print(f"• Recall (Weighted):      {recall:.4f}")
    print(f"• F1-Score (Weighted):    {f1:.4f}")
    print(f"• Specificity (Macro):    {macro_specificity:.4f}")
    print(f"• ROC-AUC (Weighted OvR): {roc_auc:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # 7. Generate Plots (Confusion Matrix + ROC Curves)
    plt.figure(figsize=(14, 5))

    # Confusion Matrix Plot
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{crop_name.capitalize()} - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    # ROC Curve Plot
    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {class_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"{crop_name.capitalize()} - Multi-Class ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)

    out_file = os.path.join("notebooks", f"{crop_name}_metrics.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved evaluation visualizations to: {out_file}")

# Execute for Potato and Apple
for crop in CROPS:
    evaluate_crop(crop)