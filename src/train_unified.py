import os
import sys
from pathlib import Path

# Anchor project root path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.dataset import get_dataset_and_loaders
from src.models.attention_mobilenet import build_model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" Training Pipeline: Crop Target={args.crop.upper()} | Model={args.model}")
    print(f" Using Device: {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # Determine Data Path
    if args.crop.lower() == "all":
        data_dir = args.data_dir if args.data_dir else os.path.join("data", "plantvillage")
        model_save_path = os.path.join("models", f"plantvillage_38class_{args.model}.pth")
    else:
        data_dir = os.path.join("data", args.crop.lower())
        model_save_path = os.path.join("models", f"{args.crop.lower()}_{args.model}.pth")

    os.makedirs("models", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"[WARNING] Data directory '{data_dir}' not found.")
        return

    # Ingest data with transform isolation fix
    train_loader, val_loader, class_names = get_dataset_and_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio
    )
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes for training: {class_names}")

    # Build model architecture (AttentionMobileNetV3, MobileNetV3, or ResNet50)
    model = build_model(num_classes=num_classes, model_type=args.model, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    start_training_time = time.time()

    for epoch in range(args.epochs):
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
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        train_epoch_loss = running_loss / total if total > 0 else 0
        train_epoch_acc = correct / total if total > 0 else 0

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total if val_total > 0 else 0
        val_epoch_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step()

        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_epoch_acc)
        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)

        print(f"[{args.model.upper()}] Epoch [{epoch+1:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc*100:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc*100:.2f}%")

        if val_epoch_acc > best_val_acc + 1e-4:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "model_type": args.model,
                "num_classes": num_classes,
                "best_val_acc": best_val_acc
            }, model_save_path)
        else:
            patience_counter += 1
            print(f"[{args.model.upper()}] Validation accuracy did not improve. Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"[{args.model.upper()}] Early stopping triggered at epoch {epoch+1} (best val acc: {best_val_acc*100:.2f}%).")
                break

    total_time = time.time() - start_training_time
    print(f"\n[{args.model.upper()}] Training Complete in {total_time/60:.2f} minutes!")
    print(f"[{args.model.upper()}] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[{args.model.upper()}] Checkpoint saved to: {model_save_path}")

    # Plot metrics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"{args.crop.capitalize()} Accuracy ({args.model})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{args.crop.capitalize()} Loss ({args.model})")
    plt.legend()

    plot_path = os.path.join("notebooks", f"{args.crop.lower()}_{args.model}_training_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Plant Disease Classification Trainer")
    parser.add_argument("--crop", type=str, default="all", help="Crop target ('all', etc.)")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom data directory")
    parser.add_argument("--model", type=str, default="attention_mobilenet", choices=["attention_mobilenet", "mobilenet_v3", "resnet50"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")

    args = parser.parse_args()
    train(args)
