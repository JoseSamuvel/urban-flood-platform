"""
train.py - Training and evaluation pipeline for custom image classification.
Tracks metrics, saves checkpoints, and plots confusion matrices.
"""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import create_dataloaders
from models import build_model


def plot_metrics(history: dict, save_path: str):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", label="Val Acc")
    ax2.set_title("Accuracy over Epochs (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Metrics plot saved to: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Plots and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)

    # Add numeric annotations inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def train_model(
    data_dir: str,
    model_name: str = "mobilenet_v3_small",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_dir: str = "output"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")

    # Save class names mapping
    class_map_path = os.path.join(save_dir, "class_names.json")
    with open(class_map_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names mapping saved to: {class_map_path}")

    # 2. Build model
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / max(total_train, 1)
        epoch_train_acc = correct_train / max(total_train, 1)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / max(total_val, 1)
        epoch_val_acc = val_correct / max(total_val, 1)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "num_classes": num_classes,
                "class_names": class_names,
                "best_val_acc": best_val_acc
            }, best_model_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed // 60:.0f}m {elapsed % 60:.1f}s.")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%. Model checkpoint: {best_model_path}")

    # Plot metrics
    metrics_path = os.path.join(save_dir, "training_metrics.png")
    plot_metrics(history, metrics_path)

    # 4. Comprehensive Evaluation on Best Checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    print("\n--- Model Classification Report ---")
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    return best_model_path, class_map_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom image classification model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to image dataset directory")
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="output", help="Directory to save model & artifacts")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir
    )
