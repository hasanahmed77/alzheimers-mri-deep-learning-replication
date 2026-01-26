# =========================================
# src/train.py
# =========================================

import torch
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from src.dataset import get_data_loaders
from src.model import create_model

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device (supports CUDA, MPS, CPU)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")


# =========================================
# Evaluation function
# =========================================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# =========================================
# Training function
# =========================================
def train_model(
    model_name="vgg16",
    data_dir="../data/",
    drive_base=None,
    epochs=30,
    batch_size=64,
    lr=0.003,
    patience=12,
    step_size=5,
    gamma=0.1,
    checkpoint_interval=5,
    num_classes=4,
    dropout=0.5,
):
    """
    Train a model with frozen backbone and custom classifier.

    model_name: "vgg16", "vgg19", "xception", "inceptionresnetv2"
    data_dir: path to your data folder
    drive_base: Google Drive base folder to save results
    """
    if drive_base is None:
        drive_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 1️⃣ Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir, batch_size
    )

    # 2️⃣ Create model
    model = create_model(
        model_name=model_name, num_classes=num_classes, dropout=dropout
    )
    model = model.to(device)

    # 3️⃣ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 4️⃣ Prepare Google Drive folders
    models_dir = os.path.join(drive_base, "models")
    results_dir = os.path.join(drive_base, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, f"{model_name}_metrics.csv")

    # Training loop
    best_val_acc = 0
    counter = 0  # early stopping counter
    metrics_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        val_acc = evaluate(model, val_loader)
        print(
            f"[{model_name}] Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}"
        )

        # Learning rate decay
        scheduler.step()

        # Early stopping & best model save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(
                model.state_dict(), os.path.join(models_dir, f"{model_name}_best.pth")
            )
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Checkpoint saving
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                models_dir, f"{model_name}_checkpoint_epoch{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss / len(train_loader),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save metrics for this epoch
        metrics_history.append(
            {
                "epoch": epoch + 1,
                "loss": running_loss / len(train_loader),
                "val_acc": val_acc,
            }
        )
        pd.DataFrame(metrics_history).to_csv(metrics_file, index=False)

    print(f"[{model_name}] Training finished. Best Val Acc:", best_val_acc)
    return model, val_loader, test_loader
