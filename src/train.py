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


# Training function
def train_model(
    model_name="vgg16",
    data_dir="../data/",
    epochs=30,
    batch_size=64,
    lr=0.003,
    patience=12,  # early stopping patience
    step_size=5,  # LR decay every 5 epochs
    gamma=0.1,  # LR decay factor
    checkpoint_interval=5,  # save checkpoint every N epochs
):
    # 1️⃣ Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir, batch_size
    )

    # 2️⃣ Create model
    model = create_model(model_name).to(device)

    # 3️⃣ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Prepare folders
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    metrics_file = f"../results/{model_name}_metrics.csv"

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
            f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}"
        )

        # Learning rate decay
        scheduler.step()

        # Early stopping & best model save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), f"../models/{model_name}_best.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Checkpoint saving
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"../models/{model_name}_checkpoint_epoch{epoch+1}.pth"
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

    print("Training finished. Best Val Acc:", best_val_acc)
    return model, val_loader, test_loader


# Evaluation function
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
