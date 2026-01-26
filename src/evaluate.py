import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(
    model,
    loader,
    model_name,
    split="test",
    save_dir=None,
):
    """
    Evaluate a trained model and save metrics to CSV.

    split: 'train', 'val', or 'test'
    """

    # Default save directory: project_root/results
    if save_dir is None:
        save_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "results")
        )

    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted"),
        "recall": recall_score(all_labels, all_preds, average="weighted"),
        "f1_score": f1_score(all_labels, all_preds, average="weighted"),
    }

    df = pd.DataFrame([metrics])

    save_path = os.path.join(save_dir, f"{model_name}_{split}_metrics.csv")
    df.to_csv(save_path, index=False)

    print(df)
    print(f"Metrics saved to: {save_path}")

    return metrics
