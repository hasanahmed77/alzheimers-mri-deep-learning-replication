import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# define device (Apple Silicon Macs use MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def calculate_metrics(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = (torch.tensor(all_labels) == torch.tensor(all_preds)).float().mean().item()
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # store results
    df = pd.DataFrame(
        [{"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}]
    )
    df.to_csv("../results/metrics.csv", index=False)
    print(df)
