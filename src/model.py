import torch
import torch.nn as nn
from torchvision import models


def create_model(model_name="vgg16", num_classes=4, dropout=0.5):
    if model_name == "vgg16":
        base = models.vgg16(pretrained=True)
        for param in base.parameters():
            param.requires_grad = False
        in_features = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
    # you can add Xception, VGG19, InceptionResNetV2 similarly
    return base
