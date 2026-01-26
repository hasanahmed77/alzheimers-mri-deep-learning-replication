import torch
import torch.nn as nn
from torchvision import models
import timm  # for Xception, InceptionResNetV2


def create_model(model_name="vgg16", num_classes=4, dropout=0.5):
    """
    Create a model with frozen base layers and custom classifier.

    model_name: "vgg16", "vgg19", "xception", "inceptionresnetv2"
    num_classes: number of output classes
    dropout: dropout probability
    """
    if model_name.lower() == "vgg16":
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

    elif model_name.lower() == "vgg19":
        base = models.vgg19(pretrained=True)
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

    elif model_name.lower() == "xception":
        # timm automatically downloads pretrained weights
        base = timm.create_model("xception", pretrained=True)
        for param in base.parameters():
            param.requires_grad = False
        in_features = base.fc.in_features  # Xception uses 'fc' as classifier
        base.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    elif model_name.lower() == "inceptionresnetv2":
        base = timm.create_model("inception_resnet_v2", pretrained=True)
        for param in base.parameters():
            param.requires_grad = False
        in_features = base.classifier.in_features  # InceptionResNetV2 uses 'classifier'
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    else:
        raise ValueError(
            f"Model {model_name} not supported. Choose from vgg16, vgg19, xception, inceptionresnetv2"
        )

    return base
