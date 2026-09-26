"""
models.py - Transfer learning model builder for image classification.
Supports MobileNetV3 (fast/lightweight on CPU) and ResNet-18.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    model_name: str = "mobilenet_v3_small",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Builds a computer vision model with pre-trained weights and a custom classification head.
    
    Args:
        model_name: 'mobilenet_v3_small' or 'resnet18'
        num_classes: Number of target classes
        pretrained: Whether to load ImageNet pre-trained weights
        freeze_backbone: If True, freezes the feature extractor layers
    """
    model_name = model_name.lower()

    if "mobilenet" in model_name:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        # Replace classification head
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif "resnet" in model_name:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Use 'mobilenet_v3_small' or 'resnet18'.")

    return model
