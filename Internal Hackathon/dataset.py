"""
dataset.py - Dataset preparation, augmentation, and data loaders for Image Classification.
"""

import os
import shutil
import random
from typing import Tuple, Dict, List
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

# ImageNet standard normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns data transforms for training and evaluation.
    Training uses robust data augmentations to prevent overfitting.
    Validation/Inference uses standard resizing and center-cropping.
    """
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {"train": train_transform, "val": val_transform}


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    val_split: float = 0.2,
    image_size: int = 224,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads images from directory.
    If 'train' and 'val' subdirectories exist in data_dir, loads them directly.
    Otherwise, splits the root classes into train and val splits.
    
    Returns:
        (train_loader, val_loader, class_names)
    """
    transforms_dict = get_transforms(image_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Case 1: Pre-split train and val directories
        train_dataset = datasets.ImageFolder(train_dir, transform=transforms_dict["train"])
        val_dataset = datasets.ImageFolder(val_dir, transform=transforms_dict["val"])
        class_names = train_dataset.classes
    else:
        # Case 2: Split a unified ImageFolder
        full_dataset = datasets.ImageFolder(data_dir)
        class_names = full_dataset.classes
        total_len = len(full_dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=generator)

        # Wrap subsets with respective transforms
        class TransformSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, idx):
                img, label = self.subset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label

            def __len__(self):
                return len(self.subset)

        train_dataset = TransformSubset(train_subset, transforms_dict["train"])
        val_dataset = TransformSubset(val_subset, transforms_dict["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, class_names
