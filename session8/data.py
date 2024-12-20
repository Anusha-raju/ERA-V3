import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

# Train data transformations
train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
        # A.PadIfNeeded(min_height=64, min_width=64, always_apply=True, border_mode=0),
        A.CoarseDropout(
            p=0.2,
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=(0.4914, 0.4822, 0.4465),
            mask_fill_value=None,
        ),
        A.CenterCrop(height=32, width=32, always_apply=True),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2463, 0.2428, 0.2607)),
        ToTensorV2(),
    ]
)

# Test data transformations
test_transforms = A.Compose(
    [
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2463, 0.2428, 0.2607)),
        ToTensorV2(),
    ]
)



class CIFAR10WithAlbumentations(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)  # Convert PIL Image to numpy array
        if self.transform:
            image = self.transform(image=image)['image']  # Apply Albumentations transformations
        return image, label

    def __len__(self):
        return len(self.dataset)
    

# Load CIFAR-10 with Albumentations transforms
trainset = CIFAR10WithAlbumentations(root='./data', train=True, transform=train_transforms)
testset = CIFAR10WithAlbumentations(root='./data', train=False, transform=test_transforms)

# DataLoader to batch the images
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)