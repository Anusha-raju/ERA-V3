from utils import imshow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset with no transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
])

# CIFAR-10 dataset (trainset) without normalization for computing stats
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Function to calculate the mean and standard deviation
def calculate_mean_std(dataloader):
    # Initialize variables to accumulate pixel values
    mean = 0.0
    std = 0.0
    total_images_count = 0
    
    # Iterate through the dataset
    for images, _ in dataloader:
        # Calculate batch size
        batch_size = images.size(0)
        # Sum up the pixel values for each channel
        mean += images.mean([0, 2, 3]) * batch_size
        std += images.std([0, 2, 3]) * batch_size
        total_images_count += batch_size

    # Calculate final mean and std for each channel
    mean /= total_images_count
    std /= total_images_count
    
    return mean, std

# Get the mean and standard deviation for CIFAR-10 dataset
mean, std = calculate_mean_std(trainloader)

print("Mean for each channel:", mean)
print("Standard Deviation for each channel:", std)


dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))