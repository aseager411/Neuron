# Author: Alex Seager
# Updated: 11/15/25

# Description: Testing for DCNN

# Imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1) Import model

# 2) Test data setup
# Define transform to tensor (normalizing 0-1)
transform = transforms.Compose([
    transforms.ToTensor(),                               # Converts to torch tensor, shape (1, 28, 28)
    transforms.Normalize((0.1307,), (0.3081,))           # Standard MNIST normalization
])
# Load test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False, # Get the test set
    download=True,
    transform=transform
)

# Batch for efficiency 
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False # Don't want to shuffle test set
)

#3) Test function