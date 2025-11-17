# Author: Alex Seager
# Updated: 11/15/25

# Description: I am attempting to create a CNN variant uisng dendritic
# hierarchy.

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1) Load MNIST data and get the dataset into tensor form
# Define transform to tensor
# Center value at 0 w/ STD 1
transform = transforms.Compose([
    transforms.ToTensor(),                               # Converts to torch tensor, shape (1, 28, 28)
    transforms.Normalize((0.1307,), (0.3081,))           # Standard MNIST normalization
])
# Load the training set
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Create batches 
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# 2) Define model(s)

# Baseline CNN: 2 conv layers, 2 fully connected
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv: 1 → 16 channels, 3×3 kernel
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Second conv: 16 → 32 channels, 3×3 kernel
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32, # Double channels in second layer
            kernel_size=3,
            padding=1
        )

        # After pooling twice: 28 → 14 → 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # 28 → 14

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # 14 → 7

        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3) Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model: the neural network (nn.Module)
        dataloader: DataLoader for training data
        optimizer: optimizer (e.g., Adam, SGD)
        criterion: loss function (e.g., CrossEntropyLoss)
        device: torch.device ('cpu' or 'cuda')

    Returns:
        avg_loss: average training loss over the epoch
        avg_acc:  average training accuracy over the epoch
    """
    model.train()  # put model in training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # 1) Zero gradients
        optimizer.zero_grad()

        # 2) Forward pass
        outputs = model(images)           # shape: (batch_size, 10)

        # 3) Compute loss
        loss = criterion(outputs, labels)

        # 4) Backward pass
        loss.backward()

        # 5) Update weights
        optimizer.step()

        # Track loss
        running_loss += loss.item() * images.size(0)

        # Track accuracy
        _, predicted = outputs.max(1)     # indices of max logit along class dimension
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


# 4) Main execution block - save and export model