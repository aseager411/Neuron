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

# a) Baseline CNN: 2 conv layers, 2 fully connected
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
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Two pooling layers (both 2×2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After pooling twice: 28 → 14 → 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)          # 28 → 14

        x = F.relu(self.conv2(x))
        x = self.pool2(x)          # 14 → 7

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Dendritic layer block
class DendriticConvBlock(nn.Module):
    """
    Dendritic convolutional block with k dendrites per soma.

    - Input:  (B, C_in, H, W)   e.g. (B, 16, 14, 14)
    - Output: (B, C_out, H, W)  e.g. (B, 32, 14, 14)

    Internally:
      * We create C_out * k dendritic conv filters (3x3).
      * Group them into C_out groups of size k (k dendrites per soma).
      * Each dendrite does: conv -> ReLU.
      * A soma sums the k dendrites feeding it, then applies another ReLU.

    So each "soma channel" is the sum of k standard conv filters applied
    to the same input feature maps.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 k_dendrites: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k_dendrites

        # All dendritic filters in one conv:
        self.dendritic_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * k_dendrites,
            kernel_size=kernel_size,
            padding=padding
        )

        # Learned soma weights: one weight per (soma channel, dendrite index)
        # Shape: (C_out, k)
        self.soma_weights = nn.Parameter(
            torch.ones(out_channels, k_dendrites)
        )

    def forward(self, x):
        # x: (B, C_in, H, W)
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels, "DendriticConvBlock: wrong input channel count."

        # Apply all dendritic conv filters at once:
        # (B, C_in, H, W) -> (B, C_out * k, H, W)
        y = self.dendritic_conv(x)

        # Reshape to separate (soma, dendrite):
        # (B, C_out * k, H, W) -> (B, C_out, k, H, W)
        y = y.view(B, self.out_channels, self.k, H, W)

        # Nonlinearity at dendrite level:
        y = F.relu(y)

        # Broadcast soma weights: (1, C_out, k, 1, 1)
        w = self.soma_weights.view(1, self.out_channels, self.k, 1, 1)

        # Weighted dendritic activations:
        # (B, C_out, k, H, W) * (1, C_out, k, 1, 1) -> (B, C_out, k, H, W)
        y_weighted = y * w

        # Sum over dendrites → soma:
        # (B, C_out, k, H, W) -> (B, C_out, H, W)
        soma = y_weighted.sum(dim=2)

        # Optional soma nonlinearity:
        soma = F.relu(soma)

        return soma

# b) Dendritic CNN with fully connected final layer
class DendriticCNN1(nn.Module):
    def __init__(self, k_dendrites: int = 4):
        super().__init__()
        # 1) First conv layer: 1 -> 16 channels, 3x3 kernel, padding=1 to keep 28x28
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Pool: 28x28 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)

        # 2) Dendritic "second conv": 16 -> 32 channels
        #    Each of the 32 somas has k 3x3 dendritic filters.
        self.dendritic = DendriticConvBlock(
            in_channels=16,
            out_channels=32,
            k_dendrites=k_dendrites,
            kernel_size=3,
            padding=1
        )

        # After conv1+pool: (B, 16, 14, 14)
        # After dendritic block: (B, 32, 14, 14)
        # Flatten: 32 * 14 * 14 = 6272
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = F.relu(self.conv1(x))   # -> (B, 16, 28, 28)
        x = self.pool(x)            # -> (B, 16, 14, 14)

        x = self.dendritic(x)       # -> (B, 32, 14, 14)

        x = x.view(x.size(0), -1)   # -> (B, 32*14*14)
        x = self.fc(x)              # -> (B, 10 logits)
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

        # a) Zero gradients
        optimizer.zero_grad()

        # b) Forward pass
        outputs = model(images)           # shape: (batch_size, 10)

        # c) Compute loss
        loss = criterion(outputs, labels)

        # d) Backward pass
        loss.backward()

        # e) Update weights
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
def main(model_name="simple", epochs=10, lr=1e-3, batch_size=64):
    """
    Main execution block to train and save a given model.

    Args:
        model_name (str): which model to train ("simple", "dendritic", etc.)
        epochs (int): number of epochs to train
        lr (float): learning rate
        batch_size (int): training batch size
    """
    # ----- a) Device setup -----
    # MPS = apple m2
    # Cuda = GPUs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # ----- b) Rebuild dataloader with parameterized batch_size -----
    # (Uses global transform and MNIST dataset from above)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # ----- c) Choose model -----
    if model_name == "simple":
        model = SimpleCNN().to(device)
        model_tag = "simpleCNN"
    elif model_name == "dendritic1":
        model = DendriticCNN1(k_dendrites=4).to(device)  # k = 4 is “4 dendrites per soma”
        model_tag = "dendriticcnn1"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Training model: {model_tag}")

    # ----- d) Loss function and optimizer -----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----- e) Training loop -----
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {train_loss:.4f} "
            f"- acc: {train_acc:.4f}"
        )

    # ----- f) Save model weights -----
    save_path = f"{model_tag}_mnist.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
