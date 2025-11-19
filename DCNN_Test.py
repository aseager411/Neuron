# Author: Alex Seager
# Updated: 11/15/25

# Description: Testing for DCNN

# Imports
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1) Import model
from DCNN import SimpleCNN  # and later DendriticCNN if you want

#2) Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 3) Test data setup
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

#4) Evaluation fucntions
#a) trainable params
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#b) Accuracy, loss, wall clock time
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    elapsed = time.time() - start_time
    avg_loss = running_loss / total
    avg_acc = correct / total
    time_per_sample = elapsed / total

    return avg_loss, avg_acc, elapsed, time_per_sample


#5) Main block
def main(
    model_name="simple",
    weights_path="simplecnn_mnist.pth"
):
    # a) Build model
    if model_name == "simple":
        model = SimpleCNN().to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # b) Load weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {weights_path}")

    # c) Params
    n_params = count_trainable_params(model)
    print(f"Trainable parameters: {n_params}")

    # d) Evaluate (loss, acc, wall-clock)
    test_loss, test_acc, elapsed, time_per_sample = evaluate(model, test_loader, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Inference time on full test set: {elapsed:.3f} s")
    print(f"Average time per image: {time_per_sample*1000:.3f} ms")


if __name__ == "__main__":
    main()
