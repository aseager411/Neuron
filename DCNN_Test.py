# Author: Alex Seager
# Updated: 11/15/25

# Description: Testing for DCNN

# Notes
# original: Better in all categories with k = 1 (but worse than normal CNN) -> shows model not utilizing dendrites
# All params are pretty variable

# Simple CNN:
# Trainable parameters: 206922
# Test loss: 0.0329
# Test accuracy: 0.9903
# Inference time on full test set: 0.851 s
# Average time per image: 0.085 ms # highly variable

# DCNN:
# Loaded weights from dendritic2_R8_S128_k16_f16_mnist.pth
# Trainable parameters: 43218
# Test loss: 0.0317
# Test accuracy: 0.9894
# Inference time on full test set: 1.102 s
# Average time per image: 0.110 ms

# Loaded weights from dendritic2_R8_S64_k8_f72_mnist.pth
# Trainable parameters: 43602
# Test loss: 0.0333
# Test accuracy: 0.9886
# Inference time on full test set: 1.109 s
# Average time per image: 0.111 ms

# DCNN1 (5 epochs, k = 4, r = 8, out = 1):
# Trainable parameters: 20068
# Test loss: 0.0364
# Test accuracy: 0.9875
# Inference time on full test set: 0.844 s
# Average time per image: 0.084 ms # highly variable

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
from DCNN import DendriticCNN1
from DCNN import DendriticCNN2


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
def build_model(model_name, k, r, out_channels, device, n_somata, dend_per_soma, fan_in):
    if model_name == "simple":
        return SimpleCNN().to(device)
    elif model_name == "dendritic1":
        # IMPORTANT: pass the SAME hyperparams used at training time
        return DendriticCNN1(k=k, r=r, out_channels=out_channels).to(device)
    elif model_name == "dendritic2":
        # IMPORTANT: pass the SAME hyperparams used at training time
        return DendriticCNN2(r=r, n_somata=n_somata, dend_per_soma=dend_per_soma, fan_in=fan_in).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def main(
    model_name = "dendritic2",
    weights_path = "dendritic2_R8_S64_k8_f72_mnist.pth",
    k = 8,               # must match checkpoint
    r = 8,               # must match checkpoint
    out_channels=2,    # must match checkpoint 
    n_somata=64, 
    dend_per_soma=8, 
    fan_in=72
):
    # a) Build model with matching hparams
    model = build_model(model_name, k, r, out_channels, device, n_somata, dend_per_soma, fan_in)

    # b) Load weights
    state_dict = torch.load(weights_path, map_location=device)

    # Strict=True is ideal when hparams match; set False only if you expect differences.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Warning: non-strict load")
        if missing:   print("  missing keys:", missing)
        if unexpected:print("  unexpected keys:", unexpected)

    print(f"Loaded weights from {weights_path}")

    # c) Params
    n_params = count_trainable_params(model)
    print(f"Trainable parameters: {n_params}")

    # d) Evaluate
    test_loss, test_acc, elapsed, time_per_sample = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Inference time on full test set: {elapsed:.3f} s")
    print(f"Average time per image: {time_per_sample*1000:.3f} ms")

if __name__ == "__main__":
    main()

