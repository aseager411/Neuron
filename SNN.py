# Author: Grady Johnson
# Updated: 11/18/25

# Implementing a SNN to maximize computational efficiency when compared to 
# a standard CNN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import spikegen, utils
from snntorch import functional as SF
from snntorch import surrogate
from torch.utils.data import DataLoader

import time
import psutil
import os
import numpy as np
from tqdm import tqdm

from DCNN import SimpleCNN

# Load Mnist data
transform = tranforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(
    root="./data", train=True, download =True, transform=transform
)

test_data = torchvision.datasets.MNIST(
    root="./data", train=False, download =True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our spiking Neuron
# Key components to increasing/decreasing sparsity in the net
# very much subject to change as testing occurs
num_steps = 25 # Determines length of spike train
beta = 0.9 # Leakage or "Membrane Decay"
threshold =1 # activation threshold
reset_mechanism ="zero" # controls what neuron goes to after firing, zero means neuron resets
spike_grad = surrogate.fast_sigmoid()

class SpikingNet(nn.Module):
    """
    A simple 2-layer fully connected SNN
    Uses LIF Neurons from SNNTorch

    Big idea:
    - Input is converted to spike trains
    - membranes evolve over time
    - Output spikes are counted to form class predictions
    - Primarily Tweak the Neuron implementation to adjust sparsity of net
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1000) # Fully Connected Layer #1
        # lif stands for Leaky-integrate-and-Fire
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, 
                                spike_grad=spike_grad, reset_mechanism =reset_mechanism)
        self.fc2 = nn.Linear(1000, 10) # Fully Connected Layer #2
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, 
                                spike_grad=spike_grad, reset_mechanism =reset_mechanism)

    def forward(self, x):
        # Spike Train
        spike_data = spikegen.rate(x, num_steps=num_steps)
        # Membranes
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []
        act_counts = []

        for step in range(num_steps):
            cur_input = spike_data[..., step]
            spk1, mem1 = self.lif1(self.fc1(cur_input), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)

            spk_rec.append(spk2)
            act_counts.append(spk1.sum().item()) #Neuron Activations

        return torch.stack(spk_rec), np.mean(act_counts)

# preparing training, loss_fn subject to change
net = SpikingNet().to(device)
optimizer = optim.Adam(net.parmaters(), lr =1e-3)

# THis loss function works in count based classification  loss
loss_fn = SF.ce_count_loss(correct_rate=0.8)

num_epochs = 3

train_acc_list, neuron_act_list, runtime_list, throughput_list = []

for epoch in range(num_epochs):
    start_time = time.time()
    correct, total, total_act = 0, 0, 0

    # Monitor system metrics
    process = psutil.Process(os.getpid())

    for data, tagrets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        #Flatten
        data =data.view(-1, 28*28).to(device)
        targets = targets.to(device)

        spk_out, avg_act = net(data)
        loss = loss_fn(spk_out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = spk_out.sum(dim=0).argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        total_act +=avg_act

    # performance metrics

    acc = correct/total
    avg_act = total_act /len(train_loader)
    end_time = time.time()
    runtime = end_time -start_time
    throughput = total / runtime

    # System metrics
    mem_used = process.memory_info().rss / (1024**2)
    cpu_util = psutil.cpu_percent(interval=None)
    gpu_util = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else None

    print(f" Epoch {epoch+1}: Accuracy={acc:.3f}, Avg. Neuron Activity = {avg_act:.2f}")
    print(f"runtime={runtime:>2f}s, Throughput={throughput:.2f} img/s")
    print(f"RAM={mem_used:.2f}MB, CPU={cpu_util}% GPU={gpu_util}")

    train_acc_list.append(acc)
    neuron_act_list.append(avg_act)
    runtime_list.append(runtime)
    throughput_list.append(throughput)

def evaluate(net, loader):
    net.eval()
    correct, total, all_spikes = 0, 0, []

    with torch.no_grad():
        for data, targets in loader:
            data = data.view(-1, 28*28).to(device)
            spk_out, avg_act = net(data)
            preds = spk_out.sum(dim=0).argmax(1)
            correct += (preds == targets.to(device)).sum().item()

            total += targets.size(0)
            all_spikes.append(avg_act)

    accuracy = correct /total
    sparsity = 1 - (np.mean(all_spikes) /(28*28)) # gives us prop. of inactive neurons

    return accuracy, sparsity

test_acc, sparsity = evaluate(net, test_loader)
print(f"Test acc: {test_acc:.2f}")
print(f'Sparsity: {sparsity:.2f}')
    

