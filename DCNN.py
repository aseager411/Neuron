# Author: Alex Seager
# Updated: 11/15/25

# Description: I am attempting to create a CNN variant uisng dendritic
# hierarchy.


# Notes
# Add k hyperparameter
# Dendrites only seeing a subset isnt making sense

# Imports
import random
import math
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


# b)
# Dendritic layer block
class DendriticPatchBlock(nn.Module):
   """
   Soma-specific dendrites (no sliding, no bases) at the 7x7 stage.


   Input : [B, 32, 7, 7]
   Output: [B, C_out, 7, 7]


   For each output channel c and location (h,w):
     - Channel mix: 1x1 (32 -> r) over the whole map
     - Take centered 3x3 patch over r channels => length P = 9*r
     - K dendrites with weights W_dend[c,k,:,h,w] and bias b[c,k,h,w]
     - Branch ReLU
     - Soma α[c,k,h,w] mixes K branches -> soma ReLU
   """
   def __init__(self, in_channels=32, out_channels=1, k_dendrites=4, r=8, H=7, W=7):
       super().__init__()
       self.in_channels  = in_channels
       self.out_channels = out_channels
       self.K            = k_dendrites
       self.r            = r
       self.H_sz, self.W_sz = H, W


       # (1) Channel mix 32 -> r
       self.chan_mix = nn.Conv2d(in_channels, r, kernel_size=1, bias=True)


       # (2) Per-location dendrite parameters
       P = 9 * r  # flattened 3x3 over r channels
       bound = 1.0 / math.sqrt(P)


       W_init = torch.empty(out_channels, self.K, P, H, W)
       W_init.uniform_(-bound, bound)
       self.W_dend = nn.Parameter(W_init)                  # [C_out, K, 9r, 7, 7]


       self.b = nn.Parameter(torch.zeros(out_channels, self.K, H, W))     # [C_out, K, 7, 7]
       self.alpha = nn.Parameter(torch.full((out_channels, self.K, H, W), 1.0 / self.K))


       self.branch_act = nn.ReLU(inplace=True)
       self.soma_act   = nn.ReLU(inplace=True)


   def forward(self, x):
       # x: [B,32,7,7]
       B_, C_, H_, W_ = x.shape
       assert C_ == self.in_channels and H_ == self.H_sz and W_ == self.W_sz, \
           f"DendriticPatchBlock7 expects (B,{self.in_channels},{self.H_sz},{self.W_sz}), got {x.shape}"


       # Channel mix -> [B,r,7,7]
       z = self.chan_mix(x)


       # 3x3 patches with same-size output grid
       patches = F.unfold(z, kernel_size=3, padding=1)        # [B, 9*r, 7*7]
       patches = patches.view(B_, 9 * self.r, self.H_sz, self.W_sz)  # [B, 9*r, 7, 7]


       # Dendrite activations per (c,k,h,w):
       # A[b,c,k,h,w] = sum_p W_dend[c,k,p,h,w] * patches[b,p,h,w] + b[c,k,h,w]
       A = torch.einsum('bphw,ckphw->bckhw', patches, self.W_dend) + self.b.unsqueeze(0)  # [B,C_out,K,7,7]
       A = self.branch_act(A)


       # Soma mix per (c,h,w): y[b,c,h,w] = sum_k alpha[c,k,h,w] * A[b,c,k,h,w]
       y = torch.einsum('bckhw,ckhw->bchw', A, self.alpha)  # [B,C_out,7,7]
       y = self.soma_act(y)
       return y




# Full model: 2 conv layers -> dendritic block -> flatten -> FC
class DendriticCNN1(nn.Module):
   """
   Pipeline:
     Conv1(1->16, 3x3, pad=1) -> ReLU -> MaxPool(2)      => [B,16,14,14]
     Conv2(16->32, 3x3, pad=1) -> ReLU -> MaxPool(2)     => [B,32,7,7]
     DendriticPatchBlock7(32 -> C_out, K, r)             => [B,C_out,7,7]
     Flatten -> FC(C_out*7*7 -> 10)
   """
   def __init__(self, k: int = 4, r: int = 8, out_channels: int = 1, fc_bias: bool = True):
       super().__init__()
       # First conv stack
       self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
       self.pool1 = nn.MaxPool2d(2, 2)   # 28->14


       # Second conv stack
       self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
       self.pool2 = nn.MaxPool2d(2, 2)   # 14->7


       # Dendritic stage at 7x7
       self.dendritic = DendriticPatchBlock(
           in_channels=32,
           out_channels=out_channels,   # default 1 soma map; set >1 if you want multiple maps
           k_dendrites=k,     # default 4 dendrites per location
           r=r,                         # latent channel mix
           H=7, W=7
       )


       # Classifier
       self.fc = nn.Linear(out_channels * 7 * 7, 10, bias=fc_bias)


   def forward(self, x):
       # x: [B,1,28,28]
       x = F.relu(self.conv1(x))  # [B,16,28,28]
       x = self.pool1(x)          # [B,16,14,14]


       x = F.relu(self.conv2(x))  # [B,32,14,14]
       x = self.pool2(x)          # [B,32,7,7]


       x = self.dendritic(x)      # [B,C_out,7,7]


       x = x.view(x.size(0), -1)  # [B, C_out*49]
       logits = self.fc(x)        # [B,10]
       return logits


# c) Flattened LRF (similar to paper)
class DendriticMLPHead(nn.Module):
   """
   Dendritic MLP head operating on flattened conv features, with a learned
   channel-mix R before 2D spatial LRF sampling.


   Input : [B, in_channels=32, 7, 7]
   Output: [B, n_somata]


   Steps:
     - 1x1 conv: 32 -> latent_channels (R), giving [B, R, 7, 7]
     - Flatten to R*7*7
     - N_dend = n_somata * dend_per_soma
     - Each dendrite samples 'fan_in' inputs from the R*7*7 vector using
       a fixed 2D spatial LRF:
         - pick a random center (h0,w0) on 7x7
         - take a 3x3 patch around it, over all R channels
         - randomly choose 'fan_in' indices from that patch
   """
   def __init__(
       self,
       in_channels: int = 32,
       H: int = 7,
       W: int = 7,
       r: int = 8,   # R: learned channel-mix dimension
       n_somata: int = 32,
       dend_per_soma: int = 4,
       fan_in: int = 16,
       device=None,
       dtype=None,
   ):
       super().__init__()
       factory_kwargs = {"device": device, "dtype": dtype}


       self.in_channels     = in_channels
       self.H               = H
       self.W               = W
       self.r = r  # R
       self.input_dim       = r * H * W


       self.n_somata      = n_somata
       self.dend_per_soma = dend_per_soma
       self.fan_in        = fan_in
       self.n_dend        = n_somata * dend_per_soma


       # --- Learn R before 2D sampling: 1x1 conv 32 -> R ---
       self.chan_mix = nn.Conv2d(
           in_channels,
           r,
           kernel_size=1,
           bias=True,
           **factory_kwargs,
       )


       # ---- Fixed index mask: [n_dend, fan_in] of input indices in [0, input_dim)
       indices = self._build_indices()
       self.register_buffer("indices", indices)  # not a Parameter, but moves with the module


       # ---- Trainable dendrite parameters
       # W_dend: [n_dend, fan_in]
       bound = 1.0 / (fan_in ** 0.5)
       self.W_dend = nn.Parameter(
           torch.empty(self.n_dend, self.fan_in, **factory_kwargs).uniform_(-bound, bound)
       )
       # b_dend: [n_dend]
       self.b_dend = nn.Parameter(torch.zeros(self.n_dend, **factory_kwargs))


       # ---- Soma mixing weights α: [n_somata, dend_per_soma]
       alpha_init = torch.full(
           (n_somata, dend_per_soma),
           1.0 / dend_per_soma,
           **factory_kwargs,
       )
       self.alpha = nn.Parameter(alpha_init)


       self.branch_act = nn.ReLU(inplace=True)
       self.soma_act   = nn.ReLU(inplace=True)


   # --------- index helpers ---------
   def _chw_to_flat(self, c, h, w):
       """Map (c,h,w) to flattened index in [0, R*7*7)."""
       return c * (self.H * self.W) + h * self.W + w


   def _build_indices(self) -> torch.LongTensor:
       """
       Build [n_dend, fan_in] index tensor using 2D spatial LRF on the 7x7 grid
       over all latent_channels (R).


       For each dendrite:
         - pick random center (h0, w0)
         - consider a 3x3 patch around (h0,w0) within [0..6]x[0..6]
         - for all channels c in [0..R-1], collect (c,h,w) in that patch
         - sample 'fan_in' distinct indices from these candidates
       """
       indices = torch.empty(self.n_dend, self.fan_in, dtype=torch.long)


       for d in range(self.n_dend):
           # pick random center (h0, w0) on 7x7 grid
           h0 = random.randint(0, self.H - 1)
           w0 = random.randint(0, self.W - 1)


           candidates = []
           for dh in (-1, 0, 1):
               for dw in (-1, 0, 1):
                   h = h0 + dh
                   w = w0 + dw
                   if 0 <= h < self.H and 0 <= w < self.W:
                       for c in range(self.r):
                           flat_idx = self._chw_to_flat(c, h, w)
                           candidates.append(flat_idx)


           # candidates is up to 9 * R indices
           if len(candidates) >= self.fan_in:
               chosen = random.sample(candidates, self.fan_in)
           else:
               # very unlikely for R>=1, but just in case
               chosen = random.choices(candidates, k=self.fan_in)


           indices[d] = torch.tensor(chosen, dtype=torch.long)


       return indices


   # --------- forward ---------
   def forward(self, x):
       """
       x: [B, in_channels, 7, 7]  (e.g. [B,32,7,7])
       returns: [B, n_somata]
       """
       B, C, H, W = x.shape
       assert C == self.in_channels and H == self.H and W == self.W, \
           f"Expected (B,{self.in_channels},{self.H},{self.W}), got {x.shape}"


       # 1) Learn R first: channel mix -> [B, R, 7,7]
       z = self.chan_mix(x)  # [B, latent_channels, 7,7]


       # 2) Flatten: [B, R*7*7]
       x_flat = z.view(B, -1)


       # 3) Select inputs for each dendrite using indices
       # x_flat_exp: [B, n_dend, input_dim]
       x_flat_exp = x_flat.unsqueeze(1).expand(-1, self.n_dend, -1)
       # idx_exp: [B, n_dend, fan_in]
       idx_exp = self.indices.unsqueeze(0).expand(B, -1, -1)
       # x_sel: [B, n_dend, fan_in]
       x_sel = torch.gather(x_flat_exp, 2, idx_exp)


       # 4) Dendritic integration
       # preact: [B, n_dend]
       preact = (x_sel * self.W_dend.unsqueeze(0)).sum(dim=-1) + self.b_dend.unsqueeze(0)
       branches = self.branch_act(preact)   # [B, n_dend]


       # 5) Group into somata: reshape to [B, n_somata, dend_per_soma]
       branches = branches.view(B, self.n_somata, self.dend_per_soma)
       # alpha: [n_somata, dend_per_soma] -> [1,n_somata,dend_per_soma]
       alpha = self.alpha.unsqueeze(0)
       soma_preact = (branches * alpha).sum(dim=-1)  # [B, n_somata]
       soma_out = self.soma_act(soma_preact)         # [B, n_somata]


       return soma_out


class DendriticCNN2(nn.Module):
   """
   Conv frontend + dendritic MLP head with learned R and 2D LRF.


   Pipeline:
     Conv1(1->16, 3x3, pad=1) -> ReLU -> MaxPool(2)      => [B,16,14,14]
     Conv2(16->32, 3x3, pad=1) -> ReLU -> MaxPool(2)     => [B,32,7,7]
     DendriticMLPHead(32,7,7, latent_channels=R, ...)   => [B, n_somata]
     FC(n_somata -> 10)
   """
   def __init__(
       self,
       r: int = 8,   # R
       n_somata: int = 32,
       dend_per_soma: int = 4,
       fan_in: int = 16,
       fc_bias: bool = True,
   ):
       super().__init__()


       # First conv stack
       self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
       self.pool1 = nn.MaxPool2d(2, 2)   # 28 -> 14


       # Second conv stack
       self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
       self.pool2 = nn.MaxPool2d(2, 2)   # 14 -> 7


       # Dendritic head on 32x7x7, with learned R before LRF
       self.dendritic = DendriticMLPHead(
           in_channels=32,
           H=7,
           W=7,
           r=r,
           n_somata=n_somata,
           dend_per_soma=dend_per_soma,
           fan_in=fan_in,
       )


       # Classifier: from n_somata -> 10 classes
       self.fc = nn.Linear(n_somata, 10, bias=fc_bias)


   def forward(self, x):
       # x: [B,1,28,28]
       x = F.relu(self.conv1(x))  # [B,16,28,28]
       x = self.pool1(x)          # [B,16,14,14]


       x = F.relu(self.conv2(x))  # [B,32,14,14]
       x = self.pool2(x)          # [B,32,7,7]


       x = self.dendritic(x)      # [B, n_somata]


       logits = self.fc(x)        # [B,10]
       return logits
  
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
def main(model_name="dendritic2", epochs=5, lr=1e-3, batch_size=64, k=4, r=4, out_channels=2, n_somata=64, dend_per_soma=8, fan_in=72):
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
       model = DendriticCNN1(k=k, r=r, out_channels=out_channels).to(device)  # k = 3 is “3 dendrites per soma”
       model_tag = f"dendritic1_k{k}_r{r}_c{out_channels}"
   elif model_name == "dendritic2":
       model = DendriticCNN2(r=r, n_somata=n_somata, dend_per_soma=dend_per_soma, fan_in=fan_in).to(device)
       model_tag = f"dendritic2_R{r}_S{n_somata}_K{dend_per_soma}_f{fan_in}"
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



