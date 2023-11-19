#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

### Optimization Loop ###

learning_rate = 1e-3
batch_size = 64
epochs = 10

# We pass our model’s output logits to nn.CrossEntropyLoss, which will normalize the logits and compute the prediction error.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Inside the training loop, optimization happens in three steps:
# * Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
# * Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
# * Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute logits and loss
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}]/[{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
# Epoch 1
# -------------------------------
# loss: 2.303636 [   64]/[60000]
# loss: 2.287273 [ 6464]/[60000]
# loss: 2.270515 [12864]/[60000]
# loss: 2.258887 [19264]/[60000]
# loss: 2.244964 [25664]/[60000]
# loss: 2.219705 [32064]/[60000]
# loss: 2.217785 [38464]/[60000]
# loss: 2.187076 [44864]/[60000]
# loss: 2.178488 [51264]/[60000]
# loss: 2.144987 [57664]/[60000]
# Test Error: 
#  Accuracy: 52.1%, Avg loss: 2.142837
# 
# Epoch 2
# -------------------------------
# loss: 2.152288 [   64]/[60000]
# loss: 2.137289 [ 6464]/[60000]
# loss: 2.085109 [12864]/[60000]
# loss: 2.097751 [19264]/[60000]
# loss: 2.044456 [25664]/[60000]
# loss: 1.984957 [32064]/[60000]
# loss: 2.005154 [38464]/[60000]
# loss: 1.927344 [44864]/[60000]
# loss: 1.931588 [51264]/[60000]
# loss: 1.847731 [57664]/[60000]
# Test Error: 
#  Accuracy: 55.2%, Avg loss: 1.856770
# 
# Epoch 3
# -------------------------------
# loss: 1.892105 [   64]/[60000]
# loss: 1.856135 [ 6464]/[60000]
# loss: 1.747229 [12864]/[60000]
# loss: 1.783465 [19264]/[60000]
# loss: 1.678980 [25664]/[60000]
# loss: 1.630189 [32064]/[60000]
# loss: 1.650488 [38464]/[60000]
# loss: 1.560148 [44864]/[60000]
# loss: 1.586274 [51264]/[60000]
# loss: 1.471100 [57664]/[60000]
# Test Error: 
#  Accuracy: 60.9%, Avg loss: 1.499342
# 
# Epoch 4
# -------------------------------
# loss: 1.567643 [   64]/[60000]
# loss: 1.530615 [ 6464]/[60000]
# loss: 1.390976 [12864]/[60000]
# loss: 1.457150 [19264]/[60000]
# loss: 1.350566 [25664]/[60000]
# loss: 1.339921 [32064]/[60000]
# loss: 1.355638 [38464]/[60000]
# loss: 1.287709 [44864]/[60000]
# loss: 1.317119 [51264]/[60000]
# loss: 1.215633 [57664]/[60000]
# Test Error: 
#  Accuracy: 63.5%, Avg loss: 1.245475
# 
# Epoch 5
# -------------------------------
# loss: 1.320593 [   64]/[60000]
# loss: 1.301696 [ 6464]/[60000]
# loss: 1.144048 [12864]/[60000]
# loss: 1.245623 [19264]/[60000]
# loss: 1.128897 [25664]/[60000]
# loss: 1.146614 [32064]/[60000]
# loss: 1.171934 [38464]/[60000]
# loss: 1.113330 [44864]/[60000]
# loss: 1.144406 [51264]/[60000]
# loss: 1.061321 [57664]/[60000]
# Test Error: 
#  Accuracy: 64.9%, Avg loss: 1.084400
# 
# Epoch 6
# -------------------------------
# loss: 1.152852 [   64]/[60000]
# loss: 1.154540 [ 6464]/[60000]
# loss: 0.978786 [12864]/[60000]
# loss: 1.110184 [19264]/[60000]
# loss: 0.987566 [25664]/[60000]
# loss: 1.014027 [32064]/[60000]
# loss: 1.055555 [38464]/[60000]
# loss: 1.000398 [44864]/[60000]
# loss: 1.029947 [51264]/[60000]
# loss: 0.963024 [57664]/[60000]
# Test Error: 
#  Accuracy: 65.8%, Avg loss: 0.978836
# 
# Epoch 7
# -------------------------------
# loss: 1.034480 [   64]/[60000]
# loss: 1.057782 [ 6464]/[60000]
# loss: 0.864493 [12864]/[60000]
# loss: 1.019095 [19264]/[60000]
# loss: 0.896239 [25664]/[60000]
# loss: 0.919890 [32064]/[60000]
# loss: 0.977692 [38464]/[60000]
# loss: 0.926751 [44864]/[60000]
# loss: 0.950825 [51264]/[60000]
# loss: 0.897018 [57664]/[60000]
# Test Error: 
#  Accuracy: 67.1%, Avg loss: 0.906801
# 
# Epoch 8
# -------------------------------
# loss: 0.947391 [   64]/[60000]
# loss: 0.990661 [ 6464]/[60000]
# loss: 0.782934 [12864]/[60000]
# loss: 0.954891 [19264]/[60000]
# loss: 0.834692 [25664]/[60000]
# loss: 0.851215 [32064]/[60000]
# loss: 0.922677 [38464]/[60000]
# loss: 0.878122 [44864]/[60000]
# loss: 0.894359 [51264]/[60000]
# loss: 0.849877 [57664]/[60000]
# Test Error: 
#  Accuracy: 68.4%, Avg loss: 0.855437
# 
# Epoch 9
# -------------------------------
# loss: 0.881126 [   64]/[60000]
# loss: 0.940934 [ 6464]/[60000]
# loss: 0.722597 [12864]/[60000]
# loss: 0.907738 [19264]/[60000]
# loss: 0.791169 [25664]/[60000]
# loss: 0.799870 [32064]/[60000]
# loss: 0.881362 [38464]/[60000]
# loss: 0.844640 [44864]/[60000]
# loss: 0.852668 [51264]/[60000]
# loss: 0.814428 [57664]/[60000]
# Test Error: 
#  Accuracy: 69.7%, Avg loss: 0.817102
# 
# Epoch 10
# -------------------------------
# loss: 0.828927 [   64]/[60000]
# loss: 0.901547 [ 6464]/[60000]
# loss: 0.676409 [12864]/[60000]
# loss: 0.871620 [19264]/[60000]
# loss: 0.758602 [25664]/[60000]
# loss: 0.760706 [32064]/[60000]
# loss: 0.848408 [38464]/[60000]
# loss: 0.820448 [44864]/[60000]
# loss: 0.820646 [51264]/[60000]
# loss: 0.786182 [57664]/[60000]
# Test Error: 
#  Accuracy: 70.9%, Avg loss: 0.786959
