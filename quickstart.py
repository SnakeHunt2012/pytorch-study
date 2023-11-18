#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

batch_size = 64

# Create data loaders.
training_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y, in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape} {X.dtype}")
    print(f"Shape of y [N]: {y.shape} {y.dtype}")
    break
# Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28]) torch.float32
# Shape of y [N]: torch.Size([64]) torch.int64

### Creating Models ###

# Get cpu, gpu or mps device for training.
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define Model
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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )

for name, param in model.named_parameters():
    print(f"Layer: {name} | size: {param.size()} | Values: {param[:2]}\n")
# Layer: linear_relu_stack.0.weight | size: torch.Size([512, 784]) | Values: tensor([[ 0.0216,  0.0139, -0.0111,  ...,  0.0010,  0.0311,  0.0025],
#         [-0.0297, -0.0109, -0.0244,  ...,  0.0318, -0.0091, -0.0065]],
#        grad_fn=<SliceBackward0>)
# 
# Layer: linear_relu_stack.0.bias | size: torch.Size([512]) | Values: tensor([ 0.0224, -0.0100], grad_fn=<SliceBackward0>)
# 
# Layer: linear_relu_stack.2.weight | size: torch.Size([512, 512]) | Values: tensor([[-0.0382, -0.0384, -0.0154,  ..., -0.0334,  0.0037, -0.0270],
#         [-0.0033,  0.0347, -0.0379,  ..., -0.0152, -0.0039,  0.0285]],
#        grad_fn=<SliceBackward0>)
# 
# Layer: linear_relu_stack.2.bias | size: torch.Size([512]) | Values: tensor([-0.0058, -0.0276], grad_fn=<SliceBackward0>)
# 
# Layer: linear_relu_stack.4.weight | size: torch.Size([10, 512]) | Values: tensor([[ 0.0315,  0.0046,  0.0179,  ...,  0.0300,  0.0371,  0.0098],
#         [-0.0031, -0.0042,  0.0283,  ..., -0.0238,  0.0315,  0.0159]],
#        grad_fn=<SliceBackward0>)
# 
# Layer: linear_relu_stack.4.bias | size: torch.Size([10]) | Values: tensor([0.0178, 0.0020], grad_fn=<SliceBackward0>)


# To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call model.forward() directly!
X = torch.rand(1, 28, 28, device = device)
logits = model(X)
print(f"Logits: {logits}")
# Logits: tensor([[-0.0420,  0.0563, -0.0711, -0.1155, -0.1208,  0.0510, -0.0380, -0.0128,
#           0.0907, -0.0102]], grad_fn=<AddmmBackward0>)


pred_probab = nn.Softmax(dim = 1)(logits)
print(f"Probability: {pred_probab}")
# Probability: tensor([[0.0977, 0.1078, 0.0949, 0.0908, 0.0903, 0.1072, 0.0981, 0.1006, 0.1116,
#          0.1009]], grad_fn=<SoftmaxBackward0>)


y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
# Predicted class: tensor([8])


### Break Down ###

input_image = torch.rand(3, 28, 28)
print(input_image.size())
# torch.Size([3, 28, 28])

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
# torch.Size([3, 784])

layer1 = nn.Linear(in_features = 28 * 28, out_features = 512)
hidden1 = layer1(flat_image)
print(hidden1.size())
# torch.Size([3, 512])

print(f"Before ReLU: {hidden1}\n\n")
# Before ReLU: tensor([[ 0.2535,  0.2910, -0.3737,  ...,  0.6741, -0.1677,  0.0478],
#         [ 0.4615,  0.0490, -0.4286,  ...,  0.2084,  0.0161, -0.0052],
#         [-0.0322,  0.0813, -0.5347,  ...,  0.2517, -0.0968, -0.0392]],
#        grad_fn=<AddmmBackward0>)

hidden1 = nn.ReLU()(hidden1)

print(f"After ReLU: {hidden1}")
# After ReLU: tensor([[0.2535, 0.2910, 0.0000,  ..., 0.6741, 0.0000, 0.0478],
#         [0.4615, 0.0490, 0.0000,  ..., 0.2084, 0.0161, 0.0000],
#         [0.0000, 0.0813, 0.0000,  ..., 0.2517, 0.0000, 0.0000]],
#        grad_fn=<ReluBackward0>)

### Optimizing the Model Parameters ###

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Sets the module in training mode.
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}\t[{current:>5d}/{size:>5d}]")

train(training_dataloader, model, loss_fn, optimizer)
# loss: 2.312596  [   64/60000]
# loss: 2.297631  [ 6464/60000]
# loss: 2.275456  [12864/60000]
# loss: 2.261955  [19264/60000]
# loss: 2.246132  [25664/60000]
# loss: 2.222351  [32064/60000]
# loss: 2.224223  [38464/60000]
# loss: 2.192300  [44864/60000]
# loss: 2.201073  [51264/60000]
# loss: 2.145515  [57664/60000]

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Sets the module in evaluation mode.
    
    test_loss, correct = 0, 0
    with torch.no_grad(): # Disables gradient calculation.
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss
            correct += (logits.argmax(dim = 1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

test(test_dataloader, model, loss_fn)
# Test Error: 
#  Accuracy: 41.2%, Avg loss: 2.148468

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(training_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Epoch 1
# -------------------------------
# loss: 2.173226  [   64/60000]
# loss: 2.158645  [ 6464/60000]
# loss: 2.113475  [12864/60000]
# loss: 2.126050  [19264/60000]
# loss: 2.075432  [25664/60000]
# loss: 2.027625  [32064/60000]
# loss: 2.044671  [38464/60000]
# loss: 1.976505  [44864/60000]
# loss: 1.971887  [51264/60000]
# loss: 1.901475  [57664/60000]
# Test Error: 
#  Accuracy: 61.0%, Avg loss: 1.902055
# 
# Epoch 2
# -------------------------------
# loss: 1.930807  [   64/60000]
# loss: 1.893971  [ 6464/60000]
# loss: 1.791640  [12864/60000]
# loss: 1.827148  [19264/60000]
# loss: 1.727020  [25664/60000]
# loss: 1.676040  [32064/60000]
# loss: 1.698708  [38464/60000]
# loss: 1.601943  [44864/60000]
# loss: 1.617948  [51264/60000]
# loss: 1.511623  [57664/60000]
# Test Error: 
#  Accuracy: 63.2%, Avg loss: 1.532957
# 
# Epoch 3
# -------------------------------
# loss: 1.598191  [   64/60000]
# loss: 1.555212  [ 6464/60000]
# loss: 1.419861  [12864/60000]
# loss: 1.484074  [19264/60000]
# loss: 1.380265  [25664/60000]
# loss: 1.368225  [32064/60000]
# loss: 1.386891  [38464/60000]
# loss: 1.308774  [44864/60000]
# loss: 1.335739  [51264/60000]
# loss: 1.235645  [57664/60000]
# Test Error: 
#  Accuracy: 64.5%, Avg loss: 1.262258
# 
# Epoch 4
# -------------------------------
# loss: 1.344972  [   64/60000]
# loss: 1.315385  [ 6464/60000]
# loss: 1.163331  [12864/60000]
# loss: 1.260793  [19264/60000]
# loss: 1.151651  [25664/60000]
# loss: 1.170022  [32064/60000]
# loss: 1.194432  [38464/60000]
# loss: 1.126299  [44864/60000]
# loss: 1.156817  [51264/60000]
# loss: 1.074267  [57664/60000]
# Test Error: 
#  Accuracy: 65.6%, Avg loss: 1.092787
# 
# Epoch 5
# -------------------------------
# loss: 1.175140  [   64/60000]
# loss: 1.163934  [ 6464/60000]
# loss: 0.994308  [12864/60000]
# loss: 1.120805  [19264/60000]
# loss: 1.010140  [25664/60000]
# loss: 1.036352  [32064/60000]
# loss: 1.073311  [38464/60000]
# loss: 1.008467  [44864/60000]
# loss: 1.039250  [51264/60000]
# loss: 0.972046  [57664/60000]
# Test Error: 
#  Accuracy: 67.0%, Avg loss: 0.982054
# 
# Done!

### Saving Models ###

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

### Loading Models ###

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    logits = model(x)
    predicted, actual = classes[logits[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
