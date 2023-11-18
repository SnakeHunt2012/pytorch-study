#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### Get Device for Training ###

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device {device}")

### Define the Class ###

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

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        return logits

model = NeuralNetwork().to(device)
print(model)
# Using device cpu
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

x = torch.rand(1, 28, 28, device = device)
logits = model(x)
pred_probability = nn.Softmax(dim = 1)(logits)
y_pred = pred_probability.argmax(1)
print(f"Predicted class: {y_pred}")
# Predicted class: tensor([0])

### Model Layers ###

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
# Before ReLU: tensor([[ 0.1843,  0.3745,  0.4864,  ..., -0.2721, -0.2958,  0.2677],
#         [ 0.2217,  0.2774,  0.8519,  ..., -0.1830, -0.4738,  0.0657],
#         [ 0.3290,  0.1523,  0.4379,  ...,  0.0857, -0.2008, -0.0394]],
#        grad_fn=<AddmmBackward0>)

hidden1 = nn.ReLU()(hidden1)

print(f"After ReLU: {hidden1}")
# After ReLU: tensor([[0.1843, 0.3745, 0.4864,  ..., 0.0000, 0.0000, 0.2677],
#         [0.2217, 0.2774, 0.8519,  ..., 0.0000, 0.0000, 0.0657],
#         [0.3290, 0.1523, 0.4379,  ..., 0.0857, 0.0000, 0.0000]],
#        grad_fn=<ReluBackward0>)

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(512, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(logits)
# tensor([[ 0.0407, -0.0473,  0.1390, -0.1575,  0.0367, -0.1078,  0.0212,  0.1292,
#          -0.1973,  0.0026],
#         [ 0.0253, -0.1273,  0.1957, -0.1754,  0.0589, -0.1308,  0.0888,  0.0410,
#          -0.0646, -0.0413],
#         [-0.0603,  0.1243,  0.1567, -0.2148,  0.1046, -0.1171,  0.0004,  0.0881,
#          -0.1481, -0.1010]], grad_fn=<AddmmBackward0>)

softmax = nn.Softmax(dim = 1)
pred_probability = softmax(logits)
print(pred_probability)
# tensor([[0.1050, 0.0962, 0.1159, 0.0861, 0.1046, 0.0905, 0.1030, 0.1147, 0.0828,
#          0.1011],
#         [0.1033, 0.0887, 0.1225, 0.0845, 0.1068, 0.0883, 0.1100, 0.1049, 0.0944,
#          0.0966],
#         [0.0950, 0.1143, 0.1180, 0.0814, 0.1120, 0.0898, 0.1010, 0.1102, 0.0870,
#          0.0912]], grad_fn=<SoftmaxBackward0>)

print(f"Model structure: {model}\n\n")
# Model structure: NeuralNetwork(
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
    print(f"Layer {name} | Size: {param.size()} | Values: {param[:2]}\n")
# Layer linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values: tensor([[ 0.0144,  0.0007, -0.0001,  ...,  0.0065, -0.0301, -0.0300],
#         [ 0.0066,  0.0318,  0.0127,  ...,  0.0051, -0.0352,  0.0339]],
#        grad_fn=<SliceBackward0>)
# 
# Layer linear_relu_stack.0.bias | Size: torch.Size([512]) | Values: tensor([-0.0276, -0.0132], grad_fn=<SliceBackward0>)
# 
# Layer linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values: tensor([[-0.0111,  0.0189,  0.0048,  ..., -0.0369,  0.0377,  0.0259],
#         [-0.0207,  0.0364, -0.0041,  ..., -0.0203, -0.0166, -0.0419]],
#        grad_fn=<SliceBackward0>)
# 
# Layer linear_relu_stack.2.bias | Size: torch.Size([512]) | Values: tensor([0.0193, 0.0166], grad_fn=<SliceBackward0>)
# 
# Layer linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values: tensor([[-0.0311, -0.0218, -0.0188,  ...,  0.0045, -0.0392, -0.0212],
#         [-0.0393,  0.0127, -0.0174,  ...,  0.0004,  0.0319, -0.0338]],
#        grad_fn=<SliceBackward0>)
# 
# Layer linear_relu_stack.4.bias | Size: torch.Size([10]) | Values: tensor([ 0.0373, -0.0293], grad_fn=<SliceBackward0>)
