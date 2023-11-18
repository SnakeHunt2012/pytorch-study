#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
# tensor([[1, 2],
#         [3, 4]])
# 

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
# tensor([[1, 2],
#         [3, 4]])

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
# Ones Tensor: 
#  tensor([[1, 1],
#         [1, 1]]) 

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random Tensor: \n {x_rand} \n")
# Random Tensor: 
#  tensor([[0.9184, 0.2719],
#         [0.0040, 0.7581]])

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n{rand_tensor}\n")
# Random Tensor: 
# tensor([[0.2149, 0.8314, 0.6327],
#         [0.8751, 0.1553, 0.2438]])

print(f"Ones Tensor: \n{ones_tensor}\n")
# Ones Tensor: 
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

print(f"Zeros Tensor: \n{zeros_tensor}\n")
# Zeros Tensor: 
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
# Shape of tensor: torch.Size([3, 4])

print(f"Datatype of tensor: {tensor.dtype}")
# Datatype of tensor: torch.float32

print(f"Device of tensor: {tensor.device}")
# Device of tensor: cpu

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
# First row: tensor([1., 1., 1., 1.])

print(f"First column: {tensor[:, 0]}")
# First column: tensor([1., 1., 1., 1.])

print(f"Last column: {tensor[..., -1]}")
# Last column: tensor([1., 1., 1., 1.])

tensor[:, 1] = 0
print(tensor)
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)
# tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

y1 = tensor @ tensor.T
print(y1)
# tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])

y2 = torch.matmul(tensor, tensor.T)
print(y2)
# tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])

y3 = torch.rand_like(y1)
# print(y3)
# tensor([[0.6006, 0.2459, 0.5310, 0.4245],
#         [0.9841, 0.4592, 0.9070, 0.3105],
#         [0.9627, 0.1838, 0.2281, 0.1386],
#         [0.4096, 0.3689, 0.2098, 0.6681]])

torch.matmul(tensor, tensor.T, out = y3)
print(y3)
# tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])

z1 = tensor * tensor
print(z1)
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

z2 = tensor.mul(tensor)
print(z2)
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

z3 = torch.rand_like(tensor)
print(z3)
# tensor([[0.4924, 0.1510, 0.0033, 0.2550],
#         [0.1814, 0.3907, 0.5724, 0.3381],
#         [0.7711, 0.5793, 0.2797, 0.8415],
#         [0.9393, 0.5660, 0.3002, 0.2941]])

torch.mul(tensor, tensor, out = z3)
print(z3)
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
# 12.0 <class 'float'>

print(f"{tensor}\n")
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
tensor.add_(5) # In-place operations Operations: that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
               # In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.
print(f"{tensor}\n")
# tensor([[6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.]])

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
# t: tensor([1., 1., 1., 1., 1.])

n = t.numpy()
print(f"n: {n}")
# n: [1. 1. 1. 1. 1.]

t.add_(1)
print(f"t: {t}")
# t: tensor([2., 2., 2., 2., 2.])

print(f"n: {n}")
# n: [2. 2. 2. 2. 2.]

n[2] = 6
print(f"t: {t}")
# t: tensor([2., 2., 6., 2., 2.])

n = np.ones(5)
t = torch.from_numpy(n)
print(f"t: {t}")
# t: tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

np.add(n, 1, out = n)
print(f"t: {t}")
# t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

t = torch.tensor(n)
print(f"t: {t}")
# t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

np.add(n, 1, out = n)
print(f"n: {n}")
# n: [3. 3. 3. 3. 3.]

print(f"t: {t}")
# t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
