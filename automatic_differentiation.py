#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad = True)
b = torch.randn(3, requires_grad = True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

### Tensors, Functions and Computational graph ###

# A function that we apply to tensors to construct computational graph is in fact an object of class Function.
# This object knows how to compute the function in the forward direction, and also how to compute its derivative during the backward propagation step.
# A reference to the backward propagation function is stored in grad_fn property of a tensor.
print(f"Gradient function of z = {z.grad_fn}")
# Gradient function of z = <AddBackward0 object at 0x14bf56bf0>

print(f"Gradient function of loss = {loss.grad_fn}")
# Gradient function of loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x14bf56bf0>

### Computing Gradients ###

print(w.grad)
# None

print(b.grad)
# None

loss.backward()

print(w.grad)
# tensor([[0.2759, 0.0015, 0.1604],
#         [0.2759, 0.0015, 0.1604],
#         [0.2759, 0.0015, 0.1604],
#         [0.2759, 0.0015, 0.1604],
#         [0.2759, 0.0015, 0.1604]])

print(b.grad)
# tensor([0.2759, 0.0015, 0.1604])

# NOTE:
# * We can only obtain the grad properties for the leaf nodes of the computational graph, which have requires_grad property set to True. For all other nodes in our graph, gradients will not be available.
print(f"z.requires_grad: {z.requires_grad}")
# z.requires_grad: True
print(f"z.grad: {z.grad}")
# /Users/zhihu555/work/pytorch-study/automatic_differentiation.py:45: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at /Users/runner/work/_temp/anaconda/conda-bld/pytorch_1699448804225/work/build/aten/src/ATen/core/TensorBody.h:494.)
#   print(f"z.grad: {z.grad}")

# NOTE:
# * We can only perform gradient calculations using backward once on a given graph, for performance reasons. If we need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.
# loss.backward()
# Traceback (most recent call last):
#   File "/Users/zhihu555/work/pytorch-study/automatic_differentiation.py", line 43, in <module>
#     loss.backward()
#   File "/Users/zhihu555/software/anaconda3/lib/python3.11/site-packages/torch/_tensor.py", line 492, in backward
#     torch.autograd.backward(
#   File "/Users/zhihu555/software/anaconda3/lib/python3.11/site-packages/torch/autograd/__init__.py", line 251, in backward
#     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

### Disabling Gradient Tracking ###

z = torch.matmul(x, w) + b
print(z.requires_grad)
# True

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
# False

# Another way to achieve the same result is to use the detach() method on the tensor:
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)
# False

# There are reasons you might want to disable gradient tracking:
# * To mark some parameters in your neural network as frozen parameters.
# * To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.

### More on Computational Graphs ###

# In a forward pass, autograd does two things simultaneously:
# * run the requested operation to compute a resulting tensor
# * maintain the operation’s gradient function in the DAG.
# 
# The backward pass kicks off when .backward() is called on the DAG root. autograd then:
# * computes the gradients from each .grad_fn,
# * accumulates them in the respective tensor’s .grad attribute
# * using the chain rule, propagates all the way to the leaf tensors.

# NOTE:
# DAGs are dynamic in PyTorch An important thing to note is that the graph is recreated from scratch;
# after each .backward() call, autograd starts populating a new graph. This is exactly what allows you to use control flow statements in your model;
# you can change the shape, size and operations at every iteration if needed.

### Tensor Gradients and Jacobian Products ###

# Instead of computing the Jacobian matrix itself, PyTorch allows you to compute Jacobian Product v^t * Jfor a given input vector v = (v_1 ... v_m).
# This is achieved by calling backward with v should be the same as the size of the original tensor, with respect to which we want to compute the product:
inp = torch.eye(4, 5, requires_grad = True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph = True)
print(f"First call\n{inp.grad}")
# tensor([[4., 2., 2., 2., 2.],
#         [2., 4., 2., 2., 2.],
#         [2., 2., 4., 2., 2.],
#         [2., 2., 2., 4., 2.]])

out.backward(torch.ones_like(out), retain_graph = True)
print(f"Second call\n{inp.grad}")
# Second call
# tensor([[8., 4., 4., 4., 4.],
#         [4., 8., 4., 4., 4.],
#         [4., 4., 8., 4., 4.],
#         [4., 4., 4., 8., 4.]])

inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph = True)
print(f"Call after zeroing gradients\n{inp.grad}")
# Call after zeroing gradients
# tensor([[4., 2., 2., 2., 2.],
#         [2., 4., 2., 2., 2.],
#         [2., 2., 4., 2., 2.],
#         [2., 2., 2., 4., 2.]])

# Notice that when we call backward for the second time with the same argument, the value of the gradient is different.
# This happens because when doing backward propagation, PyTorch accumulates the gradients, i.e. the value of computed gradients is added to the grad property of all leaf nodes of computational graph.
# If you want to compute the proper gradients, you need to zero out the grad property before.In real-life training an optimizer helps us to do this.

# NOTE:
# Previously we were calling backward() function without parameters.
# This is essentially equivalent to calling backward(torch.tensor(1.0)), which is a useful way to compute the gradients in case of a scalar-valued function, such as loss during neural network training.
