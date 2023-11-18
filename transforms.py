#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transrofm = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = torch.float).scatter_(0, torch.tensor(y), value = 1))
)
