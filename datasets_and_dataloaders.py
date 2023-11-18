#!/usr/bin/env python
# -*- coding: utf-8 -*-


### Loading a Dataset ###

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
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


### Iterating and Visualizing the Dataset ###

label_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

figure = plt.figure(figsize = (8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_index = torch.randint(len(training_data), size=(1, )).item()
    image, label = training_data[sample_index]
    figure.add_subplot(rows, cols, i)
    plt.title(label_map[label])
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap = "gray")
plt.show()


### Creating a Custom Dataset for your files ###

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform = None, target_transform = None):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        # The labels.csv file looks like:
        # tshirt1.jpg, 0
        # tshirt2.jpg, 0
        # ......
        # ankleboot999.jpg, 9    
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[index, 0])
        image = read_image(image_path)
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    
### Preparing your data for training with DataLoaders ###

from torch.utils.data import DataLoader

training_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

### Iterate through the DataLoader ###

training_features, training_labels = next(iter(training_dataloader))
print(f"Feature batch shape: {training_features.size()}")
# Feature batch shape: torch.Size([64, 1, 28, 28])

print(f"Label batch shape: {training_labels.size()}")
# Label batch shape: torch.Size([64])

image = training_features[0].squeeze()
label = training_labels[0]
plt.imshow(image, cmap = "gray")
plt.show()
print(f"Label: {label}")
# Label: 1
