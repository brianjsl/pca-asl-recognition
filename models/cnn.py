import numpy as np
import pandas as pd
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import os
import sys
from typing import Union

sys.path.append("../")
from data_loader.data_loader import get_datasets
from data_loader.transforms import Inversion, NormalNoise, Rotate
from constants import DATA_LABELS

# hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001
num_classes = 29

# gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data augmentations
# resize all images to 32x32
cnn_transforms = {
    "base": transforms.Resize((32, 32)),
    # "inversion": [Inversion(), transforms.Resize((32, 32))],
    "normal noise": [NormalNoise(), transforms.Resize((32, 32))],
    "rotate": [Rotate(), transforms.Resize((32, 32))],
}
transform_names = sorted(cnn_transforms.keys())

# datasets
all_datasets = get_datasets(os.getcwd() + "/../data", [2800, 200], cnn_transforms)
train_dataset, test_dataset = all_datasets["base"]
augmented_datasets = [all_datasets[augmentation][1] for augmentation in transform_names[1:]]

# loader to faciliate processing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
augmented_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in augmented_datasets]

# classes
classes = DATA_LABELS


# show images
def img_show(img: Tensor, title: Union[str, None] = None):
    img = torch.permute(img, [1, 2, 0]).int().cpu()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Sigmoid()  # TODO: double check, sigmoid is used by the example but we had ReLU before
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 29),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

load_model = True
model = torch.load("saved_models/cnn_model.pt") if load_model else ConvNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if not load_model:
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
    torch.save(model, 'saved_models/trained_model.pt')

with torch.no_grad():
    for i, loader in enumerate([test_loader] + augmented_loaders):
        print("testing on loader of data:", transform_names[i])
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(num_classes)]
        n_class_samples = [0 for _ in range(num_classes)]
        for images, labels in loader:
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(images)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
                # if n_class_samples[label] == 1:
                #     img_show(images[i], title=labels[i])

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(num_classes):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
