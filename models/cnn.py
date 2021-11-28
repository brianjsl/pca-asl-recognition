import numpy as np
import pandas as pd
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('../data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import sys
sys.path.append("../")
from data_loader.data_loader import get_datasets
from data_loader.transforms import Inversion, NormalNoise, Rotate
from constants import DATA_LABELS

#hyperparameters
test_size = 0.2
num_epochs = 5
batch_size = 64  
learning_rate = 0.001
num_classes = 29

#gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data augmentations
cnn_transforms = {
    "base": None,
    "inversion": Inversion(), 
    "normal": NormalNoise(),
    "rotate": Rotate()
}

#datasets
datasets = get_datasets(os.getcwd()+"/../data", [2000,500], cnn_transforms)
train_dataset, test_dataset = datasets["base"]

#loader to faciliate processign
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
    batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
    batch_size = batch_size, shuffle = True)

#classes
classes = DATA_LABELS

#show images
def imgshow(img: Tensor):
    img = torch.permute(img,[1,2,0])
    plt.imshow(img)
    plt.show()

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,128,3)
        self.conv3 = nn.Conv2d(128,256,3)

        x = torch.rand(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,1024)
        self.fc2 = nn.Linear(1024, 29)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2,2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2,2)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        return x

model = ConvNet.to(device)

criterion = torch.nn.CrossEntropyLoss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')


