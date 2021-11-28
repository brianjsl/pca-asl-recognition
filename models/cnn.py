import numpy as np
import pandas as pd
import torch
from torch.functional import Tensor
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
batch_size = 32  
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

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,64,3)
        self.pool1 = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(64,128,3)
        self.pool2 = torch.nn.MaxPool2d(2,2)
        self.conv3 = torch.nn.Conv2d(128,256,3)
        self.pool3 = torch.nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear()


model = ConvNet.to(device)

criterion = torch.nn.CrossEntropyLoss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)




