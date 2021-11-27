import numpy as np
import pandas as pd
import torch
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
from data_loader.transforms import ExampleTransform

#hyperparameters
test_size = 0.2
num_epochs = 5
batch_size = 32  
learning_rate = 0.001
num_classes = 29

cnn_transforms = {
    "base": None
    }

#datasets
datasets = get_datasets("data/asl_alphabet", [2000,500,500], cnn_transforms)
train_dataset, test_dataset = datasets["base"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = torch.nn.Conv2d(64,,)


