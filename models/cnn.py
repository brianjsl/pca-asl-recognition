import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

#datasets
train_data_path = ""
test_data_path = ""

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(64,,)


