import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt 
import os
import sys
from torch.utils.data import DataLoader

sys.path.append("../")
from data_loader.data_loader import get_datasets
import models.saved_models

model = torch.load(os.getcwd() + "/../models/saved_models/cnn_model.pt")

# resize all images to 32x32
cnn_transforms = {
    "base": transforms.Resize((32, 32)),
}

# base dataset
all_datasets = get_datasets(os.getcwd() + "/../data", [2800, 200], cnn_transforms)
train_dataset, test_dataset = all_datasets["base"]

def fgsm(ex_model, X, y, epsilon =0.1):
        """
        Creates an FGSM parameter delta for adversarial attack.
        delta is the perturbation
        Args: 
            ex_model = CNN model
            X: Features
            y: Labels
            epsilon: L_infinity norm of perturbation

        Output: 
            Perturbation delta of the same size as X so that adversarial attacks 
            can be constructed as X+delta
        """
        X_clone = torch.clone(X)
        X_clone.requires_grad = True
        loss = nn.CrossEntropyLoss()(ex_model(X_clone),y)
        loss.backward()
        delta = epsilon*torch.sign(X_clone.grad)
        return delta

epsilon = 0.1
X = train_dataset[:, 0] #get images 
y = train_dataset[:, 1] #get labels
X_adv = X + fgsm(model,X,y,epsilon) #adversarial images
yp2 = model(X + fgsm(model,X,y,epsilon)) #predicted labels for adversarial examples