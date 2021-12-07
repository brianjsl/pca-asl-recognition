"""
Script to generate, save, and load FGSM dataset, data loaders, and data matrices
for consistent usage across all experiments.
Take the base dataset, data loader, data matrices, copy them, and modify them with FGSM
"""

# Imports
import os
import sys
from typing import Any, Dict, List, Tuple, Union
from typing import cast
import time
import torch
import torch.nn as nn
from cnn_loader import model as model_cnn
from data_loader import fgsm_loader
from load_data_common import load_test_dataloaders
from models.models import cnn_accuracy

sys.path.append("../")

import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader

from data_loader.data_loader import ASLDataset, get_datasets
from utils import (
    add_resize_to_config,
    dataset_to_matrices,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_SAVE_PATH = "data/processed_data"
FGSM_DATALOADER_FILE = "fgsm_dataloader.pkl"
FGSM_MATRICES_FILE = "fgsm_matrices.pkl"

def fgsm(model, X, y, epsilon=0.1):
    """Construct adversarial attack using Fast Gradient Sign Method (FGSM).

    Args:
      model: NN model, such as model_cnn
      X: Features
      y: Labels
      epsilon: L_infinity norm of perturbation

    Returns:
      Perturbation (delta), of the same size as X, such that the adversarial
      examples can be constructed as X + delta.
    """
    X_clone = torch.clone(X.float())
    X_clone.requires_grad = True
    loss = nn.CrossEntropyLoss()(model(X_clone), y)
    loss.backward()
    delta = epsilon*torch.sign(X_clone.grad)
    return delta


#Test CNN accuracy on samples perturbed with FGSM

if __name__ == "__main__":
  print("Running accuracy tests...")
  #get test loader
  test_loaders = load_test_dataloaders()

  #get base test loader
  base_loader = test_loaders["base"]

  #convert to adversarial test loader
  epsilon = 1 #could change this to get a perturbation with more magnitude

  print("Obtaining adversarial examples.")

  num_samples = 0
  num_correct = 0

  for i, (images, labels) in enumerate(base_loader):
    images = images.float().to(device)
    labels = labels.to(device)
    adv_images = images + fgsm(model_cnn, images, labels, epsilon)

    # Forward pass
    outputs = model_cnn(adv_images)
    _, predicted = torch.max(outputs, 1)
    correct = cast(torch.Tensor, predicted == labels)
    num_samples += labels.size(0)
    num_correct += correct.sum().item()

  print("Accuracy: " + str(num_correct / num_samples))  

