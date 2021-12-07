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

def load_fgsm_dataloaders() -> Dict[str, DataLoader]:
    with open(os.path.join(DATA_SAVE_PATH, FGSM_DATALOADER_FILE), "rb") as f:
        data = pkl.load(f)
    return data


def load_fgsm_matrices() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    with open(os.path.join(DATA_SAVE_PATH, FGSM_MATRICES_FILE), "rb") as f:
        data = pkl.load(f)
    return data

def generate_and_save_fgsm_data():
  """
    Get datasets, data loaders, and flat matrices
    Save them using pickle
    :return: dictionary with keys [
        "fgsm_dataloaders",
        "fgsm_matrices"
    ]
    """

  #get test loader
  test_loaders = load_test_dataloaders()

  #get base test loader
  fgsm_loader = test_loaders["base"]

  #convert to adversarial test loader
  epsilon = 10 #could change this to get a perturbation with more magnitude

  print("Obtaining adversarial examples.")

  for i, (images, labels) in enumerate(fgsm_loader):
    images = images.float().to(device)
    labels = labels.to(device)
    images = images + fgsm(model_cnn, images, labels, epsilon)
    
  # Flatten datasets
  fgsm_dataset = fgsm_loader.dataset
  fgsm_loaders_and_matrices = {
    #"fgsm": dataset_to_matrices(fgsm_loader["fgsm"], flatten=True, shuffle=True, batch_size=64)
      "fgsm": dataset_to_matrices(fgsm_dataset, flatten=True, shuffle=True, batch_size=64)
  }
  fgsm_loader = {t: loader for t, (loader, _, _) in fgsm_loaders_and_matrices.items()}
  fgsm_matrices = {t: (m1, m2) for t, (_, m1, m2) in fgsm_loaders_and_matrices.items()}
  # test_loaders_and_matrices = {
  #       t: dataset_to_matrices(ds, flatten=True, shuffle=True, batch_size=64)
  #       for t, ds in test_datasets.items()
  #   }
  # Save everything
  with open(os.path.join(DATA_SAVE_PATH, FGSM_DATALOADER_FILE), "wb") as f:
    pkl.dump(fgsm_loader, f)
  with open(os.path.join(DATA_SAVE_PATH, FGSM_MATRICES_FILE), "wb") as f:
    pkl.dump(fgsm_matrices, f)

  return {
    "test_dataloaders": fgsm_loader,
    "test_matrices": fgsm_matrices,
  }


#either generate fgsm loader and matrices or test accuracy on adversarial examples for cnn
TEST_FGSM_CNN = True
if __name__ == "__main__":
  if TEST_FGSM_CNN:
    print("Running accuracy tests...")
    fgsm_loader = load_fgsm_dataloaders()
    dataset_names = sorted(fgsm_loader.keys())
    for ds_name in dataset_names:
      loader = fgsm_loader[ds_name]
      print(f"{ds_name} accuracy: {cnn_accuracy(model_cnn, loader):.4f}")
    print("Done.")
  else:
    generate_and_save_fgsm_data()
    print("Done.")
