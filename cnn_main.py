"""
Main script for training CNN
"""

# Imports
import time

import torch

from load_data_common import load_train_dataloader, load_test_dataloaders
from models.models import device, fit_cnn, cnn_accuracy


MODEL_SAVE_PATH = "models/saved_models/cnn.pt"
LOAD_SAVED_MODEL = False


# Load data
print("Loading training data.")
t0 = time.time()
train_dataloader = load_train_dataloader()
print(f"Loaded data in {time.time() - t0:.2f} seconds.")

# Train CNN if desired
if LOAD_SAVED_MODEL:
    print("Loading saved CNN.")
    model = torch.load(MODEL_SAVE_PATH).to(device)
    print("Loaded. Computing training accuracy...")
    print(f"Training accuracy: {cnn_accuracy(model, train_dataloader):.4f}")
else:
    print("Training CNN.")
    model = fit_cnn(train_dataloader, verbose=True)


# Load test data
print("Loading test data.")
t0 = time.time()
test_loaders = load_test_dataloaders()
print(f"Loaded data in {time.time() - t0:.2f} seconds.")


# Test on test data
print("Running accuracy tests...")
dataset_names = sorted(test_loaders.keys())
for ds_name in dataset_names:
    loader = test_loaders[ds_name]
    print(f"{ds_name} accuracy: {cnn_accuracy(model, loader):.4f}")

print("Done.")
