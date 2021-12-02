# Imports
import os
import sys
import time

sys.path.append("../")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader

from data_loader.data_loader import get_datasets
from models.models import (
    fit_pca,
    fit_svm,
    fit_cnn,
    fit_mlp,
    cnn_accuracy,
    mlp_accuracy,
)
from utils import (
    add_resize_to_config,
    dataset_to_matrices,
    normalize_matrix,
)


# Create data config
data_config = {
    "base": None,
    # TODO: add the other ones when we're ready to test them
}
resized_data_config = add_resize_to_config(data_config, (32, 32))  # add resize to 32x32

load_data_from_pickle = True
if load_data_from_pickle:
    with open("models/saved_models/data_matrices.pkl", "rb") as f:
        train_image_matrix, train_label_matrix, test_image_matrix, test_label_matrix = pkl.load(f)
else:
    # Get datasets
    all_datasets = get_datasets(os.path.join(os.getcwd(), "data"), [2000, 500], resized_data_config)
    train_dataset, test_dataset = all_datasets["base"]
    print("Got datasets, flattening...")
    t0 = time.time()

    # Flatten datasets
    train_image_matrix, train_label_matrix = dataset_to_matrices(train_dataset, flatten=True, shuffle=True)
    test_image_matrix, test_label_matrix = dataset_to_matrices(test_dataset, flatten=True, shuffle=True)

    with open("models/saved_models/data_matrices.pkl", "wb") as f:
        pkl.dump((train_image_matrix, train_label_matrix, test_image_matrix, test_label_matrix), f)
    print(f"Got datasets in {time.time() - t0:.2f} seconds, shape of image matrix is {train_image_matrix.shape}")

# PCA
print("Fitting PCA...")
# TODO: normalize here, maybe? (normalize_matrix function)
pca_model = fit_pca(train_image_matrix, num_components=100, verbose=1)
reduced_train_matrix = pca_model.transform(train_image_matrix)
reduced_test_matrix = pca_model.transform(test_image_matrix)

for C in (6, 10, 15, 30):
    gamma = 'scale'
    # SVM
    print("Fitting SVM...")
    # svm_model = joblib.load("models/saved_models/svm_test.rbf")
    svm_model = fit_svm(reduced_train_matrix, train_label_matrix, gamma=gamma, C=C, verbose=1)
    joblib.dump(svm_model, f"models/saved_models/svm_test_{gamma}_{round(C*10)}.rbf")
    print("Test acc:", svm_model.score(reduced_test_matrix, test_label_matrix))

# TODO: i haven't tested fit_mlp or mlp_accuracy at all but they should work in place of lines 41, 42
