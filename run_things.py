# Imports
import os
import sys

sys.path.append("../")

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

# Get datasets
all_datasets = get_datasets(os.path.join(os.getcwd(), "data"), [2000, 500], resized_data_config)
train_dataset, test_dataset = all_datasets["base"]
print("Got datasets, flattening...")

# Flatten datasets
train_image_matrix, train_label_matrix = dataset_to_matrices(train_dataset, flatten=True, shuffle=True)
test_image_matrix, test_label_matrix = dataset_to_matrices(test_dataset, flatten=True, shuffle=True)
print(train_image_matrix.shape)

# PCA
print("Fitting PCA...")
# TODO: normalize here, maybe? (normalize_matrix function)
pca_model = fit_pca(train_image_matrix, num_components=100, verbose=1)
reduced_train_matrix = pca_model.transform(train_image_matrix)
reduced_test_matrix = pca_model.transform(test_image_matrix)


# SVM
print("Fitting SVM...")
svm_model = fit_svm(reduced_train_matrix, train_label_matrix, gamma=0.5, C=0.09, verbose=1)
print("Test acc:", svm_model.score(reduced_test_matrix, test_label_matrix))

# TODO: i haven't tested fit_mlp or mlp_accuracy at all but they should work in place of lines 41, 42
