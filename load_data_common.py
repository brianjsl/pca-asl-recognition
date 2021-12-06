"""
Script to generate, save, and load datasets, data loaders, and data matrices
for consistent usage across all experiments.
"""

# Imports
import os
import sys
from typing import Any, Dict, List, Tuple, Union

sys.path.append("../")

import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader

from data_loader.data_loader import ASLDataset, get_datasets
from data_loader.transforms import Inversion, NormalNoise, Rotate, Blur
from utils import (
    add_resize_to_config,
    dataset_to_matrices,
)


DATA_SAVE_PATH = "data/processed_data"
ALL_DATASETS_FILE = "all_datasets.pkl"
TRAIN_DATALOADER_FILE = "train_dataloader.pkl"
TRAIN_MATRICES_FILE = "train_matrices.pkl"
TEST_DATALOADERS_FILE = "test_dataloader.pkl"
TEST_MATRICES_FILE = "test_matrices.pkl"


def get_data_config(resized: Union[Tuple[int, int], None] = None) -> Dict[str, Any]:
    """
    Get a data config dictionary for get_datasets
    :param resized: tuple of size to resize images to, or None to not resize
    :return: data config dictionary
    """
    # Create data config
    data_config = {
        "base": None,
        "inversion": Inversion(),
        "normal noise": NormalNoise(),
        "rotate": Rotate(),
        # TODO: add other ones when we're ready to test them
        "blur": Blur()
    }
    if resized is not None:
        resized_data_config = add_resize_to_config(data_config, resized)  # add resize
        return resized_data_config
    return data_config


def load_all_datasets() -> Dict[str, List[ASLDataset]]:
    with open(os.path.join(DATA_SAVE_PATH, ALL_DATASETS_FILE), "rb") as f:
        data = pkl.load(f)
    return data


def load_train_dataloader() -> DataLoader:
    with open(os.path.join(DATA_SAVE_PATH, TRAIN_DATALOADER_FILE), "rb") as f:
        data = pkl.load(f)
    return data


def load_train_matrices() -> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(DATA_SAVE_PATH, TRAIN_MATRICES_FILE), "rb") as f:
        data = pkl.load(f)
    return data


def load_test_dataloaders() -> Dict[str, DataLoader]:
    with open(os.path.join(DATA_SAVE_PATH, TEST_DATALOADERS_FILE), "rb") as f:
        data = pkl.load(f)
    return data


def load_test_matrices() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    with open(os.path.join(DATA_SAVE_PATH, TEST_MATRICES_FILE), "rb") as f:
        data = pkl.load(f)
    return data


def generate_and_save_data(data_config: Union[None, Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get datasets, data loaders, and flat matrices
    Save them using pickle
    :return: dictionary with keys [
        "all_datasets",
        "train_dataloader",
        "train_image_matrix",
        "train_label_matrix",
        "test_dataloaders",
        "test_matrices"
    ]
    """
    if data_config is None:
        data_config = get_data_config((32, 32))
    all_datasets = get_datasets(os.path.join(os.getcwd(), "data"), [2000, 500], data_config)
    train_dataset = all_datasets["base"][0]

    # Test datasets
    transform_names = sorted(all_datasets.keys())
    test_datasets = {t: all_datasets[t][1] for t in transform_names}

    # Flatten datasets
    train_loader_and_matrices = dataset_to_matrices(train_dataset, flatten=True, shuffle=True, batch_size=64)
    train_dataloader, train_image_matrix, train_label_matrix = train_loader_and_matrices
    test_loaders_and_matrices = {
        t: dataset_to_matrices(ds, flatten=True, shuffle=True, batch_size=64)
        for t, ds in test_datasets.items()
    }
    test_loaders = {t: loader for t, (loader, _, _) in test_loaders_and_matrices.items()}
    test_matrices = {t: (m1, m2) for t, (_, m1, m2) in test_loaders_and_matrices.items()}

    # Save everything
    with open(os.path.join(DATA_SAVE_PATH, ALL_DATASETS_FILE), "wb") as f:
        pkl.dump(all_datasets, f)
    with open(os.path.join(DATA_SAVE_PATH, TRAIN_DATALOADER_FILE), "wb") as f:
        pkl.dump(train_dataloader, f)
    with open(os.path.join(DATA_SAVE_PATH, TRAIN_MATRICES_FILE), "wb") as f:
        pkl.dump((train_image_matrix, train_label_matrix), f)
    with open(os.path.join(DATA_SAVE_PATH, TEST_DATALOADERS_FILE), "wb") as f:
        pkl.dump(test_loaders, f)
    with open(os.path.join(DATA_SAVE_PATH, TEST_MATRICES_FILE), "wb") as f:
        pkl.dump(test_matrices, f)

    return {
        "all_datasets": all_datasets,
        "train_dataloader": train_dataloader,
        "train_image_matrix": train_image_matrix,
        "train_label_matrix": train_label_matrix,
        "test_dataloaders": test_loaders,
        "test_matrices": test_matrices,
    }


if __name__ == "__main__":
    generate_and_save_data()
    print("Done.")
