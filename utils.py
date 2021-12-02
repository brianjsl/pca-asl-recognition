"""
Useful functions for data processing, etc.
"""

# Imports
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.data_loader import ASLDataset, get_datasets
from data_loader.transforms import Transform

"""
Data Loading
"""


def add_resize_to_config(config: Dict[str, Union[None, Transform, List]],
                         resize: Tuple[int, int]) -> Dict[str, Union[None, Transform, List]]:
    """
    Adds a Resize transform to every dataset configuration in a config dictionary.
    :param config: config dictionary as requested by get_datasets
    :param resize: size to resize images to
    :return: new config with added Resize transforms
    """
    updated_config = {}
    for transform_name, setting in config.items():
        if setting is None:
            updated_config[transform_name] = transforms.Resize(resize)
        elif isinstance(setting, list):
            updated_config[transform_name] = setting + [transforms.Resize(resize)]
        else:
            updated_config[transform_name] = [setting, transforms.Resize(resize)]
    return updated_config


def dataset_to_matrices(dataset: ASLDataset,
                        flatten: bool = False,
                        shuffle: bool = True,
                        batch_size: Union[None, float] = None) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Converts an ASLDataset into two NumPy matrices of images and labels, flattening images if desired.

    :param dataset: an ASLDataset to process
    :param flatten: True if the image matrix should be flattened per image, else False (default False)
    :param shuffle: True if the image order should be randomized, else False (default True)
    :param batch_size: if None, return loader with batch_size=dataset_size, else return loader with given batch size
    :return: (loader, images, labels)
        loader: DataLoader generated from dataset
        images: NumPy array with shape (len(dataset), <image_size>), where image size can be 1 or N dimensions
        labels: NumPy array with shape (len(dataset),) of integer image labels
    """
    dataset_size = len(dataset)
    loader = DataLoader(dataset, batch_size=(dataset_size if batch_size is None else batch_size), shuffle=shuffle)

    batched_images, batched_labels = list(zip(*[(images.numpy(), labels.numpy()) for images, labels in loader]))
    input_matrix = np.concatenate(batched_images, axis=0)
    label_matrix = np.concatenate(batched_labels, axis=0)

    assert input_matrix.shape[0] == dataset_size
    assert label_matrix.shape == (dataset_size,)

    # Flatten if necessary
    if flatten:
        input_matrix = input_matrix.reshape(dataset_size, -1)

    return loader, input_matrix, label_matrix


def normalize_matrix(mat: np.ndarray, separate_param: bool = True) -> Tuple[np.ndarray,
                                                                            Union[np.ndarray, float],
                                                                            Union[np.ndarray, float]]:
    """
    Normalize a matrix to mean 0 and standard deviation 1

    :param mat: NumPy array to normalize, num dims > 1
    :param separate_param: if True, normalize along the 0th axis (i.e. separately normalize each parameter across
                           samples), else False (default: True)
    :return: normalized matrix, normalizing mean, normalizing standard deviation
    """
    axis = 0 if separate_param else None
    mean = mat.mean(axis=axis)
    stdev = mat.std(axis=axis)

    return (mat - mean) / stdev, mean, stdev
