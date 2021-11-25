"""
Create PyTorch Dataset and DataLoader for ASL alphabet dataset.
"""

# Imports
import os
from typing import List

import numpy as np

# PyTorch imports
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import T_co
from torchvision.io import read_image

import constants


class ASLDataset(Dataset):
    """
    Custom Dataset of ASL alphabet images.
    """
    def __init__(self, directory: str, indices: np.ndarray):
        assert len(indices.shape) == 1, "expect 1-dim array of indices"
        self._labels = constants.DATA_LABELS
        self._folder_indices = indices
        self._img_dir = directory

    def __len__(self) -> int:
        return len(self._folder_indices) * len(self._labels)

    def __getitem__(self, idx: int) -> T_co:
        img_label, index_index = divmod(idx, len(self._folder_indices))
        label_name = self._labels[img_label]
        folder_index = self._folder_indices[index_index]
        file_name = label_name + str(folder_index) + ".jpg"
        img_path = os.path.join(self._img_dir, label_name, file_name)
        img = read_image(img_path)
        return img, img_label


def get_datasets(directory: str, set_sizes: List[int]) -> List[ASLDataset]:
    """
    Get datasets of given sizes, e.g. for training, validation, testing.
    :param directory: directory in which to find image files
    :param set_sizes: list of desired dataset sizes, requires that sum is less than or equal to 3000
    :return: list of len(set_sizes) ASLDatasets, with the ith dataset having set_sizes[i] * num_labels images
    """
    assert sum(set_sizes) <= constants.NUM_IMAGES_PER_LABEL, "Can't split up dataset in way that doesn't overlap"

    random_indices = np.arange(constants.NUM_IMAGES_PER_LABEL)
    np.random.shuffle(random_indices)
    split_indices = np.split(random_indices, indices_or_sections=np.cumsum(set_sizes))[:-1]

    datasets = [ASLDataset(directory, indices) for indices in split_indices]
    return datasets
