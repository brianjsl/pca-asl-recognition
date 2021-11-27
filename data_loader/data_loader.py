"""
Create PyTorch Dataset and DataLoader for ASL alphabet dataset.
"""

# Imports
import os
from typing import Dict, List, Union

import numpy as np

# PyTorch imports
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import T_co
from torchvision.io import read_image

import constants
from data_loader.transforms import Transform


class ASLDataset(Dataset):
    """
    Custom Dataset of ASL alphabet images.
    """
    def __init__(self,
                 directory: str,
                 indices: np.ndarray,
                 transforms: Union[None, Transform, List[Transform]] = None):
        """
        :param directory: path to directory of image folders
        :param indices: 1-dim array of indices of 3000 to use from each class
        :param transforms: optional, either a Transform or list of them to apply to each image
        """
        assert len(indices.shape) == 1, "expect 1-dim array of indices"
        self._labels = constants.DATA_LABELS
        self._folder_indices = indices
        self._img_dir = directory
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._folder_indices) * len(self._labels)

    def __getitem__(self, idx: int) -> T_co:
        img_label, index_index = divmod(idx, len(self._folder_indices))
        label_name = self._labels[img_label]
        folder_index = self._folder_indices[index_index]
        file_name = label_name + str(folder_index) + ".jpg"
        img_path = os.path.join(self._img_dir, label_name, file_name)
        img = read_image(img_path)

        # Apply transforms if they exist
        transforms = self._transforms
        if transforms is not None:
            if not isinstance(transforms, list):
                transforms = [transforms]
            for t in transforms:
                img = t(img)

        return img, img_label


def get_datasets(directory: str,
                 set_sizes: List[int],
                 transforms: Dict[str, Union[None, Transform, List[Transform]]]) -> Dict[str, List[ASLDataset]]:
    """
    Get datasets of given sizes, e.g. for training, validation, testing.
    :param directory: directory in which to find image files
    :param set_sizes: list of desired dataset sizes, requires that sum is less than or equal to 3000
    :param transforms: List of transforms parameters to make datasets from
    :return: dictionary mapping transform names (given by transforms) to a list of len(set_sizes) ASLDatasets
             with the ith dataset having set_sizes[i] * num_labels images
    """
    assert sum(set_sizes) <= constants.NUM_IMAGES_PER_LABEL, "Can't split up dataset in way that doesn't overlap"

    random_indices = np.arange(constants.NUM_IMAGES_PER_LABEL)
    np.random.shuffle(random_indices)
    split_indices = np.split(random_indices, indices_or_sections=np.cumsum(set_sizes))[:-1]

    datasets = {
        name: [ASLDataset(directory, indices, tfs) for indices in split_indices]
        for name, tfs in transforms.items()
    }

    return datasets
