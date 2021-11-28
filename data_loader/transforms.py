"""
Data transforms for generating irregular data from a dataset.

TODO: consider that the types of images are torch tensors, so any operations
    on them would maintain the computational graph
    idk if this would cause any inefficiencies *shrugs*
"""

# Imports
import abc

import torch
import numpy as np


class Transform(abc.ABC):
    """
    Abstract base class for a transform applicable to a dataset
    """
    @abc.abstractmethod
    def __call__(self, sample: torch.Tensor):
        """
        Return the transformed version of an image.
        :param sample: input image
        :return: transformed image
        """
        pass


class Inversion(Transform):
    """
    Inverts the colors of the Image
    """
    def __call__(self, sample: torch.Tensor):
        return 255 - sample

class NormalNoise(Transform):
    """
    Adds Normal Noise to Image
    """
    def __call__(self, sample: torch.Tensor):
        weight = 10
        noise = torch.randn(size = (3,200,200))
        return (sample + weight*noise)/255

class Rotate(Transform):
    """
    Rotates the Image
    """
    def __call__(self, sample: torch.Tensor):
        theta = 1
        pass
