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


class ExampleTransform(Transform):
    # invert image
    def __call__(self, sample: torch.Tensor):
        return 255 - sample

class NormalNoise(Transform):
    def __call__(self, sample: torch.Tensor):
        noise = np.random.normal(loc = 0.1, scale = 0.1, size = (3,200,200))
        weight = 50
        return (sample + weight*noise)/255
        return torch.add(sample,noise,alpha = 0)
