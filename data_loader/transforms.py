"""
Data transforms for generating irregular data from a dataset.

TODO: consider that the types of images are torch tensors, so any operations
    on them would maintain the computational graph
    idk if this would cause any inefficiencies *shrugs*
"""

# Imports
import abc

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.ndimage
# from cnn_loader import model as cnn_model
from torch.autograd import grad


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
        theta = random.randint(-10,10)
        copy = torch.clone(sample)
        return torch.from_numpy(scipy.ndimage.rotate(copy, theta, axes=(1,2), reshape=False))

class Blur(Transform):
    """
    Blurs the Image
    """
    def __call__(self, sample: torch.Tensor):
        blur_kernel = torch.zeros([1,4,4]) + 1/16
        copy = torch.clone(sample)
        return torch.from_numpy(scipy.ndimage.convolve(copy, blur_kernel))

# class FGSM(Transform):
#     """
#     Perturbs the image to get adversarial inputs
#     Generates attacks using the saved cnn model and inputs the same attacks into the svm and mlp.
#     """

#     def getGradient(self, sample: torch.Tensor, label):
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(cnn_model(sample), label)
#         return grad(outputs = loss, inputs = sample)


#     def __call__(self, sample: torch.Tensor, label):
#         epsilon = 0.1
        
#         sign_data_grad = self.getGradient(sample, label).sign()
#         perturbed_sample = sample + epsilon*sign_data_grad
#         perturbed_sample = torch.clamp(perturbed_sample, 0, 1)
#         return perturbed_sample