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
        #for i in range(3):
        #    for j in range(3):
        #        for k in range(3):
        #            blur_kernel[i][j][k] = 1/27
        copy = torch.clone(sample)
        return torch.from_numpy(scipy.ndimage.convolve(copy, blur_kernel))


def fgsm(self, model, X, y, epsilon =0.1):
    """
    Creates an FGSM parameter delta for adversarial attack.

    Args: 
        model = NN model, eg. CNN, PCA classifier, SPCP classifier, etc.
        X: Features
        y: Labels
        epsilon: L_infinity norm of perturbation

    Output: 
        Perturbation delta of the same size as X so that adversarial attacks 
        can be constructed as X+delta
    """
    X_clone = torch.clone(X)
    X_clone.requires_grad = True
    loss = nn.CrossEntropyLoss()(model(X_clone),y)
    loss.backward()
    delta = epsilon*torch.sign(X_clone.grad)
    return delta   

# class FGSM(Transform):
#     """
#     Returns an Image with an FGSM adversarial attack
#     """
#     def fgsm(self, model, X, y, epsilon =0.1):
#         """
#         Creates an FGSM parameter delta for adversarial attack.

#         Args: 
#             model = NN model, eg. CNN, PCA classifier, SPCP classifier, etc.
#             X: Features
#             y: Labels
#             epsilon: L_infinity norm of perturbation

#         Output: 
#             Perturbation delta of the same size as X so that adversarial attacks 
#             can be constructed as X+delta
#         """
#         X_clone = torch.clone(X)
#         X_clone.requires_grad = True
#         loss = nn.CrossEntropyLoss()(model(X_clone),y)
#         loss.backward()
#         delta = epsilon*torch.sign(X_clone.grad)
#         return delta  

#     def __call__(self, sample: torch.Tensor, model):
#         copy = torch.clone(sample)
#         delta = fgsm(model, copy.im, )
