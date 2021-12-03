"""
Functional definitions for different model types.
"""

# Imports
import time
from typing import cast, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import constants


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Model classes
"""


class ConvNet(nn.Module):
    """
    CNN to be trained on images resized to 32x32
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 29)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, layer_sizes: Tuple[int, ...]):
        """
        Initialize an MLP model, of linear layers separated by ReLU and no activation after the final layer
            (softmax can be applied manually if needed)

        :param layer_sizes: tuple of layer sizes, starting from number of input features, including all hidden layers,
                            and then ending with number of classes (must be at least 2 long)
        """
        super().__init__()
        assert len(layer_sizes) >= 2

        input_size = layer_sizes[0]
        layers = []
        for layer_size in layer_sizes[1:]:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            input_size = layer_size

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RPCA:
    def __init__(self):
        self.components_ = None

    def fit(self, X: np.ndarray):
        assert self.components_ is None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto a
        :param X: input matrix (num_samples,
        :return:
        """

"""
Training functions
"""


def fit_pca(image_matrix: np.ndarray, num_components: int, verbose: int = 1) -> PCA:
    """
    Create and fit a probabilistic PCA model to an input matrix

    :param image_matrix: numpy array shape (num_images, num_pixels) of flattened images
    :param num_components: number of dimensions to reduce to
    :param verbose: if not 0, print out time taken to fit and success message (default: 1)
    :return: PCA model object
    """
    assert len(image_matrix.shape) == 2, "Image matrix has to be flattened and 2-dimensional"
    t0 = time.time()
    pca = PCA(n_components=num_components, svd_solver='randomized').fit(image_matrix)
    if verbose:
        print(f"Fit PCA model. took {time.time() - t0:.2f} seconds.")
    return cast(PCA, pca)


def fit_robust_pca():
    pass


def fit_svm(train: np.ndarray,
            labels: np.ndarray,
            gamma: Union[float, str] = 'scale',
            C: float = 0.01,
            verbose: int = 1) -> SVC:
    """
    Create and fit a kernel SVM model to classify the training data into corresponding labels.

    :param train: training input matrix, shape (num_points, num_features), of floats
    :param labels: training labels matrix, shape (num_points,), of integers
    :param gamma: parameter of SVM (default: 'scale')
    :param C: parameter of SVM (default: 0.01)
    :param verbose: 2 to fit SVM verbose and print time, 1 to print time, 0 to print nothing (default: 1)
    :return: fitted SVC object
    """
    assert len(train.shape) == 2
    assert train.shape[0] == labels.shape[0]
    assert C > 0

    t0 = time.time()
    svm = SVC(kernel='rbf', C=C, gamma=gamma, verbose=(verbose == 2))
    svm = cast(SVC, svm.fit(train, labels))

    if verbose:
        print(f"Fit SVM model. Took {time.time() - t0:.2f} seconds.")
        t0 = time.time()
        print(f"Training accuracy: {svm.score(train, labels): .4f} (t: {time.time() - t0:.2f} s)")

    return svm


def fit_mlp(train: np.ndarray,
            labels: np.ndarray,
            layers: Tuple[int, ...],
            num_epochs: int = 5,
            batch_size: int = 64,
            lr: float = 0.001,
            verbose: bool = True) -> MLP:
    """
    Train an MLP model to classify feature vectors. Assumes data points are already shuffled.

    :param train: input matrix (num_samples, num_features)
    :param labels: labels matrix (num_samples,)
    :param layers: tuple of layer sizes to define network, starting with input size and ending with num_classes
    :param num_epochs: training epochs
    :param batch_size: sample batch size
    :param lr: learning rate for Adam optimizer
    :param verbose: if True, print intermediate training losses and final training accuracy
    :return: trained ConvNet object
    """
    assert len(train.shape) == 2
    assert train.shape[0] == labels.shape[0]
    assert len(layers) >= 2
    assert train.shape[1] == layers[0]
    assert layers[-1] == constants.NUM_DATA_LABELS
    num_batches = np.ceil(train.shape[0] / batch_size)
    t0 = time.time()

    criterion = nn.CrossEntropyLoss()
    model = MLP(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i in range(num_batches):
            images = torch.FloatTensor(train[i * batch_size: (i + 1) * batch_size]).to(device)
            labels = torch.LongTensor(labels[i * batch_size: (i + 1) * batch_size]).to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (i + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{num_batches}], Loss: {loss.item():.4f}")

    if verbose:
        print(f"Finished training MLP. Took {time.time() - t0:.2f} seconds.")
        # Measure accuracy on testing set
        print(f"Final training accuracy: {mlp_accuracy(model, train, labels, batch_size):.4f}")

    return model


def mlp_accuracy(model: MLP, test: np.ndarray, labels: np.ndarray, batch_size: int = 64) -> float:
    """
    Measure the accuracy of an MLP on a dataset.
    :param model: MLP instance to test
    :param test: input matrix of test data (num_samples, num_features)
    :param labels: labels matrix (num_samples,) of integer class indices
    :param batch_size: sample batch size
    :return: fraction of samples predicted correctly, float in [0, 1]
    """
    num_batches = np.ceil(test.shape[0] / batch_size)
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for i in range(num_batches):
            images = torch.FloatTensor(test[i * batch_size: (i + 1) * batch_size]).to(device)
            labels = torch.LongTensor(labels[i * batch_size: (i + 1) * batch_size]).to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = cast(torch.Tensor, predicted == labels)
            num_samples += labels.size(0)
            num_correct += correct.sum().item()

    return num_correct / num_samples


def fit_cnn(train_loader: DataLoader,
            num_epochs: int = 5,
            lr: float = 0.001,
            verbose: bool = True) -> ConvNet:
    """
    Train a CNN model to classify images.

    :param train_loader: DataLoader of images and labels
    :param num_epochs: training epochs
    :param lr: learning rate for Adam optimizer
    :param verbose: if True, print intermediate training losses and final training accuracy
    :return: trained ConvNet object
    """
    num_batches = len(train_loader)
    t0 = time.time()

    criterion = nn.CrossEntropyLoss()
    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (i + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{num_batches}], Loss: {loss.item():.4f}")

    if verbose:
        print(f"Finished training CNN. Took {time.time() - t0:.2f} seconds.")
        t0 = time.time()
        # Measure accuracy on testing set
        print(f"Final training accuracy: {cnn_accuracy(model, train_loader):.4f} (t: {time.time() - t0:.2f} s)")

    return model


def cnn_accuracy(model: ConvNet, data: DataLoader) -> float:
    """
    Measure the accuracy of a CNN on a dataset.
    :param model: ConvNet instance to test
    :param data: DataLoader of test data
    :return: fraction of samples predicted correctly, float in [0, 1]
    """
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data):
            images = images.float().to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = cast(torch.Tensor, predicted == labels)
            num_samples += labels.size(0)
            num_correct += correct.sum().item()

    return num_correct / num_samples
