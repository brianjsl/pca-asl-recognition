"""
Functional definitions for different model types.
"""

# Imports
import time
from typing import cast, Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import constants
from models.rpca.spca import spca
from utils import normalize_matrix


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


class ChannelPCA:
    def __init__(self, num_components: int = 33, verbose: int = 1):
        self._pca_models: List[Union[None, PCA]] = [None, None, None]
        self._num_components = num_components
        self._verbose = verbose

    def fit(self, X: np.ndarray) -> object:
        assert len(X.shape) == 3, "array should have 3 dimensions (num_images, 3, num_pixels)"
        assert X.shape[1] == 3, "second dimension of array should be 3 (RGB)"
        for channel_index in range(3):
            channel = X[:, channel_index, :]
            channel_pca_model = fit_pca_single(channel, num_components=self._num_components, verbose=self._verbose)
            self._pca_models[channel_index] = channel_pca_model
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._pca_models[0] is not None, "need to fit this model first! (call .fit(X) on a training array X)"
        assert len(X.shape) == 3, "array should have 3 dimensions (num_images, 3, num_pixels)"
        assert X.shape[1] == 3, "second dimension of array should be 3 (RGB)"
        transformed_channels = [self._pca_models[i].transform(X[:, i, :]) for i in range(3)]
        assert transformed_channels[0].shape == (X.shape[0], self._num_components)
        return np.concatenate(transformed_channels, axis=1)

    def get_eigenfingers(self) -> np.ndarray:
        assert self._pca_models[0] is not None, "need to fit this model first! (call .fit(X) on a training array X)"
        components = [model.components_ for model in self._pca_models]
        num_components, squared_size = components[0].shape
        size = round(np.sqrt(squared_size))
        assert num_components == self._num_components
        stacked = np.concatenate([fingers.reshape(num_components, 1, size, size) for fingers in components], axis=1)
        assert stacked.shape[1] == 3 and np.prod(stacked.shape) == num_components * squared_size * 3
        return stacked

    def get_state(self) -> Dict[str, Any]:
        assert self._pca_models[0] is not None, "why save an unfitted model"
        return {
            "models": self._pca_models[:],
            "num_components": self._num_components,
            "verbose": self._verbose,
        }

    @staticmethod
    def load_state(state: Dict[str, Any]) -> object:
        model = ChannelPCA(state["num_components"], state["verbose"])
        model._pca_models = state["models"]
        return model


class RPCA:
    def __init__(self,
                 lam: float = np.nan,
                 mu: float = np.nan,
                 tol: float = 10**(-7),
                 maxit: int = 1000,
                 verbose: int = 1):
        self._lam = lam
        self._mu = mu
        self._tol = tol
        self._maxit = maxit
        self.components = None
        self.rank = None
        self._verbose = verbose

    def fit(self, X: np.ndarray) -> object:
        assert len(X.shape) == 2, "Input matrix has to be flattened and 2-dimensional"
        t0 = time.time()
        L1, S1, k, rank = spca(X, self._lam, self._mu, self._tol, self._maxit, verbose=(self._verbose > 0))
        self.rank = rank
        self.components = np.linalg.svd(L1)[2][:rank]
        assert self.components.shape == (rank, X.shape[1])
        if self._verbose:
            print(f"RPCA finished on a channel. Achieved components of rank {rank} in {time.time() - t0:.2f} seconds.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2, "Input matrix has to be flattened and 2-dimensional"
        return X @ self.components.T

    def get_state(self) -> Tuple[Any, ...]:
        return self._lam, self._mu, self._tol, self._maxit, self.components, self.rank, self._verbose

    @staticmethod
    def load_state(state: Tuple[Any, ...]) -> object:
        lam, mu, tol, maxit, components, rank, verbose = state
        model = RPCA(lam, mu, tol, maxit, verbose)
        model.components = components
        model.rank = rank
        return model


class ChannelRPCA:
    def __init__(self, verbose: int = 1):
        self._rpca_models: List[Union[None, RPCA]] = [None, None, None]
        self._num_components: List[Union[None, int]] = [None, None, None]
        self._mu = None
        self._sigma = None
        self._verbose = verbose

    def fit(self, X: np.ndarray) -> object:
        assert len(X.shape) == 3, "array should have 3 dimensions (num_images, 3, num_pixels)"
        assert X.shape[1] == 3, "second dimension of array should be 3 (RGB)"

        matrix, matrix_mean, matrix_stdev = normalize_matrix(torch.Tensor.float(X))
        self._mu = matrix_mean
        self._sigma = matrix_stdev

        for channel_index in range(3):
            channel = matrix[:, channel_index, :]
            channel_rpca_model = cast(RPCA, RPCA(verbose=self._verbose).fit(channel))
            self._rpca_models[channel_index] = channel_rpca_model
            self._num_components[channel_index] = channel_rpca_model.rank
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._rpca_models[0] is not None, "need to fit this model first! (call .fit(X) on a training array X)"
        assert len(X.shape) == 3, "array should have 3 dimensions (num_images, 3, num_pixels)"
        assert X.shape[1] == 3, "second dimension of array should be 3 (RGB)"
        Y = (X - self._mu) / self._sigma
        transformed_channels = [self._rpca_models[i].transform(Y[:, i, :]) for i in range(3)]
        stacked_flat_matrix = np.concatenate(transformed_channels, axis=1)
        assert stacked_flat_matrix.shape == (X.shape[0], sum(self._num_components))
        return stacked_flat_matrix

    def get_eigenfingers(self) -> np.ndarray:
        assert self._rpca_models[0] is not None, "need to fit this model first! (call .fit(X) on a training array X)"
        components = [model.components for model in self._rpca_models]
        _, squared_size = components[0].shape
        size = round(np.sqrt(squared_size))
        num_components = max(self._num_components)
        eigenfingers_array = np.zeros((num_components, 3, size, size), dtype=np.float)
        for i, component_array in enumerate(components):
            num_features = component_array.shape[0]
            eigenfingers_array[:num_features, i] = component_array.reshape(num_features, size, size)
        return eigenfingers_array

    def get_state(self) -> Dict[str, Any]:
        assert self._rpca_models[0] is not None, "why save an unfitted model"
        return {
            "models": [model.get_state() for model in self._rpca_models],
            "num_components": self._num_components[:],
            "mu": self._mu,
            "sigma": self._sigma,
            "verbose": self._verbose,
        }

    @staticmethod
    def load_state(state: Dict[str, Any]) -> object:
        model = ChannelRPCA(state["verbose"])
        model._rpca_models = [RPCA.load_state(model_state) for model_state in state["models"]]
        model._num_components = state["num_components"]
        model._mu, model._sigma = state["mu"], state["sigma"]
        return model


"""
Training functions
"""


def fit_channel_rpca(image_matrix: np.ndarray, num_components: int, verbose: int = 1) -> ChannelRPCA:
    """
    Create and fit a probabilistic RPCA model to an input matrix

    :param image_matrix: numpy array shape (num_images, 3, num_pixels) of flattened images
    :param num_components: ignored
    :param verbose: if 1, print out time taken to fit and success message (default: 1),
                    if 2, print out same for each channel as well
    :return: ChannelRPCA model object
    """
    t0 = time.time()
    rpca = ChannelRPCA(verbose=int(verbose == 2)).fit(image_matrix)
    if verbose:
        print(f"Fit RPCA model. took {time.time() - t0:.2f} seconds.")
    return cast(ChannelRPCA, rpca)


def fit_channel_pca(image_matrix: np.ndarray, num_components: int, verbose: int = 1) -> ChannelPCA:
    """
    Create and fit a probabilistic PCA model to an input matrix

    :param image_matrix: numpy array shape (num_images, 3, num_pixels) of flattened images
    :param num_components: number of dimensions to reduce each channel to
    :param verbose: if 1, print out time taken to fit and success message (default: 1),
                    if 2, print out same for each channel as well
    :return: ChannelPCA model object
    """
    t0 = time.time()
    pca = ChannelPCA(num_components, verbose=int(verbose == 2)).fit(image_matrix)
    if verbose:
        print(f"Fit PCA model. took {time.time() - t0:.2f} seconds.")
    return cast(ChannelPCA, pca)


def fit_pca_single(input_matrix: np.ndarray, num_components: int, verbose: int = 1) -> PCA:
    """
    Create and fit a probabilistic PCA model to an input matrix

    :param input_matrix: numpy array shape (num_samples, num_features)
    :param num_components: number of dimensions to reduce to
    :param verbose: if not 0, print out time taken to fit and success message (default: 1)
    :return: PCA model object
    """
    assert len(input_matrix.shape) == 2, "Input matrix has to be flattened and 2-dimensional"
    t0 = time.time()
    pca = PCA(n_components=num_components, svd_solver='randomized').fit(input_matrix)
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
    num_batches = round(np.ceil(train.shape[0] / batch_size))
    t0 = time.time()

    criterion = nn.CrossEntropyLoss()
    model = MLP(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        indices = np.arange(train.shape[0])
        np.random.shuffle(indices)
        shuffled_train = train[indices]
        shuffled_labels = labels[indices]
        for i in range(num_batches):
            images = torch.FloatTensor(shuffled_train[i * batch_size: (i + 1) * batch_size]).to(device)
            labs = torch.LongTensor(shuffled_labels[i * batch_size: (i + 1) * batch_size]).to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss * images.shape[0]
            optimizer.step()

        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / train.shape[0]:.4f}")

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
    num_batches = round(np.ceil(test.shape[0] / batch_size))
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for i in range(num_batches):
            images = torch.FloatTensor(test[i * batch_size: (i + 1) * batch_size]).to(device)
            labs = torch.LongTensor(labels[i * batch_size: (i + 1) * batch_size]).to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = cast(torch.Tensor, predicted == labs)
            num_samples += labs.size(0)
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
