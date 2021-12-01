import numpy as np
import os
import torch, torchvision
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from torchvision import transforms
from matplotlib import pyplot as plt

import time
import sys
sys.path.append("../")
from data_loader.data_loader import get_datasets
from data_loader.transforms import Inversion, NormalNoise, Rotate
from constants import DATA_LABELS

# gpu support
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#pca transforms
pca_transforms = {
    "base": None,
    "inversion": Inversion(),
    "normal noise": NormalNoise(),
    "rotate": Rotate(),
}
transform_names = sorted(pca_transforms.keys())


#plots the images
def plot_images(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


# datasets
all_datasets = get_datasets(os.getcwd() + "/../data", [2800, 200], pca_transforms)
train_dataset, test_dataset = all_datasets["base"]
# augmented_datasets = [all_datasets[augmentation][1] for augmentation in transform_names[1:]]

#principal component analysis
pca = PCA(n_components = 100)
pca.fit(train_dataset)
X = pca.transform(train_dataset)

#eigenfingers
eigenfingers = pca.components_.reshape((100,200,200))
titles = ["eigenfingers %i" % i for i in range(len(eigenfingers))]
plot_images(eigenfingers, titles, 200, 200)

