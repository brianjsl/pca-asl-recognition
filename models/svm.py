from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from joblib import dump, load

import sys
sys.path.append("../")
from data_loader.data_loader import get_datasets
from data_loader.transforms import Inversion, NormalNoise, Rotate, Blur
from constants import DATA_LABELS

#data augmentations
svm_transforms = {
    "base": transforms.Resize((32, 32)),
    "inversion": Inversion(), 
    "normal": NormalNoise(),
    "rotate": Rotate(),
    "blur": Blur()
}

#datasets
datasets = get_datasets(os.getcwd()+"/../data", [2800, 200], svm_transforms)
train_dataset, test_dataset = datasets["base"]

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

print("loading data")

train_data = next(iter(train_loader))
flat_train_data = train_data[0].numpy().reshape(len(train_dataset), -1)
train_labels = train_data[1].numpy()

test_data = next(iter(test_loader))
flat_test_data = test_data[0].numpy().reshape(len(test_dataset), -1)
test_labels = test_data[1].numpy()

print("Converted data")
print(flat_train_data.shape, train_labels.shape, flat_test_data.shape, test_labels.shape)

#classes
classes = DATA_LABELS

verbose = 1
# rbf = svm.SVC(kernel='rbf', gamma='scale', C=0.01, verbose=verbose).fit(flat_train_data, train_labels)
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.09, verbose=verbose).fit(flat_train_data, train_labels)
print("Trained rbf kernel svm")
print("training score:", rbf.score(flat_train_data, train_labels))
dump(rbf, "saved_models/large_svm.rbf")

rbf_pred = rbf.predict(flat_test_data)
print("Got predictions")

rbf_accuracy = accuracy_score(test_labels, rbf_pred)
rbf_f1 = f1_score(test_labels, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))