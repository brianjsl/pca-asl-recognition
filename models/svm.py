from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
import numpy as np
import torch
import os

import sys
sys.path.append("../")
from data_loader.data_loader import get_datasets
from data_loader.transforms import Inversion, NormalNoise, Rotate, Blur
from constants import DATA_LABELS

#data augmentations
svm_transforms = {
    "base": None,
    "inversion": Inversion(), 
    "normal": NormalNoise(),
    "rotate": Rotate(),
    "blur": Blur()
}

#datasets
datasets = get_datasets(os.getcwd()+"/../data", [2000,500], svm_transforms)
train_dataset, test_dataset = datasets["base"]

flat_train_data = [np.array(torch.flatten(train_dataset[i][0])) for i in range(len(train_dataset))]
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

flat_test_data = [np.array(torch.flatten(test_dataset[i][0])) for i in range(len(test_dataset))]
test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

#classes
classes = DATA_LABELS

rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.09).fit(flat_train_data, train_labels)

rbf_pred = rbf.predict(flat_test_data)

rbf_accuracy = accuracy_score(test_labels, rbf_pred)
rbf_f1 = f1_score(test_labels, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))