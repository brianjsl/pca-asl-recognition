# 6.867 Machine Learning Final Project (Fall 2021)
By Brian Lee, Lilian Luong, and Srinath Mahankali

## Description
A comparison of the following models to predict static ASL sign language symbols:

* Convolutional Neural Networks (CNNs)
* The PCA Algorithm(Eigenface Algorithm) with Kernel SVM and MLPs
* Stable PCP with Kernel SVMs and MLPs
* DenseNet

## Instructions
1. Run `load_data_common.py` (generates a static train/test split)
2. Run either `cnn_main.py`, `pca_mlp_main.py`, or `pca_svm_main.py`.
    - Use the flags at the top of each file (ending in `LOAD_SAVED_MODEL`) to specify whether to train new models 
    (False) or load saved versions (which are automatically created when new models are trained).
    
#### Modifying irregular data classes
Modify the `get_data_config()` function in `load_data_common.py` and re-run the file. If a custom transform is needed,
implement a new class from the interface specified in `data_loader/transforms.py`.
    
## Additional Information
Written in Pytorch
Link to Paper: here

Robust PCA code in `models/rpca` sourced from: https://github.com/wxiao0421/onlineRPCA
