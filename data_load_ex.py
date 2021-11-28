from data_loader.data_loader import get_datasets
from data_loader.transforms import Inversion
from data_loader.transforms import NormalNoise
from data_loader.transforms import Rotate
import torch
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    example_transforms = {
        "base": None,
        "normal": NormalNoise(),
        "inversion": Inversion(),
        "rotated": Rotate()
    }

    datasets = get_datasets(os.getcwd()+"/data", [2500,500], example_transforms)
    train_dataset, test_dataset = datasets["base"]
    train_dataset_noise, test_dataset_noise = datasets["normal"]
    train_inverted_dataset, test_inverted_dataset = datasets["inversion"]
    train_rotate_dataset, test_rotate_dataset = datasets["rotated"]
    print(train_dataset[1][0])
    print(train_dataset_noise[1][0])
    
    x = train_rotate_dataset[1][0]
    y = torch.permute(x,[1,2,0])
    plt.imshow(y)
    plt.show()
    
    x= train_dataset[1][0]
    y = torch.permute(x,[1,2,0])
    plt.imshow(y)
    plt.show()

    x=train_dataset_noise[1][0]
    y = torch.permute(x,[1,2,0])
    plt.imshow(y)
    plt.show()

    x = train_inverted_dataset[1][0]
    y = torch.permute(x,[1,2,0])
    plt.imshow(y)
    plt.show()

    # x = train_inverted_dataset[1][0]
    # y = torch.permute(x,[1,2,0])
    # plt.imshow(y)
    # plt.show()
    
    # print(train_inverted_dataset[1][0])