from data_loader.data_loader import get_datasets
from data_loader.transforms import ExampleTransform
from data_loader.transforms import NormalNoise
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    example_transforms = {
        "base": None,
        "normal": NormalNoise()
        # "example": ExampleTransform(),
    }

    datasets = get_datasets("data/asl_alphabet", [2000,500,500], example_transforms)
    train_dataset = datasets["base"]
    noise_dataset = datasets["normal"]
    # train_inverted_dataset, test_inverted_dataset = datasets["example"]
    print(train_dataset[1][0])
    print(noise_dataset[1][0])
    
    x=train_dataset[0][1][0]
    y = torch.permute(x,[1,2,0])
    plt.imshow(y)
    plt.show()

    x=noise_dataset[0][1][0]
    y = torch.permute(x,[1,2,0])
    plt.imshow(y)
    plt.show()
    
    # print(train_inverted_dataset[1][0])
