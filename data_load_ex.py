from data_loader.data_loader import get_datasets
from data_loader.transforms import ExampleTransform

if __name__ == "__main__":
    example_transforms = {
        "base": None,
        # "example": ExampleTransform(),
    }

    datasets = get_datasets("data/asl_alphabet", [1000, 200], example_transforms)
    train_dataset, test_dataset = datasets["base"]
    # train_inverted_dataset, test_inverted_dataset = datasets["example"]
    print(train_dataset[1][0])
    # print(train_inverted_dataset[1][0])
