from data.data_loader import get_datasets

if __name__ == "__main__":
    train_dataset, test_dataset = get_datasets("data/asl_alphabet", [1000, 200])
    print(len(train_dataset), len(test_dataset))
    print(train_dataset._folder_indices)
    print(test_dataset._folder_indices)
