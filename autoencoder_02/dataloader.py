import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

def MyDataloader(root = './data', batch_size = 32, split = (0.8, 0.2)):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root, train=False, download=True,
                                  transform=transform)
    # x_0, y_0 = train_dataset[0]
    # print(x_0, y_0)

    dataset_size = len(train_dataset)
    # print(dataset_size)
    train_size, val_size = int(dataset_size * split[0]), int(dataset_size * split[1])
    train_dataset, val_dataset = random_split(train_dataset, lengths = [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    MyDataloader()
