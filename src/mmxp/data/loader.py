import torch
from torchvision import transforms
from medmnist import INFO, Evaluator
from medmnist.dataset import (
    PathMNIST, ChestMNIST, DermaMNIST, BloodMNIST,
    OrganAMNIST, OrganCMNIST, OrganSMNIST,
    RetinaMNIST, BreastMNIST, PneumoniaMNIST,
    TissueMNIST, OCTMNIST
)

# Mapping from dataset flag to class
DATASET_MAP = {
    "pathmnist": PathMNIST,
    "chestmnist": ChestMNIST,
    "dermamnist": DermaMNIST,
    "bloodmnist": BloodMNIST,
    "organcmnist": OrganCMNIST,
    "organamnist": OrganAMNIST,
    "organsmnist": OrganSMNIST,
    "retinamnist": RetinaMNIST,
    "breastmnist": BreastMNIST,
    "pneumoniamnist": PneumoniaMNIST,
    "tissuemnist": TissueMNIST,
    "octmnist": OCTMNIST,
}

def get_loaders(dataset_flag="pathmnist", batch_size=128, download=True, root="data", transform=None):
    info = INFO[dataset_flag]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    # Default transform
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5] * n_channels, std=[.5] * n_channels)
        ])

    DatasetClass = DATASET_MAP[dataset_flag]

    train_dataset = DatasetClass(split="train", root=root, download=download, transform=transform)
    val_dataset   = DatasetClass(split="val", root=root, download=download, transform=transform)
    test_dataset  = DatasetClass(split="test", root=root, download=download, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
