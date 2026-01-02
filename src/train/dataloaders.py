from pathlib import Path
from typing import Optional

# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch
# noinspection PyPackageRequirements, PyUnresolvedReferences
from torch.utils.data import DataLoader, TensorDataset, random_split


def get_dataloaders(tensor_path: Path, batch_size: int, val_percentage, test_percentage, device: str, seed: Optional[int]) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    assert 0 <= val_percentage <= 1, "val percentage must be between 0 and 1"
    assert 0 <= test_percentage <= 1, "test percentage must be between 0 and 1"
    assert val_percentage + test_percentage < 1, "val percentage and test percentage must be less than 1 in total"

    tensor: torch.Tensor = torch.load(tensor_path)
    X = (tensor[:, :-1].float().to(device))
    y = tensor[:, -1].float().to(device)

    dataset = TensorDataset(X, y)

    val_size = int(val_percentage * len(dataset))
    assert val_size > 0, "validation percentage must round down to at least 1 row"
    test_size = int(test_percentage * len(dataset))
    assert test_size > 0, "test percentage must round down to at least 1 row"
    train_size = len(dataset) - test_size - val_size

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], g)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader, X.shape[1]