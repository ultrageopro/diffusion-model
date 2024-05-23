"""Fashion MNIST dataset."""

from __future__ import annotations

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_dataset(image_size: int, target_idx: int | None = None) -> Dataset:
    """Get dataset."""
    transforms_list = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2),
        ],
    )

    full_dataset = torchvision.datasets.MNIST(
        "./data",
        train=True,
        transform=transforms_list,
        download=True,
    )

    if target_idx is not None:
        idx = full_dataset.targets == target_idx
        full_dataset.targets = full_dataset.targets[idx]
        full_dataset.data = full_dataset.data[idx]
    return full_dataset


def get_dataloader(
    batch_size: int,
    image_size: int,
    target_idx: int | None = None,
) -> DataLoader:
    """Get dataloader."""
    dataset = get_dataset(image_size, target_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
