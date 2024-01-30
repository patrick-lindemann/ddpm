import os
import pathlib
from typing import Callable, List, Literal, Optional, Tuple

import numpy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import (
    CIFAR10,
    LSUN,
    QMNIST,
    CelebA,
    FGVCAircraft,
    VisionDataset,
)

from src.paths import DATA_DIR

"""Functions"""


def load_dataset(
    name: Literal["cifar10", "mnist", "aircraft", "lsun", "celeba"],
    transform: Callable = image_to_tensor,
    download: bool = True,
    outdir: pathlib.Path = DATA_DIR / "datasets",
) -> VisionDataset:
    # Ensure that the data folder exists
    outdir = outdir / name
    if not outdir.exists():
        os.makedirs(outdir)
    # Download the dataset
    if name == "cifar10":
        return CIFAR10(root=outdir, download=download, transform=transform)
    elif name == "mnist":
        return QMNIST(root=outdir, download=download, transform=transform)
    elif name == "aircraft":
        return FGVCAircraft(root=outdir, download=download, transform=transform)
    elif name == "celeba":
        return CelebA(root=outdir, download=download, transform=transform)
    elif name == "lsun":
        return LSUN(
            root=outdir,
            classes=["bedroom"],
            download=download,
            transform=transform,
        )
    else:
        raise ValueError(f'Invalid dataset specified: "{name}".')


def create_dataloader(
    dataset: Dataset,
    train_size: Optional[float | int] = None,
    test_size: Optional[float | int] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> [DataLoader, DataLoader]:
    if isinstance(train_size, int):
        assert train_size <= len(dataset)
        train_size = float(train_size) / len(dataset)
    if isinstance(test_size, int):
        assert test_size <= len(dataset)
        test_size = float(test_size) / len(dataset)
    assert batch_size <= len(dataset)
    train_indices, test_indices, _, _ = train_test_split(
        numpy.arange(len(dataset)),
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=seed,
    )
    collate_fn = lambda x: tuple(i.to(device) for i in default_collate(x))
    train_loader = DataLoader(
        dataset,
        batch_size,
        collate_fn=collate_fn,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda items: tuple(
            item.to(device) for item in default_collate(items)
        ),
    )
