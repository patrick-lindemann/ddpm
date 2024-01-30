class DataLoader:
    pass


class CIFAR10(DataLoader):
    pass


class Flower102(DataLoader):
    pass


class CelebA(DataLoader):
    pass


class FGVCAircraft(DataLoader):
    pass


class QMNIST(DataLoader):
    pass


class LSUN(DataLoader):
    pass


import os
import pathlib
from typing import Callable, List, Literal, Optional, Tuple

import torch
from PIL import Image
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

from .transform import image_to_tensor


def load_dataset(
    name: Literal["cifar10", "mnist", "aircraft", "lsun", "celeba"],
    transform: Callable = image_to_tensor,
    download: bool = True,
    outdir: pathlib.Path = DATA_DIR / "datasets",
) -> VisionDataset:
    """_summary_

    Parameters
    ----------
    name : Literal["cifar10", "minst", "aircraft", "lsun", "celeba"]
        _description_
    transform : Callable, optional
        _description_, by default image_to_tensor()
    download : bool, optional
        _description_, by default True
    outdir : pathlib.Path, optional
        _description_, by default "../data/datasets/"

    Returns
    -------
    VisionDataset
        _description_

    Raises
    ------
    ValueError
        _description_
    """
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


def split_dataset(
    dataset: Dataset,
    train_size,
    random_state: int = 0,
) -> Tuple[List[int], List[int]]:
    """_summary_

    Parameters
    ----------
    dataset : Dataset
        _description_
    train_size : float, optional
        _description_, by default 0.8
    random_state : int, optional
        _description_, by default 0

    Returns
    -------
    Tuple[List[int], List[int]]
        _description_
    """
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.targets,
        stratify=dataset.targets,
        train_size=train_size,
        shuffle=True,
        random_state=random_state,
    )
    return train_indices, test_indices


def create_dataloader(
    dataset: Dataset,
    indices: Optional[torch.Tensor] = None,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> DataLoader:
    """_summary_

    Parameters
    ----------
    dataset : Dataset
        _description_
    indices : Optional[torch.Tensor], optional
        _description_, by default None
    batch_size : int, optional
        _description_, by default 16

    Returns
    -------
    DataLoader
        _description_
    """
    assert batch_size <= len(indices)
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda items: tuple(
            item.to(device) for item in default_collate(items)
        ),
    )
