import os
import pathlib
from typing import Literal, Optional, Tuple

import numpy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    QMNIST,
    CelebA,
    FGVCAircraft,
    Flowers102,
    VisionDataset,
)

from .image import image_to_tensor
from .paths import DATA_DIR

"""Types"""


AvailableDataset = Literal["cifar10", "mnist", "aircraft", "flower", "celeba"]
"""
The names of the available datasets.
"""


"""Functions"""


def load_dataset(
    name: AvailableDataset,
    resize_to: Optional[int] = None,
    out_dir: pathlib.Path = DATA_DIR / "datasets",
    **kwargs,
) -> VisionDataset:
    """Loads a dataset from torchvision.datasets and applies a set of transformations.

    Parameters
    ----------
    name : AvailableDataset
        The name of the dataset to load.
    resize_to : Optional[int], optional
        The size to which images should be resized, by default None
    out_dir : pathlib.Path, optional
        The directory to which the dataset should be downloaded, by default <data_dir> / "datasets"

    Returns
    -------
    VisionDataset
        The loaded dataset.

    Raises
    ------
    ValueError
        If an invalid dataset name is specified.
    """
    dataset = None
    if name == "cifar10":
        dataset = CIFAR10
    elif name == "mnist":
        dataset = QMNIST
    elif name == "aircraft":
        dataset = FGVCAircraft
    elif name == "flower":
        dataset = Flowers102
    elif name == "celeba":
        dataset = CelebA
    else:
        raise ValueError(f'Invalid dataset specified: "{name}".')
    out_dir = out_dir / name
    if not out_dir.exists():
        os.makedirs(out_dir)
    return dataset(
        **kwargs,
        root=out_dir,
        download=True,
        transform=transforms.Compose([image_to_tensor, transforms.Resize(resize_to)]),
    )


def create_dataloaders(
    dataset: Dataset,
    train_size: float | int,
    test_size: float | int,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader]:
    """Creates dataloaders for training and testing a model.

    Parameters
    ----------
    dataset : Dataset
        The dataset.
    train_size : float | int
        The size of the training set, as a fraction of the dataset size or an integer number of samples.
    test_size : float | int
        The size of the test set, as a fraction of the dataset size or an integer number of samples.
    batch_size : int, optional
        The batch size with which the images are loaded, by default 16
    shuffle : bool, optional
        Whether to shuffle the dataset, by default True
    seed : Optional[int], optional
        The random seed to use for shuffling, by default None
    device : torch.device, optional
        The device on which to load the images onto, by default torch.device("cpu")

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The training and testing dataloaders.
    """
    if train_size > 1.0:
        assert isinstance(train_size, int)
        assert 0 <= train_size <= len(dataset)
        train_size = float(train_size) / len(dataset)
    if test_size > 1.0:
        assert isinstance(test_size, int)
        assert 0 <= test_size <= len(dataset)
        test_size = float(test_size) / len(dataset)
    assert train_size + test_size <= 1.0
    train_indices, test_indices = train_test_split(
        numpy.arange(len(dataset)),
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        random_state=seed,
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    collate_fn = lambda X: tuple(x.to(device) for x in default_collate(X))
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    return train_dataloader, test_dataloader
