import os
import pathlib
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUN, CelebA, VisionDataset

DATA_DIR = os.path.abspath("../data/")

DATASETS: Dict[str, VisionDataset] = {
    "cifar10": CIFAR10,
    "lsun": LSUN,
    "celeba": CelebA,
}


def load_data(
    dataset: Literal["cifar10", "lsun", "celeba"],
    transformation: Optional[Callable] = transforms.ToTensor(),
    train_ratio: float = 0.8,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """_summary_

    Parameters
    ----------
    name : Literal["cifar10", "lsun", "celeba"]
        _description_
    transformation : Optional[Callable], optional
        _description_, by default None
    train_ratio : float, optional
        _description_, by default 0.8
    batch_size : int, optional
        _description_, by default 16

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Get the VisionDataset class corresponding to the dataset name
    if dataset not in DATASETS:
        raise ValueError(f'Invalid dataset specified: "{dataset}"')
    dataclass = DATASETS[dataset]
    # Download the dataset and transform it
    data: VisionDataset = dataclass(
        root=DATA_DIR, download=True, transform=transformation
    )
    # Determine the sizes and indices of the train and test sets
    n = len(data)
    n_train = int(train_ratio * n)
    n_test = n - n_train
    train_indices, test_indices = random_split(torch.arange(n), [n_train, n_test])
    # Create the train and test data loaders
    train_loader = DataLoader(
        data, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)
    )
    test_loader = DataLoader(
        data, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices)
    )
    return train_loader, test_loader


def load_image(
    file_path: pathlib.Path,
    transformation: Optional[Callable] = transforms.ToTensor(),
) -> torch.Tensor:
    """_summary_

    Parameters
    ----------
    file_path : pathlib.Path
        _description_
    transformation : Optional[Callable], optional
        _description_, by default None

    Returns
    -------
    torch.Tensor
        _description_
    """
    image = Image.open(file_path)
    tensor = transformation(image)
    return tensor
