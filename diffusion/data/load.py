import os
import pathlib
from typing import Callable, List, Literal, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUN, CelebA, VisionDataset


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


def load_dataset(
    name: Literal["cifar10", "lsun", "celeba"],
    transform: Optional[Callable] = transforms.ToTensor(),
    download: bool = True,
    outdir: pathlib.Path = pathlib.Path("../data/datasets/"),
) -> VisionDataset:
    """_summary_

    Parameters
    ----------
    name : Literal[&quot;cifar10&quot;, &quot;lsun&quot;, &quot;celeba&quot;]
        _description_
    transform : Optional[Callable], optional
        _description_, by default transforms.ToTensor()
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
    # Generate the train and test indices
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
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size)
