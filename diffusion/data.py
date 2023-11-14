import os
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
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
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Parameters
    ----------
    dataset : Literal[&quot;cifar10&quot;, &quot;lsun&quot;, &quot;celeba&quot;]
        _description_
    transformation : Optional[Callable], optional
        _description_, by default None
    n_train : Optional[int], optional
        _description_, by default None
    n_test : Optional[int], optional
        _description_, by default None

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Get the VisionDataset class corresponding to the dataset name
    if dataset not in DATASETS:
        raise ValueError(f'Invalid dataset specified: "{dataset}"')
    dataset_class = DATASETS[dataset]
    # Load the dataset and transform it
    data = dataset_class(root=DATA_DIR, download=True, transform=transformation)
    # Split the dataset into train and test sets
    train_set, val_set = torch.utils.data.random_split(data, [n_train, n_test])
    return train_set, val_set
