import pathlib
from typing import Callable, Optional

import torch
from torchvision import transforms

from .transform import tensor_to_image


def save_image(
    file_path: pathlib.Path,
    image: torch.Tensor,
    transform: Optional[Callable] = tensor_to_image,
) -> None:
    """_summary_

    Parameters
    ----------
    file_path : pathlib.Path
        _description_
    data : torch.Tensor
        _description_
    transform : Optional[Callable], optional
        _description_, by default tensor_to_image
    """
    transform = transforms.Compose([transform, transforms.ToPILImage()])
    image_transformed = transform(image)
    image_transformed.save(file_path)
