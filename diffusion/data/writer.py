import os
import pathlib

import torch
from PIL import Image
from torchvision import transforms


def save_image(file_path: pathlib.Path, data: torch.Tensor) -> None:
    """_summary_

    Parameters
    ----------
    file_path : pathlib.Path
        _description_
    data : torch.Tensor
        _description_
    """
    # Ensure the output directory exists
    if not file_path.parent.exists():
        os.makedirs(file_path.parent)
    # Transform the tensor to an image and save it
    image: Image.Image = transforms.ToPILImage()(data)
    image.save(file_path)
