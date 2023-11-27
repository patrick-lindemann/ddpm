import json
import pathlib

import numpy
import torch
from PIL.Image import Image
from torchvision import transforms

from diffusion.schedule import Scheduler


def save_image(
    file_path: pathlib.Path, image: Image | numpy.ndarray | torch.Tensor
) -> None:
    """_summary_

    Parameters
    ----------
    file_path : pathlib.Path
        _description_
    data : torch.Tensor
        _description_
    """
    if isinstance(image, numpy.ndarray) or isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    image.save(file_path)
