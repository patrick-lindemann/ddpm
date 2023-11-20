from typing import Tuple

import numpy
import torch
from torchvision import transforms


def transform_image(image: torch.Tensor, size: Tuple[int] = (128, 128)) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize(size),  # Resize image
            transforms.ToTensor(),  # Create tensor with dimensions (C=3, W, H)
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale data between [-1., 1.]
        ]
    )(image)


def reverse_transform_image(image: torch.Tensor) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),  # Scale data between [0., 1.]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(
                lambda t: (t * 255.0).numpy().astype(numpy.uint8)
            ),  # Scale data between [0., 255.]
        ]
    )(image)
