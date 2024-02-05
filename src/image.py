import pathlib
from typing import Optional

import numpy
import PIL.Image
import torch
from torchvision import transforms

"""Constants"""


image_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)
"""
Transforms an image to a torch.Tensor with pixel values within the range [-1, 1].
"""

tensor_to_image = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: (t * 255.0).numpy().astype(numpy.uint8)),
    ]
)
"""
Transforms a torch.Tensor with pixel values within the range [-1, 1] to an image.
"""


"""Functions"""


def is_image_path(file_path: pathlib.Path) -> bool:
    """Check whether a file path is a PNG, JPEG, or JPG image.

    Parameters
    ----------
    file_path : pathlib.Path
        The file path.

    Returns
    -------
    bool
        True if the file path is an image, False otherwise.
    """
    return file_path.suffix in [".jpg", ".jpeg", ".png"]


def load_image(
    file_path: pathlib.Path, resize_to: Optional[int] = None
) -> torch.Tensor:
    """Load an image from a file path and transforms it to a torch.Tensor.

    Parameters
    ----------
    file_path : pathlib.Path
        The file path.
    resize_to : Optional[int], optional
        The size to which to resize the image, by default None.

    Returns
    -------
    torch.Tensor
        The image as a torch.Tensor.
    """
    assert file_path.exists()
    assert file_path.is_file()
    image = PIL.Image.open(file_path)
    transform = (
        transforms.Compose(
            [transforms.Resize(resize_to, antialias=True), image_to_tensor]
        )
        if resize_to is not None
        else image_to_tensor
    )
    image_transformed: torch.Tensor = transform(image)
    return image_transformed


def save_image(image: torch.Tensor, file_path: pathlib.Path) -> None:
    """Save a torch.Tensor to an image file

    Parameters
    ----------
    image : torch.Tensor
        The image as a torch.Tensor.
    file_path : pathlib.Path
        The output file path.
    """
    image = image.to("cpu")
    transform = transforms.Compose([tensor_to_image, transforms.ToPILImage()])
    image_transformed: PIL.Image.Image = transform(image)
    image_transformed.save(file_path)


def save_timeline(images: torch.Tensor, file_path: pathlib.Path) -> None:
    """Save a torch.Tensor timeline to to an image file

    Parameters
    ----------
    images : torch.Tensor
        The timeline as a torch.Tensor.
    file_path : pathlib.Path
        The output file path
    """
    timeline = torch.cat(torch.unbind(images, dim=0), dim=2)
    save_image(timeline, file_path)
