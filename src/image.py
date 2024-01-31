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

tensor_to_image = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: (t * 255.0).numpy().astype(numpy.uint8)),
    ]
)


"""Functions"""


def is_image_path(file_path: pathlib.Path) -> bool:
    return file_path.suffix in [".jpg", ".jpeg", ".png"]


def load_image(
    file_path: pathlib.Path, resize_to: Optional[int] = None
) -> torch.Tensor:
    assert file_path.exists()
    assert file_path.is_file()
    image = PIL.Image.open(file_path)
    transform = (
        transforms.Compose(
            [image_to_tensor, transforms.Resize(resize_to, antialias=True)]
        )
        if resize_to is not None
        else image_to_tensor
    )
    image_transformed: torch.Tensor = transform(image)
    return image_transformed


def save_image(image: torch.Tensor, file_path: pathlib.Path) -> None:
    image = image.to("cpu")
    transform = transforms.Compose([tensor_to_image, transforms.ToPILImage()])
    image_transformed: PIL.Image.Image = transform(image)
    image_transformed.save(file_path)
