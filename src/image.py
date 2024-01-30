import numpy
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
