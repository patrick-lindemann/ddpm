import numpy
from torchvision import transforms

image_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),  # Create tensor with dimensions (C=3, W, H)
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale data between [-1., 1.]
    ]
)

tensor_to_image = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scale data between [0., 1.]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(
            lambda t: (t * 255.0).numpy().astype(numpy.uint8)
        ),  # Scale data between [0., 255.]
    ]
)
