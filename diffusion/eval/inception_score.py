from typing import Optional

import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.inception import inception_v3


def calculate_inception_score(
    x: torch.Tensor,
    batch_size: Optional[int] = None,
    num_splits: int = 1,
    device: torch.device = torch.cpu,
) -> torch.Tensor:
    """Computes the inception score of a generated image
    Adapted from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

    Parameters
    ----------
    x : torch.Tensor
        The dataset containing N images with dimension (C, H, W) which are normalized
        in the range [-1, 1]
    batch_size : int, optional
        The image batch size for InceptionV3, by default None. If None, the batch size
        is set to N
    num_splits : int, optional
        The number of splits, by default 1
    device : torch.device, optional
        The device, by default torch.cpu

    Returns
    -------
    torch.Tensor
        A tensor of length N containing the inception score for each image
    """
    N = len(x)
    batch_size = batch_size or N
    # Prepare the data loader and transformation
    transform = transforms.Resize(
        299, interpolation=transforms.InterpolationMode.BILINEAR
    )  # Inception v3 expects a images of shape (N, 3, 299, 299)
    data_loader = DataLoader(
        x, batch_size=batch_size, transform=transform, device=device
    )
    # Load the inception v3 model
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    # Prepare the preduction function
    predictions: torch.Tensor = torch.zeros((N, 1000), device=device)
    for i, batch in enumerate(data_loader):
        prediction = F.softmax(inception(batch), dim=1)
        predictions[i * batch_size, i * batch_size + batch_size] = prediction
    # Compute the scores
    scores: torch.Tensor = torch.zeros(num_splits, device=device)
    for k in range(num_splits):
        part = predictions[k * (N // num_splits) : (k + 1) * (N // num_splits), :]
        py = torch.mean(part, dim=0)
        scores[k] = torch.exp(torch.mean(torch.sum(part * torch.log(part / py), dim=1)))
    # Calculate and the means and standard deviations of the scores
    means = torch.mean(scores)
    stds = torch.std(scores)
    return means, stds
