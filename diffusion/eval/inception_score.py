from typing import Optional

import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.inception import Inception_V3_Weights, inception_v3
from tqdm import tqdm


@torch.no_grad()
def calculate_inception_score(
    x: torch.Tensor,
    batch_size: int = 32,
    num_splits: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Computes the inception score of a generated image
    Adapted from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

    Parameters
    ----------
    x : torch.Tensor
        The dataset containing N images with dimension (C, H, W) which are normalized
        in the range [-1, 1]
    batch_size : int, optional
        The image batch size for InceptionV3, by default 32
    num_splits : int, optional
        The number of splits, by default 1
    device : torch.device, optional
        The device, by default torch.device("cpu")

    Returns
    -------
    torch.Tensor
        A tuple containing the mean and standard deviation of the inception scores
    """
    N = len(x)
    batch_size = batch_size or N
    # Inception v3 expects a images of shape (N, 3, 299, 299)
    # Resize the image to (299, 299)
    resize_transform = transforms.Resize((299, 299), antialias=True)
    x_resized = resize_transform(x)
    # Prepare the data loader
    data_loader = DataLoader(x_resized, batch_size=batch_size)
    # Load the inception v3 model
    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
    )
    inception.eval()
    # Prepare the preduction function
    predictions: torch.Tensor = torch.zeros((N, 1000), device=device)

    for i, batch in enumerate(tqdm(data_loader)):
        prediction = F.softmax(inception(batch), dim=1)
        predictions[i * batch_size : (i + 1) * batch_size] = prediction
    # Compute the scores over the number of dataset splits
    scores: torch.Tensor = torch.zeros(num_splits, device=device)
    for k in range(num_splits):
        subset = predictions[k * (N // num_splits) : (k + 1) * (N // num_splits), :]
        py = torch.mean(subset, dim=0)
        scores[k] = torch.exp(
            torch.mean(torch.sum(subset * torch.log(subset / py), dim=1))
        )
    return torch.mean(scores), torch.std(scores)
