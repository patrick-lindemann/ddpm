from typing import Tuple

import torch


class Scheduler:
    """_summary_"""

    alphas: torch.Tensor
    alpha_hats: torch.Tensor

    def __init__(self, alphas: torch.Tensor) -> None:
        self.alphas = alphas
        self.alpha_hats = torch.cumprod(self.alphas, axis=0)

    @property
    def num_steps(self) -> int:
        return self.betas.shape[0]

    def apply(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply noise to an image at a given time step of the forward diffusion process

        Parameters
        ----------
        x : torch.Tensor
            The images to apply the noise to. Has size N x C x H x W.
        t : torch.Tensor
            The time step to apply the noise at for each image in the range
            [0, num_steps). Has size N x 1, where N is the number of images.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the noised image and the noise.
        """
        # Generate random noise for the image
        # FIXME: This is not normalized and adds weird noise
        noise = torch.randn_like(x)
        # Evaluate the gamma function at the given time step for each image
        N = x.shape[0]
        gamma_t = self.alpha_hats[t].reshape(shape=[N, 1, 1, 1])
        # Apply the noise to the image
        result = torch.sqrt(gamma_t) * x + torch.sqrt(1.0 - gamma_t) * noise
        return result, noise
