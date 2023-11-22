from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Diffuser(ABC):
    """_summary_"""

    @abstractmethod
    def forward(
        self, images: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply noise to an image at a given time step of the forward diffusion process

        Parameters
        ----------
        images : torch.Tensor
            The images to apply the noise to. Has size N x C x H x W.
        t : torch.Tensor
            The time step to apply the noise at for each image in the range
            [0, num_steps). Has size N x 1, where N is the number of images.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the noised image and the noise.
        """
        raise NotImplementedError()
