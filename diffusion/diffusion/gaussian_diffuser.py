from typing import Tuple

import torch

from diffusion.schedule import LinearScheduler, Scheduler

from .diffuser import Diffuser


class GaussianDiffuser(Diffuser):
    """_summary_"""

    num_steps: int

    _betas: torch.Tensor
    _alphas: torch.Tensor
    _alpha_hats: torch.Tensor

    def __init__(
        self,
        num_steps: int,
        scheduler: Scheduler = LinearScheduler(),
    ) -> None:
        """_summary_

        Parameters
        ----------
        num_steps : int
            _description_
        scheduler : Scheduler, optional
            _description_, by default LinearScheduler()
        """
        self.num_steps = num_steps
        self._betas = scheduler(torch.linspace(0.0, 1.0, num_steps))
        self._alphas = 1.0 - self._betas
        # self._alphas = torch.clip(1.0 - self._betas, 1e-9, 9.9999e-1)
        self._alpha_hats = torch.cumprod(self._alphas, axis=0)

    def forward(
        self, images: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random noise for the image
        noise = torch.randn_like(images)
        # Evaluate the gamma function at the given time step for each image
        N = images.shape[0]
        gamma_t = self._alpha_hats[t].reshape(shape=[N, 1, 1, 1])
        # Apply the noise to the image
        result = torch.sqrt(gamma_t) * images + torch.sqrt(1.0 - gamma_t) * noise
        result = torch.clamp(result, -1.0, 1.0)
        return result, noise
