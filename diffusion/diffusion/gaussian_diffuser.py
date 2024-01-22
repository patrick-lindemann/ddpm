from typing import Tuple

import torch

from diffusion.schedule import LinearScheduler, Scheduler

from .diffuser import Diffuser


class GaussianDiffuser(Diffuser):
    """_summary_"""

    num_steps: int

    _betas: torch.Tensor
    _betas_sqrt: torch.Tensor
    _alphas: torch.Tensor
    _alphas_sqrt: torch.Tensor
    _alpha_hats: torch.Tensor
    _alpha_hats_sqrt: torch.Tensor
    _one_minus_alpha_hats_sqrt: torch.Tensor

    @torch.no_grad()
    def __init__(
        self,
        num_steps: int,
        scheduler: Scheduler = LinearScheduler(),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """_summary_

        Parameters
        ----------
        num_steps : int
            _description_
        scheduler : Scheduler, optional
            _description_, by default LinearScheduler()
        device : torch.device, optional
            _description_, by default torch.device("cpu")
        """
        super().__init__(device)
        self.num_steps = num_steps
        self._betas = scheduler(torch.linspace(0.0, 1.0, num_steps)).to(self.device)
        self._betas_sqrt = torch.sqrt(self._betas)
        self._alphas = 1.0 - self._betas
        self._alphas_sqrt = torch.sqrt(self._alphas)
        self._alpha_hats = torch.cumprod(self._alphas, axis=0)
        self._alpha_hats_sqrt = torch.sqrt(self._alpha_hats)
        self._one_minus_alpha_hats_sqrt = torch.sqrt(1.0 - self._alpha_hats)

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Parameters
        ----------
        images : torch.Tensor
            The images to be diffused.
        t : torch.Tensor
            The time step of the diffusion for each image.
            Must be in the range [0, num_steps].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the noised images and the applied noises.
        """
        # Generate random noise for the image
        noise = torch.randn_like(images, device=self.device)
        # Evaluate the gamma function at the given time step for each image
        N = images.shape[0]
        gamma_sqrt_t = self._alpha_hats_sqrt[t].reshape(shape=[N, 1, 1, 1])
        one_minus_gamma_sqrt_t = self._one_minus_alpha_hats_sqrt[t].reshape(
            shape=[N, 1, 1, 1]
        )
        # Apply the noise to the image
        result = gamma_sqrt_t * images + one_minus_gamma_sqrt_t * noise
        result = torch.clamp(result, -1.0, 1.0)
        return result, noise

    @torch.no_grad()
    def reverse(
        self,
        noised_images: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the diffusion process to reconstruct the original images.

        Parameters
        ----------
        noised_images : torch.Tensor
            The noised images after the diffusion process.
        predicted_noise : torch.Tensor
            The predicted noise from the diffusion model.
        t : torch.Tensor
            The time step for each image.

        Returns
        -------
        torch.Tensor
            The reconstructed images.
        """
        # Evaluate the gamma function at the given time step for each image
        N = noised_images.shape[0]
        gamma_sqrt_t = self._alpha_hats[t].reshape(shape=[N, 1, 1, 1])
        one_minus_gamma_sqrt_t = self._one_minus_alpha_hats_sqrt[t].reshape(
            shape=[N, 1, 1, 1]
        )
        # Reverse the diffusion process
        reconstructed_images = (
            noised_images - one_minus_gamma_sqrt_t * predicted_noise
        ) / gamma_sqrt_t
        reconstructed_images = torch.clamp(reconstructed_images, -1.0, 1.0)
        return reconstructed_images

    @torch.no_grad()
    def sample(self, x_t: torch.Tensor, t: torch.Tensor, prediction: torch.Tensor):
        """_summary_

        Parameters
        ----------
        x_t : torch.Tensor
            _description_
        t : torch.Tensor
            _description_
        prediction : torch.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Calculate x_(t-1)
        result = (1 / self._alphas_sqrt[t]) * (
            x_t - ((self._betas[t] * prediction) / self._one_minus_alpha_hats_sqrt[t])
        )
        if t == 0:
            # Timestep is 0, return the result as is
            return result
        # Add random noise to the result
        noise = torch.randn_like(x_t, device=self.device)
        result += self._betas_sqrt[t] * noise
        return result
