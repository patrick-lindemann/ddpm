import json
import pathlib
from typing import Optional, Tuple

import torch
from torch.nn import functional
from tqdm import tqdm

from .model import DenoisingUNet2D
from .schedule import Schedule, get_schedule

"""Constants"""


CONFIG_FILE_NAME = "diffuser.config.json"
"""
The name of the file in which the diffuser configuration is saved.
"""


"""Classes"""


class GaussianDiffuser:
    """A Gaussian diffusor that implements the forward and reverse kernel of the DDPM framework."""

    time_steps: int
    schedule: Schedule
    device: torch.device

    _betas: torch.Tensor
    _alphas: torch.Tensor
    _sqrt_alphas: torch.Tensor
    _alphas_cumprod: torch.Tensor
    _alphas_cumprod_prev: torch.Tensor
    _sqrt_alphas_cumprod: torch.Tensor
    _sqrt_one_minus_alphas_cumprod: torch.Tensor
    _sqrt_recip_alphas_cumprod: torch.Tensor
    _sqrt_recip_alphas_cumprod_minus_one: torch.Tensor
    _posterior_variance: torch.Tensor
    _sqrt_posterior_variance: torch.Tensor

    @classmethod
    def load(
        cls, dir_path: pathlib.Path, time_steps: Optional[int] = None
    ) -> "GaussianDiffuser":
        """Load a pre-configured diffuser from a directory containing a diffuser.config.json file.

        Parameters
        ----------
        dir_path : pathlib.Path
            The diffuser directory.
        time_steps : Optional[int], optional
            The number of timesteps, by default None. If not specified, the number
            of timesteps will be loaded from the config file.

        Returns
        -------
        GaussianDiffuser
            The loaded diffuser.
        """
        config_path = dir_path / CONFIG_FILE_NAME
        assert config_path.exists()
        with open(config_path, "r") as file:
            config = json.load(file)
        schedule = get_schedule(**config["schedule"])
        return cls(
            time_steps if time_steps is not None else config["time_steps"], schedule
        )

    @torch.no_grad()
    def __init__(self, time_steps: int, schedule: Schedule) -> None:
        """Initialize a Gaussian diffuser.

        Parameters
        ----------
        time_steps : int
            The number of time steps.
        schedule : Schedule
            The schedule to use for the forward diffusion process.
        """
        self.time_steps = time_steps
        self.schedule = schedule
        self.device = torch.device("cpu")
        # Pre-calculate the diffusion parameters
        self._betas = schedule(torch.linspace(0.0, 1.0, time_steps)).to(self.device)
        self._alphas = 1.0 - self._betas
        self._sqrt_alphas = torch.sqrt(self._alphas)
        self._alphas_cumprod = torch.cumprod(self._alphas, axis=0)
        self._alphas_cumprod_prev = functional.pad(
            self._alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self._alphas_cumprod)
        self._sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self._alphas_cumprod)
        self._sqrt_recip_alphas_cumprod_minus_one = torch.sqrt(
            1.0 / self._alphas_cumprod - 1
        )
        self._posterior_variance = self._betas * (
            self._betas
            * (1.0 - self._alphas_cumprod_prev)
            / (1.0 - self._alphas_cumprod)
        )
        self._sqrt_posterior_variance = torch.sqrt(self._posterior_variance)

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse the input images using the forward kernel of the diffusion process.

        Parameters
        ----------
        images : torch.Tensor
            The images to diffuse with shape (N, 3, H, W).
        t : torch.Tensor
            A tensor with shape (N, 1)  containing the time step of the diffusion
            for each image. Values must be in the range [0, T].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the noised images and the applied noises.
        """
        noise = torch.randn_like(images, device=self.device)
        sqrt_alpha_cumprod_t = self._extract(self._sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self._sqrt_one_minus_alphas_cumprod, t
        )
        result = sqrt_alpha_cumprod_t * images + sqrt_one_minus_alpha_cumprod_t * noise
        return result, noise

    @torch.no_grad()
    def reverse(
        self,
        images: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the diffusion process to reconstruct the original images.

        Parameters
        ----------
        images : torch.Tensor
            The noised images with shape (N, 3, H, W).
        predicted_noise : torch.Tensor
            The noises predicted by the denoising model with shape (N, 3, H, W).
        t : torch.Tensor
            A tensor with shape (N, 1) containing the time step of the diffusion
            for each image. Values must be in the range [0, T].

        Returns
        -------
        torch.Tensor
            The reconstructed images.
        """
        sqrt_recip_alphas_cumprod_t = self._extract(self._sqrt_recip_alphas_cumprod, t)
        sqrt_recip_alphas_cumprod_minus_one_t = self._extract(
            self._sqrt_recip_alphas_cumprod_minus_one, t
        )
        result = (
            sqrt_recip_alphas_cumprod_t * images - predicted_noise
        ) / sqrt_recip_alphas_cumprod_minus_one_t
        return result

    @torch.no_grad()
    def sample(
        self, model: DenoisingUNet2D, num_images: int = 1, all_steps: bool = False
    ) -> torch.Tensor:
        """Sample images from a learned distribution using a trained denoising model
        and the reverse kernel of the diffusion process.

        Parameters
        ----------
        model : DenoisingUNet2D
            The trained denoising model.
        num_images : int, optional
            The number of images to generate, by default 1
        all_steps : bool, optional
            Whether to output all images of the diffusion process, by default False.
            If False, only the final images will be returned.

        Returns
        -------
        torch.Tensor
            A tensor containing the generated images with shape (N, 3, H, W) if all_steps
            is False, or (N, T, 3, H, W) otherwise.
        """
        model.train(False)  # Set the model to evaluation mode
        image_size = model.sample_size
        result = torch.zeros(
            (num_images, self.time_steps, 3, image_size, image_size),
            device=self.device,
        )
        images = torch.randn(
            (num_images, 3, image_size, image_size), device=self.device
        )
        for i in tqdm(range(self.time_steps), desc="sampling", leave=False):
            step = self.time_steps - i - 1
            t = torch.full((num_images,), step, dtype=torch.long, device=self.device)
            predicted_noise = model(images, t).sample
            images = self.reverse(images, predicted_noise, t)
            if step > 0:
                # Add random noise to the result
                noise = torch.randn_like(images, device=self.device)
                sqrt_posterior_variance_t = self._extract(
                    self._sqrt_posterior_variance, t
                )
                images += sqrt_posterior_variance_t * noise
            result[:, i] = images
        model.train(True)  # Reset the model to training mode
        return result if all_steps else result[:, -1]

    @torch.no_grad()
    def to(self, device: torch.device) -> "GaussianDiffuser":
        """Move the diffuser to a specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the diffuser to.

        Returns
        -------
        GaussianDiffuser
            The moved diffuser.
        """
        self.device = device
        self._betas = self._betas.to(device)
        self._alphas = self._alphas.to(device)
        self._sqrt_alphas = self._sqrt_alphas.to(device)
        self._alphas_cumprod = self._alphas_cumprod.to(device)
        self._alphas_cumprod_prev = self._alphas_cumprod_prev.to(device)
        self._sqrt_alphas_cumprod = self._sqrt_alphas_cumprod.to(device)
        self._sqrt_one_minus_alphas_cumprod = self._sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self._sqrt_recip_alphas_cumprod = self._sqrt_recip_alphas_cumprod.to(device)
        self._sqrt_recip_alphas_cumprod_minus_one = (
            self._sqrt_recip_alphas_cumprod_minus_one.to(device)
        )
        self._posterior_variance = self._posterior_variance.to(device)
        self._sqrt_posterior_variance = self._sqrt_posterior_variance.to(device)
        return self

    def save(self, dir_path: pathlib.Path) -> None:
        """Save the diffuser to a directory.

        Parameters
        ----------
        dir_path : pathlib.Path
            The directory to save the diffuser to.
        """
        config_path = dir_path / CONFIG_FILE_NAME
        config = {
            "time_steps": self.time_steps,
            "schedule": {
                "type": self.schedule.type,
                "start": self.schedule.start,
                "end": self.schedule.end,
                "tau": self.schedule.tau,
            },
        }
        if self.schedule.type == "linear":
            del config["schedule"]["tau"]
        with open(config_path, "w") as file:
            json.dump(config, file)

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Extract the values of a tensor at specified timesteps t.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to extract the values from with shape (T, ...).
        t : torch.Tensor
            The timesteps with shape (N, 1).

        Returns
        -------
        torch.Tensor
            The extracted values with shape (N, 1, 1, 1).
        """
        return tensor[t].reshape((t.shape[0], 1, 1, 1))
