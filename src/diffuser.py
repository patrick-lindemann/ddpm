import json
import pathlib
from typing import Tuple

import torch
from tqdm import tqdm

from .model import DenoisingUNet2D
from .schedule import Schedule, get_schedule

"""Constants"""


CONFIG_FILE_NAME = "diffuser.config.json"


"""Classes"""


class GaussianDiffuser:
    """_summary_"""

    time_steps: int
    schedule: Schedule
    device: torch.device

    _beta: torch.Tensor
    _sqrt_beta: torch.Tensor
    _alpha: torch.Tensor
    _sqrt_alpha: torch.Tensor
    _alpha_hat: torch.Tensor
    _sqrt_alpha_hat: torch.Tensor
    _sqrt_one_minus_alpha_hat: torch.Tensor

    @classmethod
    def load(cls, dir_path: pathlib.Path) -> "GaussianDiffuser":
        config_path = dir_path / CONFIG_FILE_NAME
        assert config_path.exists()
        with open(config_path, "r") as file:
            config = json.load(file)
        schedule = get_schedule(**config["schedule"])
        return cls(config["time_steps"], schedule)

    @torch.no_grad()
    def __init__(self, time_steps: int, schedule: Schedule) -> None:
        self.time_steps = time_steps
        self.schedule = schedule
        # Pre-calculate the diffusion parameters
        self._beta = schedule(torch.linspace(0.0, 1.0, time_steps))
        self._sqrt_beta = torch.sqrt(self._beta)
        self._alpha = 1.0 - self._beta
        self._sqrt_alpha = torch.sqrt(self._alpha)
        self._alpha_hat = torch.cumprod(self._alpha, dim=0)
        self._sqrt_alpha_hat = torch.sqrt(self._alpha_hat)
        self._sqrt_one_minus_alpha_hat = torch.sqrt(1 - self._alpha_hat)

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
        N = images.shape[0]
        noise = torch.randn_like(images, device=self.device)
        sqrt_alpha_t = self._sqrt_alpha_hat[t].reshape([N, 1, 1, 1])
        sqrt_one_minus_alpha_t = self._sqrt_one_minus_alpha_hat[t].reshape([N, 1, 1, 1])
        result = sqrt_alpha_t * images + sqrt_one_minus_alpha_t * noise
        result = torch.clamp(result, -1.0, 1.0)
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
            The noised images.
        predicted_noise : torch.Tensor
            The noise predicted by the denoising model.
        t : torch.Tensor
            The time step for each image.

        Returns
        -------
        torch.Tensor
            The reconstructed images.
        """
        N = images.shape[0]
        sqrt_alpha_t = self._sqrt_alpha[t].reshape(shape=[N, 1, 1, 1])
        beta_t = self._beta[t].reshape(shape=[N, 1, 1, 1])
        sqrt_one_minus_alpha_hat_t = self._sqrt_one_minus_alpha_hat[t].reshape(
            shape=[N, 1, 1, 1]
        )
        result = (
            images - ((beta_t / sqrt_one_minus_alpha_hat_t) * predicted_noise)
        ) / sqrt_alpha_t
        result = torch.clamp(result, -1.0, 1.0)
        return result

    @torch.no_grad()
    def sample(
        self, model: DenoisingUNet2D, N: int = 1, all_steps: bool = False
    ) -> torch.Tensor:
        """__summary__

        Parameters
        ----------
        model : DenoisingUNet2D
            _description_
        N : int, optional
            _description_, by default 1
        all_steps : bool, optional
            _description_, by default True

        Returns
        -------
        torch.Tensor
            _description_
        """
        model.train(False)  # Set the model to evaluation mode
        image_size = model.sample_size
        result = torch.zeros(
            (self.time_steps, N, 3, image_size, image_size), device=self.device
        )
        images = torch.randn((N, 3, image_size, image_size), device=self.device)
        for step in reversed(
            tqdm(range(0, self.time_steps), desc="sampling", leave=False)
        ):
            t = torch.full((N,), step, dtype=torch.long, device=self.device)
            predicted_noise = model(images, t).sample
            images = self.reverse(images, predicted_noise, t)
            if step > 0:
                # Add random noise to the result
                noise = torch.randn_like(images, device=self.device)
                beta_sqrt_t = self._sqrt_beta[t].reshape(shape=[N, 1, 1, 1])
                images += beta_sqrt_t * noise
            result[step] = images
        model.train(True)  # Reset the model to training mode
        return result if all_steps else result[-1]

    @torch.no_grad()
    def to(self, device: torch.device) -> "GaussianDiffuser":
        """_summary_

        Parameters
        ----------
        device : torch.device
            _description_

        Returns
        -------
        GaussianDiffuser
            _description_
        """
        self.device = device
        self._beta = self._beta.to(device)
        self._sqrt_beta = self._sqrt_beta.to(device)
        self._alpha = self._alpha.to(device)
        self._sqrt_alpha = self._sqrt_alpha.to(device)
        self._alpha_hat = self._alpha_hat.to(device)
        self._sqrt_alpha_hat = self._sqrt_alpha_hat.to(device)
        self._sqrt_one_minus_alpha_hat = self._sqrt_one_minus_alpha_hat.to(device)
        return self

    def save(self, dir_path: pathlib.Path) -> None:
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
