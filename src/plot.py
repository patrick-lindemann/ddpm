import pathlib
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch

from .image import tensor_to_image
from .schedule import Schedule


def plot_image(
    image: torch.Tensor,
    file_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot an image tensor using matplotlib.

    Parameters
    ----------
    image : torch.Tensor
        The image tensor.
    file_path : Optional[pathlib.Path], optional
        The plot output path, by default None. If not specified, the image will be displayed.
    """
    if file_path is not None:
        plt.imsave(file_path, tensor_to_image(image))
    else:
        plt.imshow(tensor_to_image(image))


def plot_schedule(
    schedule: Schedule,
    time_steps: int = 1000,
    file_path: Optional[pathlib.Path] = None,
):
    """Plot a schedule using matplotlib.

    Parameters
    ----------
    schedule : Schedule
        The schedule to plot.
    time_steps : int, optional
        The number of time steps to plot the schedule for, by default 1000
    file_path : Optional[pathlib.Path], optional
        The plot output path, by default None. If not specified, the plot will be displayed.
    """
    title = (
        f"{schedule.type.capitalize()} Scheduler\n s={schedule.start}, e={schedule.end}"
    )
    if schedule.type != "linear":
        title += f", $\\tau$={schedule.tau}"
    plt.title(title)
    plt.xlabel("$t$")
    plt.ylabel("$\\gamma(t)$")
    plt.gca().set_aspect("equal", adjustable="box")
    t = torch.linspace(0, 1, time_steps)
    plt.plot(t, schedule(t))
    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def plot_loss(
    losses: torch.Tensor,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    file_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot a loss function over epochs.

    Parameters
    ----------
    losses : torch.Tensor
        The loss values.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis, by default "Epoch".
    ylabel : str, optional
        The label for the y axis, by default "Loss".
    file_path : Optional[pathlib.Path], optional
        The plot output path, by default None. If not specified, the plot will be displayed.
    """
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(losses)
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()


def plot_denoising_results(
    t: torch.Tensor,
    image_noised: torch.Tensor,
    image_restored: torch.Tensor,
    noise: torch.Tensor,
    noise_predicted: torch.Tensor,
    file_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot the results of an image denoising experiment.

    Parameters
    ----------
    t : torch.Tensor
        The time step of the diffusion for each image.
    image_noised : torch.Tensor
        The noisy image tensor.
    image_restored : torch.Tensor
        The restored image tensor.
    noise : torch.Tensor
        The ground truth noise tensor.
    noise_predicted : torch.Tensor
        The predicted noise tensor.
    file_path : Optional[pathlib.Path], optional
        The plot output path, by default None. If not specified, the plot will be displayed.
    """
    # Prepare the figure
    plt.figure(figsize=(30, 30))
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    fig.suptitle(f"Image Denoising at t=${t[0]}$", fontsize=14)
    fig.tight_layout()
    for ax in axes.ravel():
        ax.axis("off")
    # First subplot
    ax0 = axes[0, 0]
    ax0.imshow(tensor_to_image(image_noised.squeeze()))
    ax0.set_title(f"Noisy Image at Time ${t[0]}$", fontsize=10)
    # Second subplot
    ax1 = axes[0, 1]
    ax1.imshow(tensor_to_image(image_restored.squeeze()))
    ax1.set_title(f"Restored Image", fontsize=10)
    # Third subplot
    ax2 = axes[1, 0]
    ax2.imshow(tensor_to_image(noise.squeeze()))
    ax2.set_title(f"Ground Truth Noise", fontsize=10)
    # Fourth subplot
    ax3 = axes[1, 1]
    ax3.imshow(tensor_to_image(noise_predicted.squeeze()))
    ax3.set_title(f"Predicted Noise", fontsize=10)
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()
