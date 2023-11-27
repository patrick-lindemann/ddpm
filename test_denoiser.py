import argparse
import json
import logging
import os
import pathlib

import torch

from diffusion.data import create_dataloader, load_dataset, plot_denoising_results
from diffusion.diffusion import GaussianDiffuser
from diffusion.model import BasicUNet
from diffusion.schedule import (
    CosineScheduler,
    LinearScheduler,
    PolynomialScheduler,
    Scheduler,
    SigmoidScheduler,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=pathlib.Path,
        help="The path to the model to test.\nThe model directory needs to contain a model.pt and a metadata.json file.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        help="The time step to evaluate the denoiser at.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.\nAllowed values: "CPU", "Cuda".',
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        help="The directory to save the results to.",
        default="./out/test",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()

    # Prepare the logger
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Prepare the output directory
    if not args.outdir.exists():
        os.makedirs(args.outdir)

    # Load the model
    model_path = args.model_dir / "model.pt"
    logging.info(f"Loading model from {model_path}.")
    model = BasicUNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_path))

    # Load the metadata
    metadata_path = args.model_dir / "metadata.json"
    logging.info(f"Loading metadata from {metadata_path}.")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    # Prepare the scheduler
    scheduler: Scheduler
    if metadata["schedule"]["type"] == "linear":
        scheduler = LinearScheduler(
            start=metadata["schedule"]["start"], end=metadata["schedule"]["end"]
        )
    elif metadata["schedule"]["type"] == "polynomial":
        scheduler = PolynomialScheduler(
            start=metadata["schedule"]["start"],
            end=metadata["schedule"]["end"],
            tau=metadata["schedule"]["tau"],
        )
    elif metadata["schedule"]["type"] == "cosine":
        scheduler = CosineScheduler(
            start=metadata["schedule"]["start"],
            end=metadata["schedule"]["end"],
            tau=metadata["schedule"]["tau"],
        )
    elif metadata["schedule"]["type"] == "sigmoid":
        scheduler = SigmoidScheduler(
            start=metadata["schedule"]["start"],
            end=metadata["schedule"]["end"],
            tau=metadata["schedule"]["tau"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {metadata['schedule']['type']}")

    # Prepare the diffuser
    diffuser = GaussianDiffuser(
        num_steps=metadata["schedule"]["steps"], scheduler=scheduler
    )

    # Load the data
    logging.info("Loading dataset.")
    dataset = load_dataset(metadata["dataset"]["name"])
    test_loader = create_dataloader(
        dataset,
        indices=metadata["dataset"]["indices"]["test"],
        batch_size=metadata["dataset"]["batch_size"],
    )

    # Test the model
    image_batch, _ = next(iter(test_loader))
    image = image_batch[0].unsqueeze(0)
    t = (
        torch.tensor([args.time_step])
        if args.time_step is not None
        else torch.randint(0, metadata["schedule"]["steps"], size=(1,))
    )
    image_noised, noise = diffuser.forward(image, t)
    with torch.no_grad():
        noise_predicted = model.forward(image_noised)
        image_restored = torch.clamp(image_noised - noise_predicted, -1.0, 1.0)
    # Plot the results
    plot_denoising_results(
        t=t,
        image_noised=image_noised,
        image_restored=image_restored,
        noise=noise,
        noise_predicted=noise_predicted,
        file_path=args.outdir / "test-result.svg",
    )
