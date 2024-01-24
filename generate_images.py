import argparse
import json
import logging
import os
import pathlib

import torch
from tqdm import tqdm

from diffusion.data.save import save_image
from diffusion.diffusion import GaussianDiffuser
from diffusion.model import DiffusionModel
from diffusion.paths import OUT_DIR
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
        "--num-images",
        type=int,
        help="The number of images to generate.",
        default=10,
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
        default=OUT_DIR / "generated",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()
    device = torch.device(args.device)
    args.outdir = args.outdir / args.model_dir.name

    # Prepare the logger
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load the metadata
    metadata_path = args.model_dir / "metadata.json"
    logging.info(f"Loading metadata from {metadata_path}.")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    # Load the trained model
    model_path = args.model_dir / "model.pt"
    logging.info(f"Loading model from {model_path}.")
    model = DiffusionModel(**metadata["model"])
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Prepare the output directory
    if not args.outdir.exists():
        os.makedirs(args.outdir)

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
    num_steps = metadata["schedule"]["steps"]
    diffuser = GaussianDiffuser(num_steps=num_steps, scheduler=scheduler, device=device)

    # Generate the images
    logging.info(f"Generating {args.num_images} images.")
    sample_size = metadata["model"]["sample_size"]
    with torch.no_grad():
        images = torch.zeros(
            (args.num_images, 3, sample_size, sample_size), device=device
        )
        for index in tqdm(range(args.num_images)):
            for i in reversed(range(0, num_steps)):
                # Generate random noise
                noise = torch.randn((1, 3, sample_size, sample_size), device=device)
                t = torch.full((1,), i, dtype=torch.long, device=device)
                prediction = model(noise, t).sample
                image = diffuser.sample(noise, t, prediction)
                image = torch.clamp(image, -1.0, 1.0)
                images[index] = image.squeeze()
            break

    # Save the images
    logging.info(f"Saving images to {args.outdir}.")
    for i, image in enumerate(images):
        save_image(args.outdir / f"{i}.png", image)
    logging.info("Done.")
