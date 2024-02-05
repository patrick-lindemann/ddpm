import argparse
import logging
import os
import pathlib

import torch
from tqdm import tqdm

from src.diffuser import GaussianDiffuser
from src.image import save_image, save_timeline
from src.model import DenoisingUNet2D


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=pathlib.Path,
        help="The path of the run directory.",
    )
    parser.add_argument(
        "num_images",
        type=int,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size for generating images. Must be a divisor of num_images.",
        default=None,
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        help="The number of time steps for the diffusion process. If not specified, the value from the run directory is used.",
        default=None,
    )
    parser.add_argument(
        "--show-process",
        action="store_true",
        help="Show the process of the image generation by exporting the timelines.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use. If not specified, the randomized calculations will be non-deterministic.",
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="The directory to save the results to. If not specified, the images are saved to <run_dir>/samples.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.Allowed values: "cpu", "cuda".',
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()
    num_images: int = args.num_images
    batch_size: int = args.batch_size if args.batch_size is not None else num_images
    time_steps: int | None = args.time_steps
    show_process: bool = args.show_process
    num_batches = num_images // batch_size
    run_dir: pathlib.Path = args.run_dir
    run_name = run_dir.name
    seed: int = args.seed
    out_dir = args.out_dir if args.out_dir is not None else run_dir / "samples"
    device = torch.device(args.device)
    verbose: bool = args.verbose
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if seed is not None:
        torch.manual_seed(seed)

    # Validate the arguments
    assert run_dir.exists()
    assert batch_size <= num_images
    assert num_images % batch_size == 0

    # Prepare the diffuser
    logging.info(f'Loading model from "{run_dir}".')
    model = DenoisingUNet2D.load(run_dir).to(device)
    diffuser = GaussianDiffuser.load(run_dir, time_steps=time_steps).to(device)

    # Generate and save the images
    logging.info(f"Generating {num_images} images to dir {out_dir}.")
    logging.debug(f"Using {num_batches} batches with size {batch_size}.")
    if not out_dir.exists():
        os.makedirs(out_dir)
    for _ in tqdm(range(num_batches)):
        images: torch.Tensor = diffuser.sample(
            model, batch_size, all_steps=show_process
        )
        for i, image in enumerate(images):
            image_index = i * num_batches
            if show_process:
                timeline = images[i]
                image = timeline[-1]
                save_timeline(timeline, out_dir / f"{image_index}_timeline.png")
                save_image(image, out_dir / f"{image_index}.png")
            else:
                image = images[i]
                save_image(image, out_dir / f"{image_index}.png")
