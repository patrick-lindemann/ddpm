import argparse
import logging
import os
import pathlib

import torch
from tqdm import tqdm

from src.diffuser import GaussianDiffuser
from src.image import save_image
from src.model import DenoisingUNet2D


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=pathlib.Path,
        help="The path to the run.",
    )
    parser.add_argument(
        "num_images",
        type=int,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size for generating images.\nMust be a divisor of num_images.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use.",
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="The directory to save the results to. If not specified, the images are saved to <out_dir>/images/<run_name>.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.\nAllowed values: "CPU", "Cuda".',
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
    num_batches = num_images // batch_size
    run_dir: pathlib.Path = args.run_dir
    run_name = run_dir.name
    seed: int = args.seed
    out_dir = (
        args.out_dir if args.out_dir is not None else run_dir / "samples" / "generated"
    )
    device = torch.device(args.device)
    verbose: bool = args.verbose
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if seed is not None:
        torch.manual_seed(seed)

    # Validate the arguments
    assert run_dir.exists()
    assert batch_size <= num_images
    assert num_images % batch_size == 0
    if not out_dir.exists():
        os.makedirs(out_dir)

    # Prepare the diffuser
    logging.info(f'Loading model from "{run_dir}".')
    model = DenoisingUNet2D.load(run_dir).to(device)
    diffuser = GaussianDiffuser.load(run_dir).to(device)

    # Generate and save the images
    logging.info(f"Generating {num_images} images to dir {out_dir}.")
    logging.debug(f"Using {num_batches} batches with size {batch_size}.")
    for _ in tqdm(range(num_batches)):
        images: torch.Tensor = diffuser.sample(model, batch_size)
        for i, image in enumerate(images):
            image_path = out_dir / f"{i}.png"
            save_image(image, image_path)
