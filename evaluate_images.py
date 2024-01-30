import argparse
import logging
import pathlib

import torch

from src.inception import calculate_inception_score
from src.utils import load_images


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_dir",
        type=pathlib.Path,
        help="The path to folder containing the images to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.\nAllowed values: "cpu", "cuda".',
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()
    image_dir: pathlib.Path = args.image_dir
    device = torch.device(args.device)
    verbose: bool = args.verbose
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    # Validate the arguments
    assert image_dir.exists()

    # Load the images
    logging.info(f"Loading images from {image_dir}.")
    images = load_images(image_dir).to(device)
    logging.info(f"Loaded {len(images)} images.")

    # Calculate the inception scores
    logging.info("Calculating inception score.")
    score, _ = calculate_inception_score(images, batch_size=10, device=device)
    logging.info(f"Inception score: {score:.4f}")
