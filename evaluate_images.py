import argparse
import logging
import pathlib

import torch

from src.data import load_images
from src.eval import calculate_inception_score


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
    device = torch.device(args.device)

    # Prepare the logger
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load the images
    logging.info(f"Loading images from {args.image_dir}.")
    images = load_images(args.image_dir, device=device)
    logging.info(f"Loaded {len(images)} images.")

    # Calculate the inception scores
    logging.info("Calculating inception score.")
    score, _ = calculate_inception_score(images, batch_size=10, device=device)
    logging.info(f"Inception score: {score:.4f}")
