import argparse
import logging
import pathlib
from typing import List

import torch

from src.image import is_image_path, load_image
from src.inception import INCEPTION_IMAGE_SIZE, calculate_inception_score


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
    assert image_dir.is_dir()
    image_paths: List[pathlib.Path] = []
    for file_path in image_dir.iterdir():
        if is_image_path(file_path):
            image_paths.append(file_path)
    assert len(image_paths) > 0, f"No images found in the directory {image_dir}."

    # Load the images
    N = len(image_paths)
    logging.info(f"Loading {N} images from {image_dir}.")
    images = torch.zeros((N, 3, INCEPTION_IMAGE_SIZE, INCEPTION_IMAGE_SIZE))
    for i, image_path in enumerate(image_paths):
        image = load_image(image_path, resize_to=INCEPTION_IMAGE_SIZE)
        images[i] = image

    # Calculate the inception scores
    logging.info(f"Calculating inception score and writing it to {image_dir}.")
    score, _ = calculate_inception_score(images, batch_size=100, device=device)
    with open(image_dir / "inception_score.txt", "w") as f:
        f.write(f"{score:.4f}")
    logging.info(f"Inception score: {score:.4f}")
