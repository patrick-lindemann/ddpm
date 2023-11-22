import argparse
import logging
import os
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy
import torch
from PIL.Image import Image
from torchvision import transforms

from diffusion.data import load_data, save_image


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str.lower,
        help='The name of the dataset.\nAllowed values: "CelebA", "CIFAR10", "LSUN".',
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="The number of rows in the grid.",
        default=4,
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="The number of columns in the grid.",
        default=4,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="The (quadratic) size to scale the images to.",
        default=128,
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        help="The directory to save the results to.",
        default="./out/",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all images, not only the visualization.",
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

    # Make sure the output directory exists
    outdir = args.outdir / args.dataset
    if not outdir.exists():
        os.makedirs(outdir)

    # Load the data
    logging.info("Loading dataset.")
    N = args.rows * args.cols
    train_loader, _ = load_data(args.dataset, batch_size=N, train_ratio=1.0)

    # Prepare the image transformation
    transformation = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToPILImage(),
        ]
    )

    # Pick N random images from the dataset
    data: torch.Tensor = next(iter(train_loader))
    images: List[Image] = []
    for tensor in data[0]:
        # image = reverse_transform_image(tensor)
        image = transformation(tensor)
        images.append(image)

    # Plot the images on a quadratic grid and save the result
    fig, axes = plt.subplots(nrows=args.rows, ncols=args.cols)
    for ax in axes.flat:
        ax.set_axis_off()
    for i, image in enumerate(images):
        axes.flat[i].imshow(image)
    fig.savefig(outdir / "images.png")

    if args.export_all:
        # Save all the images
        logging.info("Saving all images.")
        for i, image in enumerate(images):
            save_image(outdir / f"image_{i + 1}.png", image)
