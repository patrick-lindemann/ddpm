import argparse
import logging
import pathlib
from typing import List

import torch
import matplotlib.pyplot as plt

from diffusion.data import (
    load_image,
    reverse_transform_image,
    save_image,
    transform_image,
)
from diffusion.schedule import (
    CosineScheduler,
    LinearScheduler,
    Scheduler,
    SigmoidScheduler,
    QuadraticScheduler,
)
from diffusion.diffusion import GaussianDiffuser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        type=pathlib.Path,
        help="The path to the image to apply noise to.",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.\nAllowed values: "CPU", "Cuda".',
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        help="The number of time steps for the diffusion process.",
        default=10,
    )
    parser.add_argument(
        "--schedule",
        type=str.lower,
        help='The schedule to use.\nAllowed values: "linear", "quadratic", "cosine", "sigmoid".',
        default="linear",
    )
    parser.add_argument(
        "--schedule-start",
        type=float,
        help="The starting value for the schedule.",
        default=0.0,
    )
    parser.add_argument(
        "--schedule-end",
        type=float,
        help="The ending value for the schedule.",
        default=1.0,
    )
    parser.add_argument(
        "--schedule-tau",
        type=float,
        help="The tau value for the schedule. Only applicable for cosine and sigmoid schedules.",
        default=1.0,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="The (quadratic) size to scale the image to.",
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

    # Prepare the scheduler
    scheduler: Scheduler
    if args.schedule == "linear":
        scheduler = LinearScheduler(start=args.schedule_start, end=args.schedule_end)
    elif args.schedule == "quadratic":
        scheduler = QuadraticScheduler(start=args.schedule_start, end=args.schedule_end)
    elif args.schedule == "cosine":
        scheduler = CosineScheduler(
            start=args.schedule_start, end=args.schedule_end, tau=args.schedule_tau
        )
    elif args.schedule == "sigmoid":
        scheduler = SigmoidScheduler(
            start=args.schedule_start, end=args.schedule_end, tau=args.schedule_tau
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Prepare the diffuser
    diffuser = GaussianDiffuser(num_steps=args.time_steps, scheduler=scheduler)

    # Load the data
    logging.info(f'Loading image from "{args.image_path}"')
    image = load_image(
        args.image_path,
        transformation=lambda image: transform_image(
            image, size=(args.image_size, args.image_size)
        ),
    )
    image = image.reshape(shape=[1, *image.shape])

    # Apply noise to the image incrementally
    logging.info(
        f"Applying noise to image for {args.time_steps} time steps with {args.schedule} schedule."
    )
    noised_images: List[torch.Tensor] = []
    for t in range(args.time_steps):
        result, _ = diffuser.forward(image, torch.tensor([t]))
        noised_images.append(result[0])

    # Export the results
    file_path = args.outdir / f"timeline-{args.schedule}.png"
    logging.info(f'Saving result to "{file_path}".')
    timeline = torch.cat(noised_images, dim=2)
    timeline = reverse_transform_image(timeline)
    save_image(file_path, timeline)

    # Export the single images
    if args.export_all:
        logging.info(f'Saving single images to "{args.outdir}".')
        for i, noised_image in enumerate(noised_images):
            data = reverse_transform_image(noised_image)
            file_path = args.outdir / f"image_{i}.png"
            save_image(file_path, data)

    logging.info("Done.")
