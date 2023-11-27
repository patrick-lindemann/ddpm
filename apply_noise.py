import argparse
import logging
import os
import pathlib
from typing import List

import torch

from diffusion.data import (
    load_image,
    plot_schedule,
    reverse_transform_image,
    save_image,
    transform_image,
)
from diffusion.diffusion import GaussianDiffuser
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
        "image_path",
        type=pathlib.Path,
        help="The path to the image to apply noise to.",
    )
    parser.add_argument(
        "--schedule",
        type=str.lower,
        help='The schedule to use.\nAllowed values: "linear", "polynomial", "cosine", "sigmoid".',
        default="linear",
    )
    parser.add_argument(
        "--schedule-steps",
        type=int,
        help="The number of time steps for the diffusion process.",
        default=10,
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
        help="The tau value for the schedule. Only applicable for polynomial, cosine and sigmoid schedules.",
        default=None,
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
        default="./out/forward",
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

    # Prepare the logger
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Prepare the scheduler
    scheduler: Scheduler
    if args.schedule == "linear":
        scheduler = LinearScheduler(start=args.schedule_start, end=args.schedule_end)
    elif args.schedule == "polynomial":
        scheduler = PolynomialScheduler(
            start=args.schedule_start,
            end=args.schedule_end,
            tau=args.schedule_tau or 2.0,
        )
    elif args.schedule == "cosine":
        scheduler = CosineScheduler(
            start=args.schedule_start,
            end=args.schedule_end,
            tau=args.schedule_tau or 1.0,
        )
    elif args.schedule == "sigmoid":
        scheduler = SigmoidScheduler(
            start=args.schedule_start,
            end=args.schedule_end,
            tau=args.schedule_tau or 1.0,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Prepare the diffuser
    diffuser = GaussianDiffuser(num_steps=args.schedule_steps, scheduler=scheduler)

    # Prepare the output directory
    outdir = args.outdir / args.schedule
    if not outdir.exists():
        os.makedirs(outdir)

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
        f"Applying noise to image for {args.schedule_steps} time steps with {args.schedule} schedule."
    )
    noised_images: List[torch.Tensor] = []
    for t in range(args.schedule_steps):
        result, _ = diffuser.forward(image, torch.tensor([t]))
        noised_images.append(result[0])

    # Export the noise process results
    timeline_path = outdir / f"timeline.png"
    logging.info(f'Saving noising process timeline to "{timeline_path}".')
    timeline = torch.cat(noised_images, dim=2)
    timeline = reverse_transform_image(timeline)
    save_image(timeline_path, timeline)

    # Export the individual images
    logging.info(f'Saving individual images to "{outdir}".')
    for i, noised_image in enumerate(noised_images):
        data = reverse_transform_image(noised_image)
        file_path = outdir / f"image-{i + 1}.png"
        save_image(file_path, data)

    # Export the schedule plot
    plot_path = outdir / "schedule-plot.svg"
    logging.info(f'Saving schedule plot to "{plot_path}".')
    plot_schedule(scheduler, file_path=plot_path)

    logging.info("Done.")
