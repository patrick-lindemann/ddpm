import argparse
import logging
import os
import pathlib

import torch
from tqdm import tqdm

from src.diffuser import GaussianDiffuser
from src.image import load_image, save_image, save_timeline
from src.paths import OUT_DIR
from src.schedule import get_schedule


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        type=pathlib.Path,
        help="The path to the image to apply noise to.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="The (quadratic) size to resize the image to.",
        default=128,
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
        help='The schedule to use.\nAllowed values: "linear", "polynomial", "cosine", "sigmoid".',
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
        help="The tau value for the schedule. Only applicable for polynomial, cosine and sigmoid schedules.",
        default=None,
    )
    parser.add_argument(
        "--timeline-stepsize",
        type=int,
        help="The stepsize for exporting the timeline. Only applicable if --export-timeline is set.",
        default=1,
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export the images at each time step.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="The directory to save the results to.",
        default=OUT_DIR / "forward",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()
    image_path: pathlib.Path = args.image_path
    image_size: int = args.image_size
    time_steps: int = args.time_steps
    schedule_name: str = args.schedule
    schedule_start: int = args.schedule_start
    schedule_end: int = args.schedule_end
    schedule_tau: float | None = args.schedule_tau
    timeline_stepsize = args.timeline_stepsize
    export_all: bool = args.export_all
    out_dir: pathlib.Path = args.out_dir / schedule_name
    verbose: bool = args.verbose
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    # Validate the arguments
    assert image_path.exists()
    if schedule_name == "linear":
        assert schedule_tau is None
    if not out_dir.exists():
        os.makedirs(out_dir)

    # Prepare the diffuser
    schedule = get_schedule(
        schedule_name, start=schedule_start, end=schedule_end, tau=schedule_tau
    )
    diffuser = GaussianDiffuser(time_steps, schedule)

    # Load the image
    logging.info(f'Loading image from "{image_path}"')
    image = load_image(image_path, resize_to=image_size)
    image = image.unsqueeze(0)

    # Apply noise to the image incrementally
    logging.info(
        f"Applying noise to image within {time_steps} time steps with {schedule_name} schedule."
    )
    noise_step_images = torch.zeros((time_steps, *image.squeeze().shape))
    for t in tqdm(range(time_steps)):
        noised_image, _ = diffuser.forward(image, torch.tensor([t]))
        noised_image = torch.clip(noised_image, -1.0, 1.0)
        noise_step_images[t] = noised_image.squeeze()

    # Save the noise process results
    logging.info(f'Saving results to "{out_dir}".')
    save_timeline(noise_step_images, out_dir / f"timeline.png", timeline_stepsize)
    if export_all:
        for i, noised_image in enumerate(noise_step_images):
            save_image(noised_image, out_dir / f"image_{i + 1}.png")

    logging.info("Done.")
