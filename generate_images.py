import argparse
import logging
import os
import pathlib
import uuid

import torch
from tqdm import tqdm

from src.diffuser import GaussianDiffuser
from src.model import DenoisingUNet2D
from src.paths import OUT_DIR
from src.image import save_image


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir"
        type=str,
        help="The path to the model to test.\nThe model directory needs to contain a model.pt and a metadata.json file.",
    )
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        help='The path to the model directory, which contains the files "config.json" and "weights.pt".',
        default=None,
    )
    parser.add_argument(
        "num_images",
        type=int,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size for generating images.\nMust be a divisor of num_images, e.g. for 12 generated images batch sizes of 1, 2, 3, 4 and 6 are possible.",
        default=1,
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
    batch_size: int = args.batch_size
    num_batches = num_images // batch_size
    run_dir: pathlib.Path = args.run_dir
    run_name = run_dir.name
    seed: int = args.seed
    out_dir = args.outdir if args.outdir is not None else OUT_DIR / "images" / run_name
    device = torch.device(args.device)
    verbose: bool = args.verbose
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if seed is not None:
        torch.manual_seed(seed)

    # Validate the arguments
    assert run_dir.exists()
    assert batch_size < num_images
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
        for image in images:
            image_path = out_dir / f"{uuid.uuid4()}.png"
            save_image(image, image_path)
