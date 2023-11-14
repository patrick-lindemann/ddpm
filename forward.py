import argparse
import logging

from diffusion.data import load_data
from diffusion.model import DiffusionModel
from diffusion.plot import show_image


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help='The name of the dataset.\nAllowed values: "cifar10", "lsun", "celeba".',
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()

    # Prepare the logger
    logging.basicConfig(level="debug" if args.verbose else "info")

    # Load the data
    logging.info("Loading data...")
    train_set, val_set = load_data(args.dataset)
    logging.info("Data loaded successfully.")

    # Forward pass
    # TODO: Implement this

    # Show/Save the results
    # TODO: Implement this

    pass
