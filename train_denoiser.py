import argparse
import json
import logging
import os
import pathlib
import time
from typing import Dict

import torch
import torch.utils.data
from tqdm import tqdm

from src.data import create_dataloaders, load_dataset
from src.diffuser import GaussianDiffuser
from src.model import DenoisingUNet2D
from src.paths import OUT_DIR
from src.schedule import get_schedule


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str.lower,
        help='The name of the dataset. Allowed values: "cifar10", "mnist", "aircraft", "flower", "celeba".',
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name of the experiment. If not provided, the name will be generated automatically.",
        default=None,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="The (square) sample size of the images. If the provided image size is different from the dataset's image size, the images will be up- or downsized accordingly.",
        default=32,
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        help="The number of time steps for the diffusion process.",
        default=1000,
    )
    parser.add_argument(
        "--schedule",
        type=str.lower,
        help='The schedule to use.\nAllowed values: "linear", "polynomial", "cosine", "exponential".',
        default="linear",
    )
    parser.add_argument(
        "--schedule-start",
        type=float,
        help="The start value for the schedule.",
        default=0.0,
    )
    parser.add_argument(
        "--schedule-end",
        type=float,
        help="The end value for the schedule.",
        default=1.0,
    )
    parser.add_argument(
        "--schedule-tau",
        type=float,
        help="The tau value for the schedule. Only applicable for polynomial, cosine and sigmoid schedules.",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="The number of epochs to train for.",
        default=300,
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="The number of samples to use from the dataset. If not provided, the entire dataset is used.",
        default=None,
    )
    parser.add_argument(
        "--train-split",
        type=float,
        help="The percentage in [0, 1] of the dataset to use for training.",
        default=0.8,
    )
    parser.add_argument(
        "--disable-validation",
        action="store_true",
        help="Disable validation on the test set during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size for the data loader.",
        default=16,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="The learning rate for the optimizer.",
        default=1e-3,
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        help="The dropout rate for the model.",
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use. If not specified, the randomized calculations will be non-deterministic.",
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="The directory to save the results to.",
        default=OUT_DIR / "runs",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.Allowed values: "cpu", "cuda".',
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the arguments
    args = get_args()
    dataset_name: str = args.dataset
    run_name: str = (
        args.run_name
        if args.run_name is not None
        else f"{int(time.time())}-{dataset_name}"
    )
    image_size: int = args.image_size
    time_steps: int = args.time_steps
    schedule_name: str = args.schedule
    schedule_start: int = args.schedule_start
    schedule_end: int = args.schedule_end
    schedule_tau: float | None = args.schedule_tau
    epochs: int = args.epochs
    do_validation: bool = not args.disable_validation
    subset_size: int | None = args.subset_size
    train_split: float = args.train_split
    batch_size: int = args.batch_size
    learning_rate: float = args.learning_rate
    dropout_rate: float = args.dropout_rate
    seed: int = args.seed
    out_dir: pathlib.Path = args.out_dir / run_name
    device = torch.device(args.device)
    verbose: bool = args.verbose
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if seed is not None:
        torch.manual_seed(seed)

    # Load the dataset
    dataset = load_dataset(dataset_name, resize_to=image_size)

    # Validate the arguments
    dataset_size = subset_size if subset_size is not None else len(dataset)
    assert batch_size < dataset_size
    train_size = int(dataset_size * train_split)
    test_size = dataset_size - train_size
    if do_validation:
        assert test_size > 0
    if not out_dir.exists():
        os.makedirs(out_dir)

    # Prepare the model and training
    train_loader, test_loader = create_dataloaders(
        dataset,
        train_size=train_size,
        test_size=test_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
    )
    schedule = get_schedule(
        schedule_name, start=schedule_start, end=schedule_end, tau=schedule_tau
    )
    model = DenoisingUNet2D(image_size, dropout_rate=dropout_rate).to(device)
    diffuser = GaussianDiffuser(time_steps, schedule).to(device)
    loss_func = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)

    # Prepare the output directory
    logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}.")
    train_losses = torch.zeros((epochs), device=device)
    test_losses = torch.zeros((epochs), device=device)
    learning_rates = torch.zeros((epochs), device=device)
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc="train", leave=False)):
            optimizer.zero_grad()
            images = batch[0]
            t = torch.randint(0, time_steps, (images.shape[0],), device=device)
            noised_images, noises = diffuser.forward(images, t)
            predicted_noises = model(noised_images.float(), t).sample
            loss = loss_func(predicted_noises, noises)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses[epoch] = train_loss / len(train_loader)
        learning_rates[epoch] = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        if do_validation:
            test_loss = 0.0
            model.eval()
            for step, batch in enumerate(tqdm(test_loader, desc="test", leave=False)):
                images = batch[0]
                t = torch.randint(0, time_steps, (images.shape[0],), device=device)
                noised_images, noises = diffuser.forward(images, t)
                predicted_noises = model(noised_images.float(), t).sample
                loss = loss_func(predicted_noises, noises)
                test_loss += loss.item()
            test_losses[epoch] = test_loss / len(test_loader)
            model.train()

    # Export the model and diffuser
    logging.info(f"Saving model and diffuser to {out_dir}.")
    model.save(out_dir)
    diffuser.save(out_dir)

    # Export the metadata
    metadata = {
        "name": run_name,
        "dataset": dataset_name,
        "dataset_size": len(dataset),
        "train_size": len(train_loader),
        "test_size": len(test_loader) if do_validation else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rates": learning_rates.tolist(),
        "train_losses": train_losses.tolist(),
        "test_losses": test_losses.tolist(),
    }
    metadata_path = out_dir / "run.json"
    logging.info(f"Saving metadata to {metadata_path}.")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)
