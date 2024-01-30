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
        help='The name of the dataset.\nAllowed values: "CelebA", "MNIST", "FGVCAircraft", "CIFAR10", "LSUN".',
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name of the experiment. If not provided, the name will be generated from the timestamp, dataset and scheduler.",
        default=None,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="The size of the images in the dataset.",
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
        "--epochs",
        type=int,
        help="The number of epochs to train for.",
        default=300,
    )
    parser.add_argument(
        "--validate-at",
        type=int,
        help="Validate the model with the test set every nth epoch.",
        default=None,
    )
    parser.add_argument(
        "--sample-at",
        type=int,
        help="Sample every nth epoch.",
        default=None,
    )
    parser.add_argument(
        "--train-size",
        type=float,
        help="The size of the training set. Can be a percentage in [0, 1] or an integer specifying the number of training samples.",
        default=0.8,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="The size of the test set. Can be a percentage in [0, 1] or an integer specifying the number of test samples.",
        default=0.2,
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
        help="The random seed to use.",
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="The directory to save the results to.",
        default=OUT_DIR / "train",
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
    dataset_name: str = args.dataset
    run_name: str = (
        args.run_name
        if args.run_name is not None
        else f"{int(time.time())}_{args.dataset}_{args.schedule}"
    )
    image_size: int = args.image_size
    time_steps: int = args.time_steps
    schedule_name: str = args.schedule
    schedule_start: int = args.schedule_start
    schedule_end: int = args.schedule_end
    schedule_tau: float | None = args.schedule_tau
    epochs: int = args.epochs
    validate_at: int | None = args.validate_at
    sample_at: int | None = args.sample_at
    train_size: float = args.train_size
    test_size: float = args.test_size
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

    # Validate the arguments
    if not out_dir.exists():
        os.makedirs(out_dir)

    # Prepare the dataset, model and training
    dataset = load_dataset(dataset_name, resize_to=image_size)
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
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # Prepare the output directory
    logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}.")
    train_losses: Dict[int, float] = {}
    test_losses: Dict[int, float] = {}
    learning_rates: Dict[int, float] = {}
    for epoch in tqdm(range(args.epochs, "epochs")):
        train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, "train", leave=False)):
            optimizer.zero_grad()
            images = batch[0]
            t = torch.randint(0, time_steps, (batch_size,), device=device)
            noised_images, noises = diffuser.forward(images, t)
            predicted_noises = model(noised_images.float(), t).sample
            loss = loss_func(predicted_noises, noises)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses[epoch] = train_loss / len(train_loader)
        learning_rates[epoch] = lr_scheduler.get_lr()
        lr_scheduler.step()
        if validate_at is not None and epoch % validate_at == 0:
            test_loss = 0.0
            with torch.no_grad():
                for step, batch in enumerate(tqdm(test_loader, "test", leave=False)):
                    images = batch[0]
                    t = torch.randint(0, time_steps, (batch_size,), device=device)
                    noised_images, noises = diffuser.forward(images, t)
                    predicted_noises = model(noised_images.float(), t).sample
                    loss = loss_func(predicted_noises, noises)
                    test_loss += loss.item()
            test_losses[epoch] = test_loss / len(test_loader)
        if sample_at is not None and epoch % sample_at == 0:
            for step, batch

    # Export the model and diffuser
    logging.info(f"Saving model and diffuser to {out_dir}.")
    model.save(out_dir)
    diffuser.save(out_dir)

    # Export the metadata
    metadata = {
        "name": run_name,
        "dataset": dataset_name,
        "dataset_size": len(dataset),
        "train_size": int(train_size * len(dataset)),
        "test_size": int(test_size * len(dataset)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rates": learning_rates,
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    metadata_path = args.outdir / "run.json"
    logging.info(f"Saving metadata to {metadata_path}.")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)
