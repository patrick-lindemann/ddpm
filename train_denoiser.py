import argparse
import json
import logging
import os
import pathlib
import time

import torch
import torch.utils.data
from tqdm import tqdm

from src.diffuser import GaussianDiffuser
from src.model import DiffusionModel
from src.paths import OUT_DIR
from src.schedule import (
    CosineScheduler,
    LinearScheduler,
    PolynomialScheduler,
    Scheduler,
    SigmoidScheduler,
)
from src.utils import load_image


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


"""Functions"""


def one_epoch():
    pass


def train():
    pass


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

    dataset = get_dataset(dataset_name)
    train_loader, validation_loader = create_dataloader(dataset, batch_size=batch_size)

    # Prepare the logger
    # Prepare the diffuser
    schedule = get_schedule(
        schedule_name, start=schedule_start, end=schedule_end, tau=schedule_tau
    )
    diffuser = GaussianDiffuser(num_steps=time_steps, schedule=schedule, device=device)

    # Prepare the output directory

    # Load the data
    logging.info("Loading dataset.")
    dataset = load_dataset(args.dataset)
    train_indices, test_indices = split_dataset(
        dataset, train_size=args.train_split_size
    )
    train_loader = create_dataloader(
        dataset, train_indices, batch_size=args.batch_size, device=device
    )

    # Prepare the model
    model = DiffusionModel(sample_size=args.sample_size, dropout_rate=args.dropout_rate)
    model.to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.learning_rate_stepsize, gamma=args.learning_rate_gamma
    )

    sample_in_between = True

    for epoch in tqdm(range(epochs)):
        for step, batch in enumerate(tqdm(train_loader, leave=False)):
            pass

    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(
            sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg")
        )
        torch.save(
            model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt")
        )

    # Train the model
    logging.info(
        f"Starting training for {args.epochs} epochs with batch size {args.batch_size}."
    )
    epoch_losses = torch.zeros(args.epochs, dtype=torch.float32, device=device)
    for epoch in tqdm(range(args.epochs)):
        epoch_loss_sum = 0
        for step, batch in enumerate(tqdm(train_loader, leave=False)):
            optimizer.zero_grad()
            image_batch = batch[0]
            # Select a random time step for each image in the batch and apply the
            # noise for that time step
            t = torch.randint(0, args.schedule_steps, (args.batch_size,), device=device)
            noised_image_batch, noise_batch = diffuser.forward(image_batch, t)
            # Predict the noise for the noised images and calculate the loss
            predicted_noise_batch = model(noised_image_batch.float(), t).sample
            loss = loss_func(predicted_noise_batch, noise_batch)
            epoch_loss_sum += loss.item()
            # Backpropagate the loss and update the model parameters
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss_sum / len(dataset)
        epoch_losses[epoch] = epoch_loss
        # Update the learning rate
        lr_scheduler.step()
        # TODO: Save the model if the loss is the best so far

    # Export the metadata
    metadata = {
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "learning_rate_stepsize": args.learning_rate_stepsize,
            "learning_rate_gamma": args.learning_rate_gamma,
            "losses": epoch_losses.tolist(),
            "final_loss": epoch_losses[-1].item(),
        },
        "schedule": {
            "type": scheduler.name,
            "start": scheduler.start,
            "end": scheduler.end,
            "tau": scheduler.tau if scheduler.name != "linear" else None,
            "steps": args.schedule_steps,
        },
        "dataset": {
            "name": args.dataset,
            "size": len(dataset),
            "batch_size": args.batch_size,
            "split": {
                "train": {
                    "size": len(train_indices),
                    "indices": train_indices,
                },
                "test": {
                    "size": len(test_indices),
                    "indices": test_indices,
                },
            },
        },
    }
    metadata_path = args.outdir / "metadata.json"
    logging.info(f"Saving metadata to {metadata_path}.")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)

    # Plot the losses
    losses_path = args.outdir / "train-loss.svg"
    logging.info(f"Saving loss plot to {losses_path}.")
    plot_loss(
        losses=epoch_losses.cpu(),
        file_path=losses_path,
    )

    # Export the model
    model_path = args.outdir / "model.pt"
    logging.info(f"Saving model to {model_path}.")
    torch.save(model.state_dict(), model_path)
