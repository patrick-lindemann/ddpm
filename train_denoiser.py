import argparse
import json
import logging
import os
import pathlib
import time

import torch
import torch.utils.data
from tqdm import tqdm

from diffusion.data import (
    create_dataloader,
    load_dataset,
    plot_loss,
    split_dataset,
)
from diffusion.diffusion import GaussianDiffuser
from diffusion.model import DiffusionModel
from diffusion.paths import OUT_DIR
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
        "dataset",
        type=str.lower,
        help='The name of the dataset.\nAllowed values: "CelebA", "MNIST", "FGVCAircraft", "CIFAR10", "LSUN".',
    )
    parser.add_argument(
        "--name",
        type=str,
        help="The name of the experiment. If not provided, the name will be generated from the timestamp, dataset and scheduler.",
        default=None,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="The size of the images in the dataset.",
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="The number of epochs to train for.",
        default=300,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size for the data loader.",
        default=16,
    )
    parser.add_argument(
        "--schedule",
        type=str.lower,
        help='The schedule to use.\nAllowed values: "linear", "polynomial", "cosine", "exponential".',
        default="linear",
    )
    parser.add_argument(
        "--schedule-steps",
        type=int,
        help="The number of time steps for the diffusion process.",
        default=1000,
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
        "--train-split-size",
        type=float,
        help="The size of the training set.",
        default=0.8,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="The learning rate for the optimizer.",
        default=1e-3,
    )
    parser.add_argument(
        "--learning-rate-stepsize",
        type=int,
        help="The step size for the learning rate scheduler.",
        default=5,
    )
    parser.add_argument(
        "--learning-rate-gamma",
        type=float,
        help="The gamma value for the learning rate scheduler.",
        default=0.8,
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        help="The dropout rate for the model.",
        default=0.1,
    )
    parser.add_argument(
        "--outdir",
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
    device = torch.device(args.device)
    if not args.name:
        args.name = f"{int(time.time())}_{args.dataset}_{args.schedule}"
    args.outdir = args.outdir / args.name

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
    diffuser = GaussianDiffuser(
        num_steps=args.schedule_steps, scheduler=scheduler, device=device
    )

    # Prepare the output directory
    if not args.outdir.exists():
        os.makedirs(args.outdir)

    # Load the data
    logging.info("Loading dataset.")
    dataset = load_dataset(args.dataset, device=device)
    train_indices, test_indices = split_dataset(
        dataset, train_size=args.train_split_size
    )
    train_loader = create_dataloader(dataset, train_indices, batch_size=args.batch_size)

    # Prepare the model
    model = DiffusionModel(sample_size=args.sample_size, dropout=args.dropout_rate)
    model.to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.learning_rate_stepsize, gamma=args.learning_rate_gamma
    )

    # Train the model
    logging.info(
        f"Starting training for {args.epochs} epochs with batch size {args.batch_size}."
    )
    epoch_losses = torch.zeros(args.epochs, dtype=torch.float32, device=device)
    for epoch in tqdm(range(args.epochs)):
        epoch_loss_sum = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image_batch = batch[0]
            # Select a random time step for each image in the batch and apply the
            # noise for that time step
            t = torch.randint(0, args.schedule_steps, (args.batch_size,))
            noised_image_batch, noise_batch = diffuser.forward(image_batch, t)
            # Predict the noise for the noised images and calculate the loss
            predicted_noise_batch = model(noised_image_batch.float(), t).sample
            loss = loss_func(predicted_noise_batch, noise_batch)
            epoch_loss_sum += loss.item()
            # Backpropagate the loss and update the model parameters
            loss.backward()
            optimizer.step()
            break
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
        "model": {
            "sample_size": args.sample_size,
            "down_blocks": len(model.down_blocks),
            "up_blocks": len(model.up_blocks),
            "dropout_rate": args.dropout_rate,
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
        losses=epoch_losses,
        file_path=losses_path,
    )

    # Export the model
    model_path = args.outdir / "model.pt"
    logging.info(f"Saving model to {model_path}.")
    torch.save(model.state_dict(), model_path)
