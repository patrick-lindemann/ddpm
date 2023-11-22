import argparse
import logging
import pathlib
from typing import List

import matplotlib.pyplot as plt
import torch

from diffusion.data import load_data
from diffusion.model import BasicUNet
from diffusion.schedule import (
    CosineScheduler,
    LinearScheduler,
    Scheduler,
    SigmoidScheduler,
)
from diffusion.diffusion import GaussianDiffuser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str.lower,
        help='The name of the dataset.\nAllowed values: "CelebA", "CIFAR10", "LSUN".',
        default="CIFAR10",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        help='The device to use.\nAllowed values: "CPU", "Cuda".',
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--schedule",
        type=str.lower,
        help='The schedule to use.\nAllowed values: "linear", "cosine", "exponential".',
        default="linear",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        help="The number of time steps for the diffusion process.",
        default=10,
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
        "--learning-rate",
        type=float,
        help="The learning rate for the optimizer.",
        default=1e-3,
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
        scheduler = LinearScheduler()
    elif args.schedule == "cosine":
        scheduler = CosineScheduler()
    elif args.schedule == "sigmoid":
        scheduler = SigmoidScheduler()
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Prepare the diffuser
    diffuser = GaussianDiffuser(num_steps=args.time_steps, scheduler=scheduler)

    # Load the data
    logging.info("Loading dataset.")
    train_loader, test_loader = load_data(args.dataset, batch_size=args.batch_size)

    # Prepare the model
    model = BasicUNet(channels_in=3, channels_out=3)
    # model.to(args.device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train_iter = iter(train_loader)
    losses: List[float] = []
    for epoch in range(args.epochs + 1):
        # Retrieve the next batch of images
        image_batch, _ = next(train_iter)
        # Select a random time step for each image in the batch and apply the
        # noise for that time step
        t = torch.randint(0, args.time_steps, (args.batch_size,))
        noised_image_batch, noise_batch = scheduler(image_batch, t)
        # Predict the noise for the noised images and calculate the loss
        for noised_image, noise in zip(noised_image_batch, noise_batch):
            predicted_noise = model(noised_image.float())
            loss = loss_func(predicted_noise, noise)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print the average of the loss values for this epoch
        if epoch % 100 == 0:
            avg_loss = sum(losses[-args.batch_size :]) / args.batch_size
            logging.info(
                f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}"
            )

    # Plot the training losses
    plt.plot(losses)
    plt.show()

    # Test the model

    # Plot the test losses

    # Export the model
    # torch.save(model.state_dict(), "model.pt")
