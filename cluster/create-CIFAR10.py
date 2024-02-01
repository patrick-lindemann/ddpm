import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import UNet2DModel
import torch.nn.functional as F
import logging
from datetime import datetime
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
)

IMG_SIZE = 32
TIME_STEPS = 1000


class diffusion:
    def __init__(self, timesteps, scheduler_type=None):
        betas = torch.linspace(0.0001, 0.02, timesteps)

        alphas = 1 - betas
        sqrt_alphas = torch.sqrt(alphas)
        alpha_hats = torch.cumprod(1 - betas, axis=0)
        sqrt_alpha_hats = torch.sqrt(alpha_hats)

        sqrt_one_minus_alpha_hats = torch.sqrt(1 - alpha_hats)

        alpha_hats_prev = F.pad(alpha_hats[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_hats_prev) / (1.0 - alpha_hats)

        self.betas = betas
        self.alphas = alphas
        self.sqrt_alphas = sqrt_alphas
        self.alpha_hats = alpha_hats
        self.sqrt_alpha_hats = sqrt_alpha_hats
        self.sqrt_one_minus_alpha_hats = sqrt_one_minus_alpha_hats
        self.posterior_variance = posterior_variance

    def noise_schedule(self, x_0, t, device="cpu"):
        noise = torch.randn(size=x_0.shape)

        sqrt_alpha_hats_t = self.sqrt_alpha_hats[t]
        sqrt_alpha_hats_t = sqrt_alpha_hats_t.reshape(shape=(t.shape[0], 1, 1, 1))

        sqrt_one_minus_alpha_hats_t = self.sqrt_one_minus_alpha_hats[t]
        sqrt_one_minus_alpha_hats_t = sqrt_one_minus_alpha_hats_t.reshape(
            shape=(t.shape[0], 1, 1, 1)
        )

        mean = sqrt_alpha_hats_t * x_0
        variance = sqrt_one_minus_alpha_hats_t * noise

        x_ts = mean + variance

        return x_ts.to(device), noise.to(device)

    @torch.no_grad()
    def sample(self, x_t, t, prediction):

        betas_t = self.betas[t]
        sqrt_alphas_t = self.sqrt_alphas[t]

        sqrt_one_minus_alpha_hats_t = self.sqrt_one_minus_alpha_hats[t]

        x_t_minus_one = (1 / sqrt_alphas_t) * (
            x_t - ((betas_t * prediction) / sqrt_one_minus_alpha_hats_t)
        )

        if t == 0:
            return x_t_minus_one
        else:
            noise = torch.randn_like(x_t)
            posterior_variance_t = self.posterior_variance[t]
            return x_t_minus_one + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def create_image(
        self,
        model,
        num_img,
        img_size=32,
        timesteps=300,
        num_channels=3,
        show_process=False,
    ):
        self.betas = self.betas.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.sqrt_one_minus_alpha_hats = self.sqrt_one_minus_alpha_hats.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

        img = torch.randn((num_img, num_channels, img_size, img_size), device=device)
        num_images = 10
        stepsize = int(timesteps / num_images)

        if show_process:
            fig = plt.figure(figsize=(15, 2))
            plt.axis("off")

        for i in range(0, timesteps)[::-1]:
            t = torch.full((num_img,), i, dtype=torch.long, device=device)

            prediction = model(img, t).sample

            img = self.sample(img, i, prediction)

            if show_process:
                if i % stepsize == 0:
                    plt.subplot(1, num_images, int(i / stepsize) + 1)
                    plt.imshow(undo_transform(img.cpu()[0]))

        if show_process:
            plt.show()
            fig.savefig(f"{datetime.now()}.png")
        return img.cpu()


transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

undo_transform = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: (t * 255.0).numpy().astype(np.uint8)),
    ]
)

diffuser = diffusion(timesteps=TIME_STEPS)

model = UNet2DModel().from_pretrained("model_CIFAR10_checkpoint")
model = model.to(device)

num_per_batch = 100

for i in range(5):
    images = diffuser.create_image(
        model,
        num_img=num_per_batch,
        img_size=IMG_SIZE,
        timesteps=TIME_STEPS,
        num_channels=3,
        show_process=False,
    )

    images = torch.clamp(images, -1.0, 1.0)

    for j in range(len(images)):
        img = undo_transform(images[j]).squeeze()

        image = Image.fromarray(img)
        image.save(f"images/CIFAR10/CIFAR10_{i*num_per_batch+j}.png")

    logging.info(f"Created CIFAR10 {i+1} {num_per_batch} batch")
