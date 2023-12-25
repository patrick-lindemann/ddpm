import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from diffusers import UNet2DModel
import matplotlib.image

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 32
BATCH_SIZE = 64
NO_EPOCHS = 20
TIME_STEPS = 300

class diffusion:
    def __init__(self, timesteps, scheduler_type=None):
        betas = torch.linspace(0.0001, 0.02, timesteps)

        alphas = 1 - betas
        sqrt_alphas = torch.sqrt(alphas)
        alpha_hats = torch.cumprod(1 - betas, axis=0)
        sqrt_alpha_hats = torch.sqrt(alpha_hats)

        sqrt_one_minus_alpha_hats = torch.sqrt(1 - alpha_hats)

        self.betas = betas
        self.alphas = alphas
        self.sqrt_alphas = sqrt_alphas
        self.alpha_hats = alpha_hats
        self.sqrt_alpha_hats = sqrt_alpha_hats
        self.sqrt_one_minus_alpha_hats = sqrt_one_minus_alpha_hats

    def noise_schedule(self, x_0, t, device="cpu"):
        noise = torch.randn(size=x_0.shape)

        alpha_hats_t = self.alpha_hats[t]
        alpha_hats_t = alpha_hats_t.reshape(shape=(t.shape[0], 1, 1, 1))

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
        x_t = x_t.cpu()
        t = t.cpu()
        prediction = prediction.cpu()

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
            return x_t_minus_one + torch.sqrt(betas_t) * noise
        
    def create_image(self, model, img_size=32, timesteps=300, num_channels=3):
        with torch.no_grad():
            img = torch.randn((1, num_channels, img_size, img_size))
            # plt.figure(figsize=(15, 2))
            # plt.axis("off")
            num_images = 10
            stepsize = int(timesteps / num_images)

            for i in range(0, timesteps)[::-1]:
                t = torch.full((1,), i, dtype=torch.long)

                img = img.to(device)
                t = t.to(device)

                prediction = model(img, t).sample

                img = self.sample(img, t, prediction)

                img = torch.clamp(img, -1.0, 1.0)
            #     if i % stepsize == 0:
            #         plt.subplot(1, num_images, int(i / stepsize) + 1)
            #         plt.imshow(undo_transform(img.cpu()[0]))
            # plt.show()
            return img.cpu()[0]



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

model_new = UNet2DModel().from_pretrained('./model_MNIST_checkpoint/')
model_new = model_new.to(device)

for i in range(100):
    img = diffuser.create_image(model_new, img_size=IMG_SIZE, timesteps=TIME_STEPS, num_channels=1)
    img = undo_transform(img).squeeze()
    plt.imsave(f'MNIST-images/MNIST_{i}.png', img, cmap='gray')

