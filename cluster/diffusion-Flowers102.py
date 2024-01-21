import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from diffusers import UNet2DModel
from torch.optim import lr_scheduler
import os
import logging
from datetime import datetime
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

IMG_SIZE = 128
BATCH_SIZE = 32
NO_EPOCHS = 200
TIME_STEPS = 1000


# Transformations
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


class diffusion:
    def __init__(self, timesteps, scheduler_type=None):
        betas = torch.linspace(0.0001, 0.02, timesteps)

        alphas = 1 - betas
        sqrt_alphas = torch.sqrt(alphas)
        alpha_hats = torch.cumprod(1 - betas, axis=0)
        sqrt_alpha_hats = torch.sqrt(alpha_hats)

        sqrt_one_minus_alpha_hats = torch.sqrt(1 - alpha_hats)

        alpha_hats_prev = F.pad(alpha_hats[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1. - alpha_hats_prev) / (1. - alpha_hats)

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
            posterior_variance_t = self.posterior_variance[t]
            return x_t_minus_one + torch.sqrt(posterior_variance_t) * noise
        
    @torch.no_grad()
    def create_image(
        self, model, img_size=32, timesteps=300, num_channels=3, show_process=False
    ):
        img = torch.randn((1, num_channels, img_size, img_size))
        num_images = 10
        stepsize = int(timesteps / num_images)

        if show_process:
            fig = plt.figure(figsize=(15, 2))
            plt.axis("off")

        for i in range(0, timesteps)[::-1]:
            t = torch.full((1,), i, dtype=torch.long)

            img = img.to(device)
            t = t.to(device)

            prediction = model(img, t).sample

            img = self.sample(img, t, prediction)

            if show_process:
                if i % stepsize == 0:
                    plt.subplot(1, num_images, int(i / stepsize) + 1)
                    plt.imshow(undo_transform(img.cpu()[0]))

        if show_process:
            plt.show()
            fig.savefig(f'{datetime.now()}.png')
        return img.cpu()[0]


model = UNet2DModel(
    sample_size=IMG_SIZE,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # block_out_channels = (128, 256, 384, 512, 768),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        # "AttnDownBlock2D",
    ),
    up_block_types=(
        # "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
    ),
    dropout=0.1,
)

if os.path.exists('model_Flowers102_checkpoint') and os.path.isdir('model_Flowers102_checkpoint'):
    model = UNet2DModel().from_pretrained("model_Flowers102_checkpoint")

model.to(device)

diffuser = diffusion(timesteps=TIME_STEPS)

image_data = torchvision.datasets.Flowers102(
    "./data", download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(
    image_data, batch_size=BATCH_SIZE, shuffle=True
)

loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

logging.info("start training")
best_result = 10
for epoch in range(NO_EPOCHS):
    losses = []

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, TIME_STEPS, (len(batch[0]),), device=device).long()

        img_batch_noisy, noise_batch = diffuser.noise_schedule(
            x_0=batch[0], t=t.cpu(), device=device
        )

        predicted_noise_batch = model(img_batch_noisy, t)

        loss = loss_fn(predicted_noise_batch.sample, noise_batch)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()


    logging.info("====================================================================================")
    logging.info(f"Epoch {epoch} | Loss: {sum(losses)/len(losses)} ")
    # logging.info(f"Using {torch.cuda.mem_get_info()[0]/1000000000:.2f} GB")
    # diffuser.create_image(model, img_size=IMG_SIZE, timesteps=TIME_STEPS, num_channels=3, show_process=True)
    if(sum(losses)/len(losses) < best_result):
        model.save_pretrained("model_Flowers102_checkpoint", from_pt=True)

    scheduler.step()

model.save_pretrained("model_Flowers102_checkpoint", from_pt=True)
