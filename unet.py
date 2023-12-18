from torch import nn
import torch
import math


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, embed_time_dims):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_time_dims, out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, t):
        h = self.conv1(x)
        time_emb = self.time_mlp(t)
        h = h + time_emb[..., None, None]
        h = self.conv2(h)

        return h


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.half_dim = dim // 2

    def forward(self, t):
        device = t.device

        embeddings = (
            torch.arange(self.half_dim, device=device)
            * math.log(10000)
            / (self.half_dim - 1)
        )
        embeddings = torch.exp(-embeddings)

        embeddings = t[:, None] * embeddings[None, :]

        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class SimpleUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        features = [64, 128, 256, 512]
        time_emb_dim = 32

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(
                Block(in_channels, feature, time_emb_dim),
            )
            in_channels = feature

        # Upsample
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(Block(feature * 2, feature, time_emb_dim))

        self.bottleneck = Block(features[-1], features[-1] * 2, time_emb_dim)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, t)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = self.ups[i + 1](x, t)

        return self.final_conv(x)
