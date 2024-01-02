from typing import List

import torch
from torch import nn

from diffusion.embedding import SinusodialTimeEmbedding, TimeEmbedding


class Block(nn.Module):
    """_summary_"""

    conv1: nn.Module
    conv2: nn.Module
    time_mlp: nn.Module

    def __init__(
        self, in_channels: int, out_channels: int, time_embedding: TimeEmbedding
    ) -> None:
        super().__init__()
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
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding.dim, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        t_embedded = self.time_mlp(t)
        h = h + t_embedded[..., None, None]
        h = self.conv2(h)
        return h


class SimpleUNet(nn.Module):
    """__summary__"""

    down_layers: nn.ModuleList
    bottleneck: nn.Module
    up_layers: nn.ModuleList
    out_layer: nn.Module
    pool: nn.Module
    time_mlp: nn.Module

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: List[int] = [64, 128, 256, 512],
        time_embedding=SinusodialTimeEmbedding(32),
    ):
        super().__init__()
        # Prepare the downward layers
        self.down_layers = nn.ModuleList()
        for feature in features:
            self.down_layers.append(Block(in_channels, feature, time_embedding.dim))
            in_channels = feature
        # Prepare bottleneck
        self.bottleneck = Block(features[-1], features[-1] * 2, time_embedding.dim)
        # Prepare upward layers
        self.up_layers = nn.ModuleList()
        for feature in reversed(features):
            self.up_layers.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.up_layers.append(Block(feature * 2, feature, time_embedding.dim))
        # Prepare the output layer
        self.out_layer = nn.Conv2d(features[0], out_channels, 1)
        # Prepare other layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_mlp = nn.Sequential(
            time_embedding,
            nn.Linear(time_embedding.dim, time_embedding.dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embedded = self.time_mlp(t)
        skip_connections: List[torch.Tensor] = []
        # Downward pass
        for down_layer in self.down_layers:
            x = down_layer(x, t_embedded)
            skip_connections.append(x)
            x = self.pool(x)
        # Bottleneck
        x = self.bottleneck(x, t_embedded)
        # Upward pass
        for up_layer, residual_layer in zip(self.up_layers[0::2], self.up_layers[1::2]):
            x = up_layer(x)
            skip_connection = skip_connections.pop()
            x = torch.cat((x, skip_connection), dim=1)
            x = residual_layer(x, t)
        # Final convolution
        return self.out_layer(x)
