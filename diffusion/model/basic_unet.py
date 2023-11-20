from typing import List

import torch


class BasicUNet(torch.nn.Module):
    """__summary__"""

    down_layers: torch.nn.ModuleList
    up_layers: torch.nn.ModuleList
    activation_function: torch.nn.SiLU
    downscale: torch.nn.MaxPool2d
    upscale: torch.nn.Upsample

    def __init__(self, channels_in: int = 1, channels_out: int = 1) -> None:
        """_summary_

        Parameters
        ----------
        in_channels : int, optional
            _description_, by default 1
        out_channels : int, optional
            _description_, by default 1
        """
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(channels_in, 32, kernel_size=5, padding=2),
                torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
                torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
                torch.nn.Conv2d(64, 32, kernel_size=5, padding=2),
                torch.nn.Conv2d(32, channels_out, kernel_size=5, padding=2),
            ]
        )
        self.activation_func = torch.nn.SiLU()
        self.downscale = torch.nn.MaxPool2d(2)
        self.upscale = torch.nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        # Residual connections
        h: List[torch.Tensor] = []
        # Downward pass
        for i, down_layer in enumerate(self.down_layers):
            x = self.activation_func(down_layer(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)
        # Upward pass
        for i, up_layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x[None, :, :, :])[0]
                x += h.pop()
            x = self.activation_func(up_layer(x))
        return x
