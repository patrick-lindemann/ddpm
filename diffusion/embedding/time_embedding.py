import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """__summary__"""

    dim: int

    def __init__(self, dim: int, device: torch.device = torch.cpu) -> None:
        super().__init__()
        self.dim = dim
        self.device = device


class SinusodialTimeEmbedding(TimeEmbedding):
    """__summary__"""

    _embedding_vector: torch.Tensor

    def __init__(self, dim: int, device: torch.device = torch.cpu) -> None:
        super().__init__(dim=dim, device=device)
        self._embedding_vector = torch.exp(
            -torch.arange(dim // 2, device=device) * torch.log(10000) / (dim // 2 - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embedded = x[:, None] * self._embedding_vector[None, :]
        x_embedded = torch.cat((torch.sin(x_embedded), torch.cos(x_embedded)), dim=-1)
        return x_embedded
