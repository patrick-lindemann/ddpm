import json
import pathlib
from typing import List, TypedDict

import torch
from diffusers import UNet2DModel
from typing_extensions import Literal

"""Types"""


DownBlockType = Literal["DownBlock2D", "AttnDownBlock2D"]
UpBlockType = Literal["UpBlock2D", "AttnUpBlock2D"]
TimeEmbeddingType = Literal["positional"]


"""Classes"""


class ModelConfig(TypedDict):
    image_size: int
    down_blocks_types: List[DownBlockType]
    up_blocks_types: List[UpBlockType]
    layers_per_block: int
    time_embedding_type: TimeEmbeddingType
    dropout_rate: float


class DenoisingUNet2DModel(UNet2DModel):
    """__summary__"""

    @classmethod
    def load(
        cls,
        config_path: pathlib.path,
        weights_path: pathlib.Path,
    ) -> "DenoisingUNet2DModel":
        with open(config_path, "r") as file:
            config = json.load(file)
        model = cls(**config)
        model.load_state_dict(torch.load(weights_path))
        return model

    def __init__(
        self,
        image_size: int,
        down_block_types: List[DownBlockType] = [
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ],
        up_block_types: List[UpBlockType] = [
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ],
        layers_per_block: int = 2,
        time_embedding_type: TimeEmbeddingType = "positional",
        dropout_rate: float = 0.1,
    ) -> None:
        assert len(down_block_types) > 0
        assert len(up_block_types) > 0
        assert len(down_block_types) == len(up_block_types)
        super().__init__(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            time_embedding_type=time_embedding_type,
            dropout=dropout_rate,
        )

    def save(self, path: pathlib.Path) -> None:
        torch.save(self.state_dict(), path)
