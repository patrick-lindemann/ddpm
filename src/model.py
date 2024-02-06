import json
import pathlib
from typing import List, TypedDict

import torch
from diffusers import UNet2DModel
from typing_extensions import Literal

"""Types"""


DownBlockType = Literal["DownBlock2D", "AttnDownBlock2D"]
"""
The types of downward blocks that can be used in the model. 
"""

UpBlockType = Literal["UpBlock2D", "AttnUpBlock2D"]
"""
The types of upward blocks that can be used in the model.
"""

TimeEmbeddingType = Literal["positional"]
"""
The types of time embeddings that can be used in the model.
"""


"""Constants"""


CONFIG_FILE_NAME = "model.config.json"
"""
The name of the file in which the model configuration is saved.
"""

WEIGHTS_FILE_NAME = "weights.pt"
"""
The name of the file in which the model weights are saved.
"""


"""Classes"""


class DenoisingUNet2DConfig(TypedDict):
    image_size: int
    down_blocks_types: List[DownBlockType]
    up_blocks_types: List[UpBlockType]
    layers_per_block: int
    time_embedding_type: TimeEmbeddingType
    dropout_rate: float


class DenoisingUNet2D(UNet2DModel):
    """A denoising U-Net model for 2D images that predicts the noise of an image
    at a given time step of the diffusion process."""

    _config: DenoisingUNet2DConfig

    @classmethod
    def load(cls, dir_path: pathlib.Path) -> "DenoisingUNet2D":
        """Load a pre-trained model from a directory containing a model.config.json and a weights.pt file.

        Parameters
        ----------
        dir_path : pathlib.Path
            The model directory.
        Returns
        -------
        DenoisingUNet2D
            The loaded model.
        """
        config_path = dir_path / CONFIG_FILE_NAME
        assert config_path.exists()
        weights_path = dir_path / WEIGHTS_FILE_NAME
        assert weights_path.exists()
        with open(config_path, "r") as file:
            config = json.load(file)
        model = cls(**config)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
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
            "AttnUpBlock2D",
            "UpBlock2D",
        ],
        layers_per_block: int = 2,
        time_embedding_type: TimeEmbeddingType = "positional",
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize a denoising U-Net model.

        Parameters
        ----------
        image_size : int
            The size of the input images.
        down_block_types : List[DownBlockType], optional
            The downward blocks of the model, by default ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"]
        up_block_types : List[UpBlockType], optional
            The upward blocks of the model, by default ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"]
        layers_per_block : int, optional
            The number of residual layers in each block, by default 2
        time_embedding_type : TimeEmbeddingType, optional
            The type of time embedding to use, by default "positional"
        dropout_rate : float, optional
            The dropout rate, by default 0.1
        """
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
        self._config: DenoisingUNet2DConfig = {
            "image_size": image_size,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "layers_per_block": layers_per_block,
            "time_embedding_type": time_embedding_type,
            "dropout_rate": dropout_rate,
        }

    def save(self, dir_path: pathlib.Path) -> None:
        """Save the model configuration and state to a directory.

        Parameters
        ----------
        dir_path : pathlib.Path
            The directory to save the model to.
        """
        config_path = dir_path / CONFIG_FILE_NAME
        weights_path = dir_path / WEIGHTS_FILE_NAME
        with open(config_path, "w") as file:
            json.dump(self._config, file)
        torch.save(self.state_dict(), weights_path)
