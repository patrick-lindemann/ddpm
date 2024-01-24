from diffusers import UNet2DModel


class DiffusionModel(UNet2DModel):
    """TODO: Summary"""

    def __init__(
        self,
        sample_size: int,
        down_blocks: int = 4,
        up_blocks: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        """_summary_

        Parameters
        ----------
        sample_size : int
            The size of the sample images, which are assumed to be square.
        down_blocks : int, optional
            The number of downward blocks, by default 4
        up_blocks : int, optional
            The number of up blocks, by default 4
        dropout_rate : float, optional
            The dropout rate, by default 0.1
        """
        assert down_blocks > 0, "down_blocks must be greater than 0"
        assert up_blocks > 0, "up_blocks must be greater than 0"
        super().__init__(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            down_block_types=("DownBlock2D", *["AttnDownBlock2D"] * (down_blocks - 1)),
            up_block_types=(*["AttnUpBlock2D"] * (up_blocks - 1), "UpBlock2D"),
            dropout=dropout_rate,
        )
