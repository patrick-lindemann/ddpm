from diffusers import UNet2DModel


class DiffusionModel(UNet2DModel):
    """TODO: Summary"""

    def __init__(
        self, sample_size: int, num_blocks: int = 4, dropout: float = 0.1
    ) -> None:
        """_summary_

        Parameters
        ----------
        sample_size : int
            The size of the sample images, which are assumed to be square.
        num_blocks : int, optional
            The number of down- and upward blocks, by default 4
        dropout : float, optional
            The dropout rate, by default 0.1
        """
        assert num_blocks > 0, "num_blocks must be greater than 0"
        super().__init__(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            down_block_types=("DownBlock2D", *["AttnDownBlock2D"] * (num_blocks - 1)),
            up_block_types=(*["AttnUpBlock2D"] * (num_blocks - 1), "UpBlock2D"),
        )
