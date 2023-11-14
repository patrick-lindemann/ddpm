from typing import Optional

import matplotlib.pyplot as plt
import torch


def show_image(image: torch.Tensor, outfile: Optional[str] = None) -> None:
    """_summary_

    Parameters
    ----------
    image : torch.Tensor
        _description_
    outfile : Optional[str], optional
        _description_, by default None
    """
    plt.imshow(image)
