import torch

from .scheduler import Scheduler


class LinearScheduler(Scheduler):
    """__summary__"""

    start: float
    end: float

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        """_summary_

        Parameters
        ----------
        start : float, optional
            _description_, by default 0.0
        end : float, optional
            _description_, by default 1.0
        """
        self.start = start
        self.end = end

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        t : torch.Tensor
            _description_. Between 0 and 1

        Returns
        -------
        torch.Tensor
            _description_
        """
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        return t * (self.end - self.start) + self.start
