import numpy
import torch

from .scheduler import Scheduler


class CosineScheduler(Scheduler):
    """_summary_"""

    start: float
    end: float
    tau: float

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
        tau: float = 1.0,
    ) -> None:
        """_summary_

        Parameters
        ----------
        start : float, optional
            _description_, by default 0.0
        end : float, optional
            _description_, by default 1.0
        tau : float, optional
            _description_, by default 1.0
        """
        self.start = start
        self.end = end
        self.tau = tau

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
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
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: numpy.cos(x * numpy.pi / 2) ** (2 * self.tau)
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clamp(result, min=0.0, max=1.0)
