from abc import ABC, abstractmethod
from typing import Literal

import numpy
import torch

"""Types"""


ScheduleType = Literal["linear", "cosine", "polynomial", "sigmoid"]


"""Constants"""


CLIP_MIN = 0.00001
"""
The minimum value for noise variances, which must be greater than zero to avoid numerical issues.
"""

CLIP_MAX = 0.99999
"""
The maximum value for noise variances, which must be less than one to avoid numerical issues.
"""


"""Classes"""


class Schedule(ABC):
    """The base class for schedules."""

    start: float
    end: float
    tau: float | None

    def __init__(self, start: float, end: float, tau: float | None) -> None:
        """Initialize the schedule.

        Parameters
        ----------
        start : float
            The start value of the schedule.
        end : float
            The end value of the schedule.
        tau : float | None
            The tau value of the schedule.
        """
        self.start = start
        self.end = end
        self.tau = tau

    @property
    @abstractmethod
    def type(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Apply the schedule to a tensor of time steps to get the noise variances beta.

        Parameters
        ----------
        t : torch.Tensor
            A tensor of time steps.

        Returns
        -------
        torch.Tensor
            A tensor containing the noise variances beta_t.

        Raises
        ------
        NotImplementedError
            This method must be implemented by the derived classes.
        """
        raise NotImplementedError()


class LinearSchedule(Schedule):
    """A linear schedule"""

    def __init__(self, start: float = 0.0, end: float = 1.0) -> None:
        super().__init__(start, end, None)

    @property
    def type(self) -> str:
        return "linear"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        result = self.start + t * (self.end - self.start)
        return torch.clip(result, CLIP_MIN, CLIP_MAX)


class PolynomialSchedule(Schedule):
    """A schedule using an (even-degree) polynomial function."""

    def __init__(self, start: float = 0.0, end: float = 1.0, tau: float = 2.0) -> None:
        super().__init__(start, end, tau)

    @property
    def type(self) -> str:
        return "polynomial"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: x**self.tau
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clip(result, CLIP_MIN, CLIP_MAX)


class CosineSchedule(Schedule):
    """A schedule using the cosine function, as proposed by [Nichol et. al (2021)](https://arxiv.org/pdf/2102.09672.pdf)."""

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
        tau: float = 1.0,
    ) -> None:
        super().__init__(start, end, tau)

    @property
    def type(self) -> str:
        return "cosine"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: numpy.cos(x * numpy.pi / 2) ** (2 * self.tau)
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clip(result, CLIP_MIN, CLIP_MAX)


class SigmoidSchedule(Schedule):
    """A schedule using the sigmoid function, as proposed by [Ting Cheng (2023)](https://arxiv.org/pdf/2301.10972.pdf)."""

    def __init__(
        self,
        start: float = -3.0,
        end: float = 3.0,
        tau: float = 1.0,
    ) -> None:
        super().__init__(start, end, tau)

    @property
    def type(self) -> str:
        return "sigmoid"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: 1.0 / (1.0 + numpy.exp(-x / self.tau))
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clip(result, CLIP_MIN, CLIP_MAX)


"""Functions"""


def get_schedule(type: ScheduleType, **kwargs) -> Schedule:
    """Initialize a schedule by name and parameters.

    Parameters
    ----------
    type : str
        The name of the schedule.

    Returns
    -------
    Schedule
        The initialized schedule.

    Raises
    ------
    ValueError
        If the schedule name is invalid.
    """
    if "tau" in kwargs and (kwargs["tau"] is None or type == "linear"):
        del kwargs["tau"]
    if type == "linear":
        return LinearSchedule(**kwargs)
    elif type == "polynomial":
        return PolynomialSchedule(**kwargs)
    elif type == "cosine":
        return CosineSchedule(**kwargs)
    elif type == "sigmoid":
        return SigmoidSchedule(**kwargs)
    else:
        raise ValueError(f"Invalid schedule name: {type}")
