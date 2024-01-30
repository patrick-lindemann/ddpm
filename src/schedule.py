from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy
import torch

"""Types"""


ScheduleType = Literal["linear", "cosine", "polynomial", "sigmoid"]


"""Classes"""


class Schedule(ABC):
    """_summary_"""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        t : torch.Tensor
            _description_

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError()


class LinearSchedule(Schedule):
    """__summary__"""

    start: float
    end: float

    def __init__(
        self,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        self.start = start
        self.end = end

    @property
    def name(self) -> str:
        return "linear"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        return t * (self.end - self.start) + self.start


class CosineSchedule(Schedule):
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
        self.start = start
        self.end = end
        self.tau = tau

    @property
    def name(self) -> str:
        return "cosine"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: numpy.cos(x * numpy.pi / 2) ** (2 * self.tau)
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clamp(result, min=0.0, max=1.0)


class PolynomialSchedule(Schedule):
    """__summary__"""

    start: float
    end: float

    def __init__(self, start: float = 0.0, end: float = 1.0, tau: float = 2.0) -> None:
        self.start = start
        self.end = end
        self.tau = tau

    @property
    def name(self) -> str:
        return "polynomial"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: x**self.tau
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clamp(result, min=0.0, max=1.0)


class SigmoidSchedule(Schedule):
    """_summary_"""

    start: float
    end: float
    tau: float

    def __init__(
        self,
        start: float = -3.0,
        end: float = 3.0,
        tau: float = 1.0,
    ) -> None:
        self.start = start
        self.end = end
        self.tau = tau

    @property
    def name(self) -> str:
        return "sigmoid"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        f = lambda x: 1.0 / (1.0 + numpy.exp(-x / self.tau))
        v_start = f(self.start)
        v_end = f(self.end)
        v_t = f(t * (self.end - self.start) + self.start)
        result = (v_t - v_start) / (v_end - v_start)
        return torch.clamp(result, min=0.0, max=1.0)


"""Functions"""


def get_schedule(name: str, **kwargs) -> Schedule:
    """_summary_

    Parameters
    ----------
    name : str
        _description_

    Returns
    -------
    Schedule
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if "tau" in kwargs and (kwargs["tau"] is None or name == "linear"):
        del kwargs["tau"]
    if name == "linear":
        return LinearSchedule(**kwargs)
    elif name == "polynomial":
        return PolynomialSchedule(**kwargs)
    elif name == "cosine":
        return CosineSchedule(**kwargs)
    elif name == "sigmoid":
        return SigmoidSchedule(**kwargs)
    else:
        raise ValueError(f"Invalid schedule name: {name}")
