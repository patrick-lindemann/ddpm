from abc import ABC, abstractmethod
from typing import Literal

import numpy
import torch

"""Types"""


ScheduleType = Literal["linear", "cosine", "polynomial", "sigmoid"]


"""Constants"""


CLIP_MIN = 0.000001
CLIP_MAX = 0.999999


"""Classes"""


class Schedule(ABC):
    """_summary_"""

    start: float
    end: float
    tau: float | None

    def __init__(self, start: float, end: float, tau: float | None) -> None:
        self.start = start
        self.end = end
        self.tau = tau

    @property
    @abstractmethod
    def type(self) -> str:
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

    def __init__(self, start: float = 0.0, end: float = 1.0) -> None:
        super().__init__(start, end, None)

    @property
    def type(self) -> str:
        return "linear"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
        result = (t - self.start) / (self.end - self.start)
        return torch.clamp(result, CLIP_MIN, CLIP_MAX)


class CosineSchedule(Schedule):
    """_summary_"""

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
        return torch.clamp(result, CLIP_MIN, CLIP_MAX)


class PolynomialSchedule(Schedule):
    """__summary__"""

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
        return torch.clamp(result, CLIP_MIN, CLIP_MAX)


class SigmoidSchedule(Schedule):
    """_summary_"""

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
        return torch.clamp(result, CLIP_MIN, CLIP_MAX)


"""Functions"""


def get_schedule(type: ScheduleType, **kwargs) -> Schedule:
    """_summary_

    Parameters
    ----------
    type : str
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
