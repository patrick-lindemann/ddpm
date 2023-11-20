import numpy
import torch

from .scheduler import Scheduler


class CosineScheduler(Scheduler):
    def __init__(
        self,
        time_steps: int,
        start: float = 0.0,
        end: float = 1.0,
        tau: float = 1.0,
    ) -> None:
        f = lambda x: numpy.cos(x * numpy.pi / 2) ** (2 * tau)
        t = numpy.linspace(start, end, time_steps)
        y = (f(end) - f(t)) / (f(end) - f(start))
        super().__init__(torch.from_numpy(y))
