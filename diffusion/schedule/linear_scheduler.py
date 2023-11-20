import numpy
import torch

from .scheduler import Scheduler


class LinearScheduler(Scheduler):
    def __init__(
        self,
        time_steps: int,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        t = numpy.linspace(start, end, time_steps)
        y = 1.0 - t
        super().__init__(torch.from_numpy(y))
