import numpy
import torch

from .scheduler import Scheduler


class SigmoidScheduler(Scheduler):
    def __init__(
        self,
        time_steps: int,
        start: float = -3.0,
        end: float = 3.0,
        tau: float = 1.0,
    ) -> None:
        f = lambda x: 1.0 / (1.0 + numpy.exp(-x / tau))
        t = numpy.linspace(start, end, time_steps)
        y = (f(end) - f(t)) / (f(end) - f(start))
        super().__init__(torch.from_numpy(y))
