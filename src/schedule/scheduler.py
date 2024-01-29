from abc import ABC, abstractmethod

import torch


class Scheduler(ABC):
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
