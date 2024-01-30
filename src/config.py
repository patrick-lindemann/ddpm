import pathlib
from typing import List, Literal, TypedDict


class ModelConfig(TypedDict):
    pass


class ScheduleConfig(TypedDict):
    pass


class TrainingConfig(TypedDict):
    pass


class Metadata:
    run_name: str
    seed: int

    scheduler_type: Literal["linear", "polynomial", "cosine", "sigmoid"]
    scheduler_steps: int
    scheduler_start: float
    scheduler_end: float
    scheduler_tau: float

    model_sample_size: int
    model_layers_per_block: int
    model_down_blocks: List[Literal["AttnDownBlock2D", "DownBlock2D"]]
    model_up_blocks: List[Literal["AttnUpBlock2D", "UpBlock2D"]]

    training_epochs: int
    training_batch_size: int
    training_learning_rate: float
    training_pass: int
    train_mse: List

    training_epochs: int
    training_batch_size: int
    training_learning_rate: float
    training_learning_rate_stepsize: int

    @classmethod
    def load(cls, path: pathlib.Path) -> "Metadata":
        pass

    def __init__(self, **kwargs) -> None:
        pass

    def save(self, path: pathlib.Path) -> None:
        pass
