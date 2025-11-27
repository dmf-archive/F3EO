from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from utils.data import MetricStore, StepMetric

class Callback(ABC):

    @abstractmethod
    def on_train_begin(self, store: "MetricStore", **kwargs):
        pass

    @abstractmethod
    def on_train_end(self, store: "MetricStore", **kwargs):
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, total_steps: int, **kwargs):
        pass

    @abstractmethod
    def on_epoch_end(self, store: "MetricStore", **kwargs):
        pass

    @abstractmethod
    def on_step_begin(self, step: int, **kwargs):
        pass

    @abstractmethod
    def on_step_end(self, step_metric: "StepMetric", total_steps: int, **kwargs):
        pass

    @abstractmethod
    def save(self, epoch: int, model: "torch.nn.Module", optimizer: "torch.optim.Optimizer",
             scheduler: "torch.optim.lr_scheduler._LRScheduler | None", store: "MetricStore", **kwargs):
        pass

    @abstractmethod
    def load(self, path: str, model: "torch.nn.Module", optimizer: "torch.optim.Optimizer",
             scheduler: "torch.optim.lr_scheduler._LRScheduler | None", **kwargs) -> dict | None:
        return None
