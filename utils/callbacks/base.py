from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import TrainerContext


class Callback(ABC):

    @abstractmethod
    def on_train_begin(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def on_train_end(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def on_epoch_begin(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def on_epoch_end(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def on_step_begin(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def on_step_end(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def save(self, context: "TrainerContext"):
        pass

    @abstractmethod
    def load(self, context: "TrainerContext") -> bool:
        return False
