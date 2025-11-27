from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseTask(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        pass

    @abstractmethod
    def get_param_groups(self, model: nn.Module) -> list[dict]:
        pass

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool, optimizer_handles_backward: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        pass

    @abstractmethod
    def validate_epoch(self, model: nn.Module, test_loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict[str, float]:
        pass
