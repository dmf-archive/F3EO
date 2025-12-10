from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from utils.data import MetricStore


@dataclass
class TrainerContext:
    # Static Configuration
    config: dict[str, Any]
    output_dir: Path
    device: torch.device

    # Core Components
    model: nn.Module
    optimizer: Optimizer
    store: MetricStore
    scheduler: _LRScheduler | None = None

    # Mutable Training State
    current_epoch: int = 0
    total_epochs: int = 0
    global_step: int = 0

    # Task-Specific State (updated during task loops)
    current_task_name: str | None = None
    total_steps_in_epoch: int = 0
    current_step_in_epoch: int = 0

    # Flags
    is_training: bool = False
