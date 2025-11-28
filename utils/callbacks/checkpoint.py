from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .base import Callback

if TYPE_CHECKING:
    from .context import TrainerContext


class CheckpointSaver(Callback):
    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files: list[Path] = []

    def on_train_begin(self, context: "TrainerContext"):
        context.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, context: "TrainerContext"):
        pass

    def on_epoch_begin(self, context: "TrainerContext"):
        pass

    def on_epoch_end(self, context: "TrainerContext"):
        pass

    def on_step_begin(self, context: "TrainerContext"):
        pass

    def on_step_end(self, context: "TrainerContext"):
        pass

    def save(self, context: "TrainerContext"):
        checkpoint = {
            "epoch": context.current_epoch,
            "model_state_dict": context.model.state_dict(),
            "optimizer_state_dict": context.optimizer.state_dict(),
            "scheduler_state_dict": context.scheduler.state_dict() if context.scheduler else None,
            "store": context.store
        }

        checkpoint_path = context.output_dir / f"checkpoint_epoch_{context.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)

        self.checkpoint_files.append(checkpoint_path)
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoint_files.pop(0)
            if oldest_checkpoint.exists():
                oldest_checkpoint.unlink()

        latest_path = context.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

    def load(self, context: "TrainerContext") -> bool:
        checkpoint_path = context.output_dir / "latest_checkpoint.pt"
        if not checkpoint_path.exists():
            return False

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        context.model.load_state_dict(checkpoint["model_state_dict"])
        context.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if context.scheduler and checkpoint.get("scheduler_state_dict"):
            context.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        context.store = checkpoint["store"]
        context.current_epoch = checkpoint["epoch"]

        return True
