from pathlib import Path
from typing import Any

import torch

from utils.data import MetricStore


class CheckpointSaver:
    def __init__(self, output_dir: Path, max_checkpoints: int = 3):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler._LRScheduler | None, store: MetricStore):

        # Note: We save the whole store, which might be large.
        # A more advanced version could save only recent history.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "store": store
        }

        # Save current epoch's checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Manage rotation
        self.checkpoint_files.append(checkpoint_path)
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoint_files.pop(0)
            if oldest_checkpoint.exists():
                oldest_checkpoint.unlink()

        # Always update the latest_checkpoint.pt for easy resume
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

    @staticmethod
    def load(checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler._LRScheduler | None) -> dict[str, Any]:

        if not checkpoint_path.exists():
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # The caller is responsible for restoring the store and other states
        return checkpoint
