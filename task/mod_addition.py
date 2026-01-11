import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any
from .base import BaseTask

class ModAdditionTask(BaseTask):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.p = config["task"].get("p", 113)
        self.fraction = config["task"].get("fraction", 0.3)
        self.batch_size = config["data"].get("batch_size", 512)
        self.seed = config["experiment"].get("seed", 42)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        torch.manual_seed(self.seed)
        
        equals_token = self.p
        x, y = torch.meshgrid(torch.arange(self.p), torch.arange(self.p), indexing='ij')
        x = x.flatten()
        y = y.flatten()
        equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
        
        prompts = torch.stack([x, y, equals], dim=1)
        answers = (x + y) % self.p
        
        dataset = TensorDataset(prompts, answers)
        train_size = int(self.fraction * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        return train_loader, test_loader

    def get_model(self) -> nn.Module:
        from model import get_model
        model_config = self.config["model"].copy()
        arch = model_config.pop("arch")
        # num_classes in get_model maps to d_vocab
        return get_model(arch, num_classes=self.p + 1, **model_config)

    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        hidden_weights = []
        non_hidden_weights = []

        for name, param in model.named_parameters():
            # Muon should ONLY be applied to hidden layers (Attention and MLP weights).
            # Embedding (W_E) and Unembed (W_U) must use AdamW to preserve sparse updates.
            if param.ndim >= 2 and "embed" not in name:
                hidden_weights.append(param)
            else:
                non_hidden_weights.append(param)

        return [
            {'params': hidden_weights, 'use_muon': True},
            {'params': non_hidden_weights, 'use_muon': False}
        ]

    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   device: torch.device, needs_second_order: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        # We only care about the prediction at the last position (the '=' token)
        logits = model(inputs)[:, -1]
        loss = criterion(logits, targets)
        
        return logits.detach(), loss, {}

    def validate_epoch(self, model: nn.Module, test_loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)[:, -1]
                loss = criterion(logits, targets)
                
                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
        return {
            "loss": total_loss / total,
            "accuracy": 100.0 * correct / total
        }
