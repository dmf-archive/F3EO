from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .base import BaseTask


class MnistClTask(BaseTask):
    """
    极简持续学习沙盒：MNIST → FashionMNIST
    """
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.num_classes = 10

    def _get_transform(self):
        """28×28 灰度图 → 展平 784 向量，归一化到 [0,1]"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))  # 展平
        ])

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = True):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        # This task is special, it returns a list of loaders.
        # The main train loop will need to be adapted to handle this.
        root = Path("./data")
        transform = self._get_transform()
        mnist_train = datasets.MNIST(root, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root, train=False, transform=transform)
        return self._build_dataloader(mnist_train, shuffle=True), self._build_dataloader(mnist_test, shuffle=False)

    def get_model(self) -> nn.Module:
        """
        返回 Micro-SwinViT：
        4 层, dim=96, head=4, patch=4, 输入 28×28 灰度图
        总参数量 ≈ 80 K，10 分类输出
        """
        from model.swin import swin_t
        model = swin_t(
            num_classes=self.num_classes,
            hidden_dim=96,
            layers=(2, 2, 2, 2),  # 每层 2 个 block，共 4 层
            heads=(3, 6, 12, 24),
            downscaling_factors=(4, 2, 2, 2),  # 首下采样 4→14→7→4→2
            window_size=4,  # 7 太大，改用 4
            channels=1  # 灰度
        )
        return model

    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item() * 100.0

    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, pi_config: dict[str, Any] | None) -> tuple[torch.Tensor, float]:
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        needs_second_order = pi_config is not None

        optimizer.zero_grad()
        logits = model(data.unsqueeze(1))
        loss = criterion(logits, target)
        loss.backward(create_graph=needs_second_order)
        optimizer.step()

        return logits.detach(), loss.item()

    def validate_epoch(self, model: nn.Module, test_loader: DataLoader,
                       criterion: nn.Module) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data.unsqueeze(1))
                loss = criterion(logits, target)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        return {"loss": avg_loss, "accuracy": accuracy}
