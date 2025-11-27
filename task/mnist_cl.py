import gzip
import os
import shutil
from typing import Any

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.download import resumable_download

from .base import BaseTask


class MnistClTask(BaseTask):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.num_classes = 10
        self.fashion_test_loader = self._get_fashion_test_loader()

    def _get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _build_dataloader(self, dataset, shuffle: bool = True):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, drop_last=False
        )

    def _get_fashion_test_loader(self) -> DataLoader:
        transform = self._get_transform()
        fashion_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        return self._build_dataloader(fashion_test, shuffle=False)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        def patched_download_and_extract_archive(url, download_root, filename, md5):
            resumable_download(url, download_root, filename)
            gz_path = os.path.join(download_root, filename)
            raw_path = os.path.join(download_root, filename.replace('.gz', ''))
            if not os.path.exists(raw_path):
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(raw_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        torchvision.datasets.utils.download_and_extract_archive = patched_download_and_extract_archive

        transform = self._get_transform()

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        return self._build_dataloader(train_dataset, shuffle=True), self._build_dataloader(test_dataset, shuffle=False)

    def get_model(self) -> nn.Module:
        from model.resnet_mnist import ResNet18_mnist
        return ResNet18_mnist(num_classes=self.num_classes)

    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_param_groups(self, model: nn.Module) -> list:
        hidden_weights = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'embed' not in n]
        others = [p for n, p in model.named_parameters() if p.ndim < 2 or 'embed' in n]
        return [
            {'params': hidden_weights, 'use_diag_hadron': True, 'lr': 0.02, 'weight_decay': 0.01},
            {'params': others, 'use_diag_hadron': False, 'lr': 3e-4, 'betas': (0.9, 0.95), 'weight_decay': 0.01}
        ]

    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool, optimizer_handles_backward: bool) -> tuple[torch.Tensor, float, dict[str, float]]:
        data, target = batch
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)

        if not optimizer_handles_backward:
            loss.backward(create_graph=needs_second_order)
            optimizer.step()

        return logits.detach(), loss, {}

    def validate_epoch(self, model: nn.Module, test_loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        results = {}

        mnist_loss, mnist_correct, mnist_total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                loss = criterion(logits, target)
                mnist_loss += loss.item()
                _, predicted = logits.max(1)
                mnist_total += target.size(0)
                mnist_correct += predicted.eq(target).sum().item()

        results['mnist_accuracy'] = 100.0 * mnist_correct / mnist_total
        results['mnist_loss'] = mnist_loss / len(test_loader)

        fashion_loss, fashion_correct, fashion_total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in self.fashion_test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                loss = criterion(logits, target)
                fashion_loss += loss.item()
                _, predicted = logits.max(1)
                fashion_total += target.size(0)
                fashion_correct += predicted.eq(target).sum().item()

        results['fashion_accuracy'] = 100.0 * fashion_correct / fashion_total
        results['fashion_loss'] = fashion_loss / len(self.fashion_test_loader)

        results['learning_shock'] = None

        return results
