import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .base import BaseTask


class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.shape[1]
        w = img.shape[2]

        mask = np.ones((h, w), np.float32)

        for _n in range(self.n_holes):
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Cifar10Task(BaseTask):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.num_classes = config["model"]["num_classes"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        transform_train_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if self.config.get("data", {}).get("cutout", False):
            n_holes = self.config.get("data", {}).get("n_holes", 1)
            cutout_length = self.config.get("data", {}).get("cutout_length", 16)
            transform_train_list.append(Cutout(n_holes=n_holes, length=cutout_length))

        transform_train = transforms.Compose(transform_train_list)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        return train_loader, test_loader

    def get_model(self) -> nn.Module:
        from model import get_model
        model_name = self.config["model"]["arch"]
        return get_model(model_name, num_classes=self.num_classes)

    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    def get_param_groups(self, model: nn.Module) -> list[dict]:
        hidden_weights = []
        non_hidden_weights = []
        
        for name, param in model.named_parameters():
            if param.ndim >= 2 and any(layer in name for layer in ['conv', 'linear', 'fc']):
                hidden_weights.append(param)
            else:
                non_hidden_weights.append(param)
        
        if hidden_weights and non_hidden_weights:
            return [
                {'params': hidden_weights, 'use_muon': True},
                {'params': non_hidden_weights, 'use_muon': False}
            ]
        else:
            return [{'params': model.parameters(), 'use_muon': False}]


    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool, optimizer_handles_backward: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if not optimizer_handles_backward:
            loss.backward(create_graph=needs_second_order, retain_graph=needs_second_order)

        return outputs.detach(), loss, {}

    def validate_epoch(self, model: nn.Module, test_loader: DataLoader,
                      criterion: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}
