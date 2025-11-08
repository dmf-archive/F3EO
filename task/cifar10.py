import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Cutout:
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
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


class Cifar10Task:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.num_classes = config["model"]["num_classes"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.device = config["experiment"]["device"]

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:

        # 基础变换
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        # 添加ToTensor和Normalize
        transform_train_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 根据配置添加cutout增强，在ToTensor和Normalize之后
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

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   monitor: Any, progress_callback=None, optimizer_tags=None) -> dict[str, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        last_callback_time = time.time()

        accepts_pi_signal = optimizer_tags.get("accepts_pi_signal", False) if optimizer_tags else False
        needs_second_order = optimizer_tags.get("requires_second_order", False) if optimizer_tags else False

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward(create_graph=needs_second_order)

            # --- PI Calculation and Optimizer Step ---
            # For CIFAR-10, we pass `outputs` (logits) to the monitor
            metrics = monitor.end_step(model, loss.item(), optimizer.param_groups[0]['lr'], outputs)

            step_args = {}
            if accepts_pi_signal:
                step_args['effective_gamma'] = metrics.effective_gamma

            optimizer.step(**step_args)
            # --- End of PI Calculation ---

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if progress_callback and (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * correct / total if total > 0 else 0.0
                current_time = time.time()
                time_elapsed = current_time - last_callback_time
                steps_processed = 10
                steps_per_sec = steps_processed / time_elapsed if time_elapsed > 0 else 0.0
                last_callback_time = current_time

                progress_callback(
                    monitor.current_epoch, batch_idx + 1, len(train_loader), loss.item(), current_acc,
                    metrics.grad_norm, steps_per_sec, metrics.pi, metrics.effective_gamma, metrics.entropy
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate_epoch(self, model: nn.Module, test_loader: DataLoader,
                      criterion: nn.Module) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}
