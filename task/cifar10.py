import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any, Optional
import os


class Cifar10Task:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_classes = config["model"]["num_classes"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.device = config["experiment"]["device"]
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
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
        return get_model("resnet18_cifar", num_classes=self.num_classes)
    
    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   progress_callback=None) -> Dict[str, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 检测是否使用需要二阶梯度的优化器
        needs_second_order = hasattr(optimizer, '__class__') and optimizer.__class__.__name__ in ['F3EO', 'AdaHessian']
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 根据优化器类型决定是否创建计算图
            if needs_second_order:
                loss.backward(create_graph=True)
            else:
                loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 每10个batch更新一次进度
            if progress_callback and (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * correct / total if total > 0 else 0.0
                progress_callback(batch_idx + 1, len(train_loader), loss.item(), current_acc)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def validate_epoch(self, model: nn.Module, test_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
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