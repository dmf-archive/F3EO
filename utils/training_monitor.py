import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from rich.console import Console
from rich.table import Table


@dataclass
class TrainingMetrics:
    """训练过程中的完整指标集合"""
    epoch: int
    step: int
    total_steps: int
    loss: float
    accuracy: float | None = None
    perplexity: float | None = None
    grad_norm: float | None = None
    learning_rate: float = 0.0
    iter_per_sec: float = 0.0
    gpu_memory_gb: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_memory_percent: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingMonitor:
    """增强的训练监控器，支持完整的指标追踪和断点续训"""

    def __init__(self, config: dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 指标历史记录
        self.metrics_history: list[TrainingMetrics] = []
        self.epoch_times: list[float] = []
        self.start_time = time.time()

        # 最佳指标追踪
        self.best_metric = float('inf') if config['experiment']['task'] == 'wikitext2' else 0.0
        self.best_epoch = 0
        self.best_step = 0

        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0

        # 检查点管理 - 轮动机制，最多保存3个
        self.checkpoint_files: list[Path] = []
        self.max_checkpoints = 3

        # 性能监控
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

        # 控制台
        self.console = Console()

    def compute_grad_norm(self, model: torch.nn.Module) -> float:
        """计算模型梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def get_gpu_memory_info(self) -> tuple[float, float]:
        """获取GPU内存使用情况（GB和百分比）"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_percent = (allocated / total_memory) * 100
            return allocated, memory_percent
        return 0.0, 0.0

    def get_system_info(self) -> tuple[float, float]:
        """获取系统资源使用情况"""
        cpu_memory = psutil.virtual_memory().percent
        gpu_memory_gb, gpu_memory_percent = self.get_gpu_memory_info()
        return cpu_memory, gpu_memory_gb, gpu_memory_percent

    def start_step(self):
        """记录步骤开始时间"""
        self.step_start_time = time.time()

    def end_step(self, model: torch.nn.Module, loss: float, lr: float) -> TrainingMetrics:
        """记录步骤结束并计算所有指标"""
        step_time = time.time() - self.step_start_time
        iter_per_sec = 1.0 / step_time if step_time > 0 else 0.0

        # 计算梯度范数
        grad_norm = self.compute_grad_norm(model)

        # 获取系统信息
        cpu_memory, gpu_memory_gb, gpu_memory_percent = self.get_system_info()

        # 创建指标对象
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            total_steps=self.total_steps,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=lr,
            iter_per_sec=iter_per_sec,
            gpu_memory_gb=gpu_memory_gb,
            gpu_memory_percent=gpu_memory_percent,
            cpu_memory_percent=cpu_memory,
            timestamp=time.time()
        )

        self.metrics_history.append(metrics)
        self.current_step += 1

        return metrics

    def start_epoch(self, epoch: int, total_steps: int):
        """记录epoch开始"""
        self.current_epoch = epoch
        self.total_steps = total_steps
        self.epoch_start_time = time.time()
        self.console.print(f"\n[bold cyan]Starting Epoch {epoch + 1}[/bold cyan]")

    def end_epoch(self, valid_metric: float) -> dict[str, Any]:
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        # 更新最佳指标
        if self.config['experiment']['task'] == 'wikitext2':
            if valid_metric < self.best_metric:
                self.best_metric = valid_metric
                self.best_epoch = self.current_epoch
        else:
            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                self.best_epoch = self.current_epoch

        return {
            "epoch_time": epoch_time,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch
        }

    def create_live_table(self) -> Table:
        """创建实时更新的Rich表格"""
        table = Table(title="Training Metrics (Real-time)")
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", justify="right", style="magenta", width=12)
        return table

    def update_live_table(self, table: Table, metrics: TrainingMetrics) -> Table:
        """更新实时表格内容"""
        table.rows.clear()

        # 核心训练指标
        table.add_row("Epoch", f"{metrics.epoch + 1}")
        table.add_row("Step", f"{metrics.step}/{metrics.total_steps}")
        table.add_row("Loss", f"{metrics.loss:.4f}")

        if metrics.accuracy is not None:
            table.add_row("Accuracy", f"{metrics.accuracy:.2f}%")
        if metrics.perplexity is not None:
            table.add_row("Perplexity", f"{metrics.perplexity:.2f}")

        # 性能指标
        table.add_row("Grad Norm", f"{metrics.grad_norm:.4f}" if metrics.grad_norm else "N/A")
        table.add_row("Speed", f"{metrics.iter_per_sec:.1f} it/s")
        table.add_row("Learning Rate", f"{metrics.learning_rate:.6f}")

        # 资源使用
        table.add_row("GPU Memory", f"{metrics.gpu_memory_gb:.1f}GB ({metrics.gpu_memory_percent:.1f}%)")
        table.add_row("CPU Memory", f"{metrics.cpu_memory_percent:.1f}%")

        return table

    def get_summary_stats(self) -> dict[str, Any]:
        """获取训练摘要统计"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-100:]  # 最近100步

        return {
            "total_time": time.time() - self.start_time,
            "total_epochs": self.current_epoch + 1,
            "total_steps": self.current_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch + 1,
            "avg_iter_per_sec": sum(m.iter_per_sec for m in recent_metrics) / len(recent_metrics),
            "avg_gpu_memory_gb": sum(m.gpu_memory_gb for m in recent_metrics) / len(recent_metrics),
            "peak_gpu_memory_gb": max(m.gpu_memory_gb for m in self.metrics_history),
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        }

    def save_checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler | None,
                       is_best: bool = False) -> Path:
        """保存训练检查点，使用轮动机制控制磁盘空间"""
        checkpoint = {
            "epoch": epoch,
            "step": self.current_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.config,
            "metrics_history": [m.to_dict() for m in self.metrics_history[-1000:]],  # 保存最近1000步
            "epoch_times": self.epoch_times[-50:],  # 保存最近50个epoch
            "timestamp": time.time()
        }

        # 保存epoch检查点
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)

        # 添加到检查点列表
        self.checkpoint_files.append(checkpoint_path)

        # 如果超过最大数量，删除最老的检查点
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoint_files.pop(0)
            if oldest_checkpoint.exists():
                oldest_checkpoint.unlink()  # 删除文件
                self.console.print(f"[dim]Removed old checkpoint: {oldest_checkpoint.name}[/dim]")

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.console.print("[green]Saved best model checkpoint[/green]")

        # 保存最新检查点（总是更新）
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        self.console.print(f"[dim]Saved checkpoint: {checkpoint_path.name}[/dim]")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """加载训练检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 恢复训练状态
        self.current_epoch = checkpoint["epoch"]
        self.current_step = checkpoint["step"]
        self.best_metric = checkpoint["best_metric"]
        self.best_epoch = checkpoint["best_epoch"]

        # 恢复指标历史
        if "metrics_history" in checkpoint:
            self.metrics_history = [TrainingMetrics(**m) for m in checkpoint["metrics_history"]]
        if "epoch_times" in checkpoint:
            self.epoch_times = checkpoint["epoch_times"]

        return checkpoint

    def save_metrics_log(self) -> Path:
        """保存完整的训练日志"""
        log_path = self.output_dir / "training_log.json"

        log_data = {
            "config": self.config,
            "summary_stats": self.get_summary_stats(),
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "epoch_times": self.epoch_times
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        return log_path
