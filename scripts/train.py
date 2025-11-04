import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import toml
import torch
from rich.console import Console

# 忽略 timm 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from utils.early_stop import EarlyStop
from utils.report_generator import ReportGenerator
from utils.training_monitor import TrainingMonitor

console = Console()


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return toml.load(f)


def get_task_module(task_name: str):
    try:
        module = importlib.import_module(f"task.{task_name}")
        task_class = getattr(module, f"{task_name.capitalize()}Task")
        return task_class
    except (ImportError, AttributeError) as e:
        console.print(f"[red]Error loading task module '{task_name}': {e}[/red]")
        sys.exit(1)


def create_optimizer_scheduler(model, config, train_loader):
    optimizer_name = config["optimizer"]["name"]
    lr = config["optimizer"]["lr"]
    weight_decay = config["optimizer"]["weight_decay"]

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdaHessian":
        from optimizer.ada_hessian import Adahessian
        optimizer = Adahessian(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdaFisher":
        from optimizer.ada_fisher import AdaFisher
        optimizer = AdaFisher(model, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "F3EO":
        from optimizer.F3EO import F3EO
        optimizer = F3EO(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = None
    if "scheduler" in config:
        scheduler_name = config["scheduler"].get("name", "none")
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["scheduler"]["T_max"]
            )
        elif scheduler_name == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=config["scheduler"]["milestones"],
                gamma=config["scheduler"]["gamma"]
            )

    return optimizer, scheduler


def train(config: dict[str, Any], task_class) -> None:
    device = torch.device(config["experiment"]["device"])
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    task = task_class(config)
    train_loader, valid_loader = task.get_dataloaders()
    model = task.get_model().to(device)
    criterion = task.get_criterion()
    optimizer, scheduler = create_optimizer_scheduler(model, config, train_loader)

    epochs = config["train"]["epochs"]

    # 设置输出目录
    output_dir = Path("outputs") / config["experiment"]["task"] / config["optimizer"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化训练监控器（支持断点续训）
    monitor = TrainingMonitor(config, output_dir)

    # 检查是否存在检查点文件
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    start_epoch = 0

    if checkpoint_path.exists():
        console.print("[yellow]Found checkpoint, resuming training...[/yellow]")
        checkpoint = monitor.load_checkpoint(checkpoint_path)

        # 恢复模型状态
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        console.print(f"[green]Resumed from epoch {start_epoch}[/green]")

    # 初始化早停机制
    early_stop_patience = config.get("early_stop", {}).get("patience", 10)
    early_stop_threshold = config.get("early_stop", {}).get("threshold", -1.0)
    early_stop_mode = "min" if config['experiment']['task'] == 'wikitext2' else "max"
    early_stop = EarlyStop(patience=early_stop_patience, threshold=early_stop_threshold, mode=early_stop_mode)
    early_stop.best_metric = monitor.best_metric  # 恢复早停状态

    # 初始化报告生成器
    report_gen = ReportGenerator(config, output_dir)

    console.print(Panel.fit(
        f"[bold cyan]Task:[/bold cyan] {config['experiment']['task']}\n"
        f"[bold cyan]Model:[/bold cyan] {config['model']['arch']}\n"
        f"[bold cyan]Optimizer:[/bold cyan] {config['optimizer']['name']}\n"
        f"[bold cyan]Epochs:[/bold cyan] {epochs}\n"
        f"[bold cyan]Device:[/bold cyan] {device}\n"
        f"[bold cyan]Output:[/bold cyan] {output_dir}",
        title="[bold]F3EO-Bench Training[/bold]",
        border_style="cyan"
    ))

    best_metric = float('inf') if config['experiment']['task'] == 'wikitext2' else 0.0
    best_epoch = 0

    # 创建实时表格用于 step 级监控

    def create_metrics_table():
        table = Table(title="Training Metrics (Step-level)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        return table

    def update_metrics_table(table, epoch, step, total_steps, loss, acc, lr):
        table.rows.clear()
        table.add_row("Epoch", f"{epoch}/{epochs}")
        table.add_row("Step", f"{step}/{total_steps}")
        table.add_row("Loss", f"{loss:.4f}")
        table.add_row("Accuracy", f"{acc:.2f}%")
        table.add_row("Learning Rate", f"{lr:.6f}")
        return table

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall_epoch_task = progress.add_task("Training Progress", total=epochs)

        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            current_epoch = epoch + 1

            # Step 级进度条和实时监控
            def step_progress_callback(step, total_steps, loss, metric, grad_norm=None, items_per_sec=None):
                if step % 10 == 0:  # 每10步更新一次
                    base_msg = f"[dim]Epoch {current_epoch}/{epochs} | Step {step}/{total_steps} | Loss: {loss:.4f}"
                    if config['experiment']['task'] == 'wikitext2':
                        base_msg += f" | PPL: {metric:.2f}"
                    else:
                        base_msg += f" | Acc: {metric:.2f}%"

                    if grad_norm is not None:
                        base_msg += f" | Grad: {grad_norm:.4f}"
                    if items_per_sec is not None:
                        base_msg += f" | {items_per_sec:.1f}it/s"

                    console.print(base_msg + "[/dim]")

            train_results = task.train_epoch(model, train_loader, optimizer, criterion, step_progress_callback)
            valid_results = task.validate_epoch(model, valid_loader, criterion)

            epoch_time = time.time() - start_time

            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 记录指标到报告生成器
            report_gen.log_epoch(epoch, train_results, valid_results, current_lr, epoch_time)

            if scheduler:
                scheduler.step()

            # 更新最佳指标
            monitor_metric = valid_results.get("perplexity", valid_results.get("accuracy"))
            is_best = False
            if config['experiment']['task'] == 'wikitext2':
                if monitor_metric < best_metric:
                    best_metric = monitor_metric
                    best_epoch = epoch + 1
                    is_best = True
            else:
                if monitor_metric > best_metric:
                    best_metric = monitor_metric
                    best_epoch = epoch + 1
                    is_best = True

            table = Table(title=f"Epoch {epoch+1} Results")
            table.add_column("Split", style="cyan")
            table.add_column("Loss", justify="right", style="magenta")
            if config['experiment']['task'] == 'wikitext2':
                table.add_column("Perplexity", justify="right", style="green")
                table.add_row("Train", f"{train_results['loss']:.4f}", f"{train_results['perplexity']:.2f}")
                table.add_row("Valid", f"{valid_results['loss']:.4f}", f"{valid_results['perplexity']:.2f}")
            else:
                table.add_column("Accuracy", justify="right", style="green")
                table.add_row("Train", f"{train_results['loss']:.4f}", f"{train_results['accuracy']:.2f}%")
                table.add_row("Valid", f"{valid_results['loss']:.4f}", f"{valid_results['accuracy']:.2f}%")

            console.print(table)
            console.print(f"[dim]Epoch time: {epoch_time:.2f}s, LR: {current_lr:.6f}[/dim]")

            progress.update(overall_epoch_task, advance=1)

            # 保存检查点
            if (epoch + 1) % config["train"]["ckpt_every"] == 0 or is_best:
                monitor.save_checkpoint(epoch, model, optimizer, scheduler, is_best=is_best)

            # 早停检查
            if early_stop(monitor_metric):
                console.print(f"[yellow]Early stopping triggered at epoch {epoch+1}[/yellow]")
                break

            # 生成当前 epoch 的报告
            report_gen.generate_summary()

    console.print("\n[bold green]Training completed![/bold green]")
    console.print(f"[bold]Best validation {metric_name.lower()}: {best_metric:.2f} at epoch {best_epoch}[/bold]")
    console.print(f"[dim]Report saved to: {output_dir}/summary.md[/dim]")


def main():
    parser = argparse.ArgumentParser(description="F3EO-Bench Training Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML configuration file")

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    config = load_config(config_path)
    task_name = config["experiment"]["task"]

    task_class = get_task_module(task_name)
    train(config, task_class)


if __name__ == "__main__":
    main()
