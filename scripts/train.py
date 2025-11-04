import argparse
import importlib
import time
import toml
import torch
from pathlib import Path
from typing import Dict, Any
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel


console = Console()


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
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


def train(config: Dict[str, Any], task_class) -> None:
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
    log_every = config["train"]["log_every"]
    
    console.print(Panel.fit(
        f"[bold cyan]Task:[/bold cyan] {config['experiment']['task']}\n"
        f"[bold cyan]Model:[/bold cyan] {config['model']['arch']}\n"
        f"[bold cyan]Optimizer:[/bold cyan] {config['optimizer']['name']}\n"
        f"[bold cyan]Epochs:[/bold cyan] {epochs}\n"
        f"[bold cyan]Device:[/bold cyan] {device}",
        title="[bold]F3EO-Bench Training[/bold]",
        border_style="cyan"
    ))
    
    best_metric = float('inf') if config['experiment']['task'] == 'wikitext2' else 0.0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        for epoch in range(epochs):
            epoch_task = progress.add_task(f"Epoch {epoch+1}/{epochs}", total=epochs)
            
            start_time = time.time()
            
            train_results = task.train_epoch(model, train_loader, optimizer, criterion)
            valid_results = task.validate_epoch(model, valid_loader, criterion)
            
            epoch_time = time.time() - start_time
            
            if scheduler:
                scheduler.step()
            
            table = Table(title=f"Epoch {epoch+1} Results")
            table.add_column("Split", style="cyan")
            table.add_column("Loss", justify="right", style="magenta")
            if config['experiment']['task'] == 'wikitext2':
                table.add_column("Perplexity", justify="right", style="green")
                table.add_row("Train", f"{train_results['loss']:.4f}", f"{train_results['perplexity']:.2f}")
                table.add_row("Valid", f"{valid_results['loss']:.4f}", f"{valid_results['perplexity']:.2f}")
                
                if valid_results['perplexity'] < best_metric:
                    best_metric = valid_results['perplexity']
            else:
                table.add_column("Accuracy", justify="right", style="green")
                table.add_row("Train", f"{train_results['loss']:.4f}", f"{train_results['accuracy']:.2f}%")
                table.add_row("Valid", f"{valid_results['loss']:.4f}", f"{valid_results['accuracy']:.2f}%")
                
                if valid_results['accuracy'] > best_metric:
                    best_metric = valid_results['accuracy']
            
            console.print(table)
            console.print(f"[dim]Epoch time: {epoch_time:.2f}s[/dim]")
            
            progress.update(epoch_task, completed=epoch+1)
    
    console.print(f"\n[bold green]Training completed![/bold green]")
    if config['experiment']['task'] == 'wikitext2':
        console.print(f"[bold]Best validation perplexity: {best_metric:.2f}[/bold]")
    else:
        console.print(f"[bold]Best validation accuracy: {best_metric:.2f}%[/bold]")


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