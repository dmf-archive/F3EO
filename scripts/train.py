import argparse
import sys
from pathlib import Path
from typing import Any

import toml
import torch

from optimizer import get_optimizer
from utils.callbacks.checkpoint import CheckpointSaver
from utils.callbacks.console import ConsoleLogger
from utils.callbacks.markdown import MDLogger
from utils.trainer import Trainer


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return toml.load(f)


def create_optimizer(model, task, config):
    from optimizer import get_optimizer
    optimizer_config = config["optimizer"]
    opt_name = optimizer_config["name"]

    opt_factory_config = optimizer_config.copy()
    opt_factory_config.pop("name")

    opt_factory_config["model"] = model

    param_groups = task.get_param_groups(model)

    return get_optimizer(opt_name, param_groups, **opt_factory_config)

def create_scheduler(optimizer, config):
    if "scheduler" not in config:
        return None
    sched_name = config["scheduler"].get("name")
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler"]["T_max"])
    elif sched_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["scheduler"]["milestones"], gamma=config["scheduler"]["gamma"])
    return None

def train(config: dict[str, Any], config_name: str):
    device = torch.device(config["experiment"]["device"])
    torch.manual_seed(config["experiment"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["experiment"]["seed"])

    output_dir = Path("outputs") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = config["experiment"]["tasks"]
    from task import get_task
    tasks = {name: get_task(name, config) for name in task_names}

    # Assume single task for model and criterion creation
    main_task = tasks[task_names[0]]
    model = main_task.get_model().to(device)
    criterion = main_task.get_criterion()

    optimizer_config = config["optimizer"].copy()
    optimizer_name = optimizer_config.pop("name")
    optimizer, optimizer_tags, pi_config = get_optimizer(
        optimizer_name,
        main_task.get_param_groups(model),
        model=model,
        **optimizer_config
    )
    scheduler = create_scheduler(optimizer, config)

    callbacks = [
        ConsoleLogger(config),
        MDLogger(config, output_dir),
        CheckpointSaver(output_dir)
    ]

    train_loaders = {}
    valid_loaders = {}
    for name, task in tasks.items():
        train_loader, valid_loader = task.get_dataloaders()
        train_loaders[name] = train_loader
        valid_loaders[name] = valid_loader

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        config=config
    )

    # Pass optimizer to fit method to allow access to internal diagnostics
    trainer.fit(
        tasks=tasks,
        train_loaders=train_loaders,
        valid_loaders=valid_loaders,
        scheduler=scheduler,
        optimizer_tags=optimizer_tags,
        pi_config=pi_config,
        output_dir=str(output_dir)
    )

def main():
    parser = argparse.ArgumentParser(description="Unified F3EO-Bench Training Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML configuration file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config_name = config_path.stem
    train(config, config_name)

if __name__ == "__main__":
    main()
