import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import toml
import torch

from utils.callbacks.checkpoint import CheckpointSaver
from utils.callbacks.console import ConsoleLogger
from utils.callbacks.markdown import MDLogger
from utils.trainer import Trainer


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return toml.load(f)

def get_task_class(task_name: str):
    try:
        module = importlib.import_module(f"task.{task_name}")
        # e.g., Cifar10Task, Wikitext2Task
        task_class_name = "".join(part.capitalize() for part in task_name.split('_')) + "Task"
        return getattr(module, task_class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading task module '{task_name}': {e}")
        sys.exit(1)

def create_optimizer(model, task, config):
    from optimizer import get_optimizer
    optimizer_config = config["optimizer"]
    opt_name = optimizer_config["name"]
    opt_factory_config = optimizer_config.copy()
    opt_factory_config.pop("name")

    param_groups = task.get_param_groups(model)

    if opt_name in ["AdaFisher", "KFAC", "FOG", "DiagKFAC", "DiagFOG"]:
        opt_factory_config["model"] = model

    if opt_name == "Muon":
        # Use MuonWithAuxAdam for proper parameter grouping
        from optimizer.muon import SingleDeviceMuonWithAuxAdam
        muon_groups = []
        for group in param_groups:
            if group.get('use_diag_fog', True):  # hidden weights -> use Muon
                muon_group = {
                    'params': group['params'],
                    'use_muon': True,
                    'lr': opt_factory_config.get('lr', 0.02),
                    'momentum': opt_factory_config.get('momentum', 0.95),
                    'weight_decay': opt_factory_config.get('weight_decay', 0.1)
                }
            else:  # non-hidden weights -> use AdamW
                muon_group = {
                    'params': group['params'],
                    'use_muon': False,
                    'lr': 1e-4,  # AdamW learning rate matching DiagFOG
                    'betas': (0.9, 0.95),
                    'eps': 1e-10,
                    'weight_decay': opt_factory_config.get('weight_decay', 0.1)
                }
            muon_groups.append(muon_group)
        optimizer = SingleDeviceMuonWithAuxAdam(muon_groups)
        return optimizer, {}, None
    elif len(param_groups) > 1:
        if opt_name == "DiagFOG":
            adam_lr = opt_factory_config.pop("adam_lr", 1e-4)
            adam_wd = opt_factory_config.pop("adam_weight_decay", 0.01)
            adam_betas = opt_factory_config.pop("adam_betas", (0.9, 0.95))
            for group in param_groups:
                if not group.get('use_diag_fog', False):
                    group.setdefault('lr', adam_lr)
                    group.setdefault('weight_decay', adam_wd)
                    group.setdefault('betas', adam_betas)
        elif opt_name in ["DiagKFAC", "FOG"]:
            # Pop AdamW-specific args from the factory config, as FOG/KFAC don't accept them
            adam_lr = opt_factory_config.pop("adam_lr", 1e-4)
            adam_wd = opt_factory_config.pop("adam_weight_decay", 0.01)
            adam_betas = opt_factory_config.pop("adam_betas", (0.9, 0.95))

            for group in param_groups:
                # use_diag_fog is the generic flag for "use main optimizer"
                if not group.get('use_diag_fog', False):
                    group.setdefault('lr', adam_lr)
                    group.setdefault('weight_decay', adam_wd)
                    group.setdefault('betas', adam_betas)

        return get_optimizer(opt_name, param_groups, **opt_factory_config)
    else:
        params = next(iter(param_groups))['params']
        return get_optimizer(opt_name, params, **opt_factory_config)

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
    tasks = {name: get_task_class(name)(config) for name in task_names}

    # Assume single task for model and criterion creation
    main_task = tasks[task_names[0]]
    model = main_task.get_model().to(device)
    criterion = main_task.get_criterion()

    optimizer, optimizer_tags, pi_config = create_optimizer(model, main_task, config)
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
