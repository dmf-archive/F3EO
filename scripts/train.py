import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import toml
import torch

from utils.data import MetricStore, EpochMetric, StepMetric, TaskMetrics
from utils.observers.console import ConsoleLogger
from utils.observers.markdown import MDLogger
from utils.observers.checkpoint import CheckpointSaver
from utils.metrics import PICalculator, compute_grad_norm

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return toml.load(f)

def get_task_class(task_name: str):
    try:
        module = importlib.import_module(f"task.{task_name}")
        return getattr(module, f"{task_name.capitalize()}Task")
    except (ImportError, AttributeError) as e:
        print(f"Error loading task module '{task_name}': {e}")
        sys.exit(1)

def create_optimizer(model, config):
    from optimizer import get_optimizer
    opt_name = config["optimizer"]["name"]
    opt_config = {k: v for k, v in config["optimizer"].items() if k != "name"}
    if opt_name in ["AdaFisher", "PI_Muon_AdaFisher", "PI_Muon_KFAC"]:
        opt_config["model"] = model
    return get_optimizer(opt_name, model.parameters(), **opt_config)

def create_scheduler(optimizer, config):
    if "scheduler" not in config:
        return None
    sched_name = config["scheduler"].get("name")
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler"]["T_max"])
    elif sched_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["scheduler"]["milestones"], gamma=config["scheduler"]["gamma"])
    return None

def train(config: Dict[str, Any], config_name: str):
    device = torch.device(config["experiment"]["device"])
    torch.manual_seed(config["experiment"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["experiment"]["seed"])

    output_dir = Path("outputs") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = config["experiment"]["tasks"]
    tasks = {name: get_task_class(name)(config) for name in task_names}
    model = tasks[task_names[0]].get_model().to(device)
    criterion = tasks[task_names[0]].get_criterion()
    optimizer, optimizer_tags, pi_config = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    store = MetricStore()
    console_logger = ConsoleLogger(config)
    md_logger = MDLogger(config, output_dir)
    ckpt_saver = CheckpointSaver(output_dir)
    
    pi_gamma = pi_config.get("gamma", 1.0) if pi_config else 1.0
    pi_alpha = pi_config.get("alpha", 1.0) if pi_config else 1.0
    pi_ema_beta = pi_config.get("ema_beta") if pi_config else None
    pi_calculator = PICalculator(gamma=pi_gamma, alpha=pi_alpha, ema_beta=pi_ema_beta)

    console_logger.on_train_begin(str(output_dir))

    start_epoch = 0
    checkpoint = ckpt_saver.load(output_dir / "latest_checkpoint.pt", model, optimizer, scheduler)
    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        store = checkpoint["store"]
        print(f"Resuming training from epoch {start_epoch}")
    
    train_loaders = {name: task.get_dataloaders()[0] for name, task in tasks.items()}
    valid_loaders = {name: task.get_dataloaders()[1] for name, task in tasks.items()}
    
    global_step = 0
    epochs = config["train"]["epochs"]

    for global_epoch in range(start_epoch, epochs):
        model.train()
        
        for task_name in task_names:
            current_task = tasks[task_name]
            current_train_loader = train_loaders[task_name]
            task_epoch = len(store.get_history_for_task(task_name))
            
            console_logger.on_epoch_begin(global_epoch, len(current_train_loader))

            epoch_logits_list = []
            epoch_loss_sum = 0.0
            epoch_grad_norm_list = []

            for step, batch in enumerate(current_train_loader):
                latest_task_epoch_metric = store.get_latest_epoch_for_task(task_name)
                eff_gamma = latest_task_epoch_metric.avg_effective_gamma if latest_task_epoch_metric else None

                logits, loss, _ = current_task.train_step(
                    model=model, batch=batch, criterion=criterion, optimizer=optimizer, device=device,
                    needs_second_order=optimizer_tags.get("requires_second_order", False),
                    accepts_pi_signal=optimizer_tags.get("accepts_pi_signal", False),
                    eff_gamma=eff_gamma
                )
                
                step_metric = StepMetric(
                    task_name=task_name, global_step=global_step, task_epoch=task_epoch,
                    step_in_epoch=step, loss=loss, learning_rate=optimizer.param_groups[0]['lr']
                )
                store.add_step(step_metric)
                console_logger.on_step_end(step_metric, len(current_train_loader))
                
                if logits is not None:
                     epoch_logits_list.append(logits.detach())
                epoch_loss_sum += loss
                epoch_grad_norm_list.append(compute_grad_norm(model))
                global_step += 1

            model.eval()
            with torch.no_grad():
                task_metrics_dict = current_task.validate_epoch(model, valid_loaders[task_name], criterion, device)
            task_metrics = TaskMetrics(metrics=task_metrics_dict)
            model.train()

            avg_train_loss = epoch_loss_sum / len(current_train_loader)
            avg_grad_norm = sum(epoch_grad_norm_list) / len(epoch_grad_norm_list) if epoch_grad_norm_list else None
            
            avg_entropy, avg_pi, avg_eff_gamma = None, None, None
            if pi_calculator and epoch_logits_list:
                with torch.no_grad():
                    all_logits = torch.cat(epoch_logits_list, dim=0)
                    probas = torch.softmax(all_logits, dim=-1)
                    entropy_tensor = -(probas * torch.log_softmax(all_logits, dim=-1)).sum(dim=-1).mean()
                    avg_entropy = entropy_tensor.item()
                    
                    if avg_grad_norm is not None:
                        _, avg_pi = pi_calculator.calculate_pi(torch.tensor(avg_entropy), torch.tensor(avg_grad_norm))
                        if avg_pi is not None:
                            avg_eff_gamma = -torch.log(1.0 - torch.tensor(avg_pi) + pi_calculator.eps).item()

            epoch_metric = EpochMetric(
                task_name=task_name, task_epoch=task_epoch, global_epoch=global_epoch,
                avg_train_loss=avg_train_loss, task_metrics=task_metrics,
                avg_pi=avg_pi, avg_effective_gamma=avg_eff_gamma, avg_entropy=avg_entropy,
                grad_norm=avg_grad_norm, learning_rate=optimizer.param_groups[0]['lr']
            )
            store.add_epoch(epoch_metric)
            # Debug: 验证数据是否正确存储
            print(f"[DEBUG] Added epoch metric for task {task_name}, epoch {global_epoch}")
            print(f"[DEBUG] Store has {len(store.get_history_for_task(task_name))} epochs for this task")
            print(f"[DEBUG] Flat history has {len(store.get_flat_epoch_history())} total epochs")

        console_logger.on_epoch_end(store)
        ckpt_saver.save(global_epoch, model, optimizer, scheduler, store)
        
        if scheduler:
            scheduler.step()

    md_logger.on_train_end(store)
    console_logger.on_train_end(store)

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
