import time
from pathlib import Path
from typing import Any

import torch

from utils.data import EpochMetric, MetricStore, StepMetric, TaskMetrics
from utils.metrics import PICalculator, compute_grad_norm

from .callbacks.base import Callback
from .callbacks.context import TrainerContext


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module, device: torch.device,
                 callbacks: list[Callback], config: dict[str, Any]):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks
        self.config = config
        self.store = MetricStore()
        self.context: TrainerContext | None = None

    def _broadcast(self, event: str):
        for cb in self.callbacks:
            getattr(cb, event)(self.context)

    def fit(self, tasks: dict, train_loaders: dict, valid_loaders: dict,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None,
            optimizer_tags: dict, pi_config: dict | None, output_dir: Path):

        self.context = TrainerContext(
            config=self.config,
            output_dir=output_dir,
            device=self.device,
            model=self.model,
            optimizer=self.optimizer,
            store=self.store,
            scheduler=scheduler,
            total_epochs=self.config["train"]["epochs"]
        )

        pi_gamma = pi_config.get("gamma", 1.0) if pi_config else 1.0
        pi_alpha = pi_config.get("alpha", 1.0) if pi_config else 1.0
        pi_ema_beta = pi_config.get("ema_beta") if pi_config else None
        pi_calculator = PICalculator(gamma=pi_gamma, alpha=pi_alpha, ema_beta=pi_ema_beta)

        self._broadcast("on_train_begin")

        start_epoch = 0
        for cb in self.callbacks:
            if cb.load(self.context):
                start_epoch = self.context.current_epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
                break

        self.context.global_step = 0
        task_names = list(tasks.keys())

        for global_epoch in range(start_epoch, self.context.total_epochs):
            self.context.current_epoch = global_epoch
            self.context.is_training = True
            self.model.train()

            for task_name in task_names:
                self.context.current_task_name = task_name
                current_task = tasks[task_name]
                current_train_loader = train_loaders[task_name]
                self.context.total_steps_in_epoch = len(current_train_loader)

                self._broadcast("on_epoch_begin")

                epoch_start_time = time.time()
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                epoch_loss_sum = torch.tensor(0.0, device=self.device)
                epoch_grad_norm_list = []
                epoch_entropy_sum = torch.tensor(0.0, device=self.device)
                num_tokens_in_epoch = 0

                for step, batch in enumerate(current_train_loader):
                    self.context.current_step_in_epoch = step
                    self._broadcast("on_step_begin")

                    logits, loss_tensor, _ = current_task.train_step(
                        model=self.model, batch=batch, criterion=self.criterion, optimizer=self.optimizer,
                        device=self.device,
                        needs_second_order=optimizer_tags.get("requires_second_order", False),
                        optimizer_handles_backward=optimizer_tags.get("requires_loss_for_step", False)
                    )

                    if optimizer_tags.get("requires_loss_for_step", False):
                        self.optimizer.step(loss_tensor)
                    else:
                        self.optimizer.step()

                    step_metric = StepMetric(
                        task_name=task_name, global_step=self.context.global_step,
                        task_epoch=len(self.store.get_history_for_task(task_name)),
                        step_in_epoch=step, loss=loss_tensor.item(), learning_rate=self.optimizer.param_groups[0]['lr']
                    )
                    self.store.add_step(step_metric)
                    self._broadcast("on_step_end")

                    with torch.no_grad():
                        if pi_calculator and logits is not None:
                            probas = torch.softmax(logits, dim=-1)
                            batch_entropy = -(probas * torch.log_softmax(logits, dim=-1)).sum()
                            epoch_entropy_sum += batch_entropy
                            num_tokens_in_epoch += logits.numel()

                        epoch_loss_sum += loss_tensor
                        grad_norm_tensor = compute_grad_norm(self.model, return_tensor=True)
                        if grad_norm_tensor is not None:
                            epoch_grad_norm_list.append(grad_norm_tensor)

                    self.context.global_step += 1

                self.context.is_training = False
                self.model.eval()
                with torch.no_grad():
                    task_metrics_dict = current_task.validate_epoch(self.model, valid_loaders[task_name], self.criterion, self.device)
                task_metrics = TaskMetrics(metrics=task_metrics_dict)

                epoch_time = time.time() - epoch_start_time
                peak_gpu_mem_mb = None
                if self.device.type == 'cuda':
                    peak_gpu_mem_bytes = torch.cuda.max_memory_allocated()
                    peak_gpu_mem_mb = peak_gpu_mem_bytes / (1024 ** 2)

                avg_train_loss = (epoch_loss_sum / len(current_train_loader)).item()

                avg_grad_norm_tensor = None
                if epoch_grad_norm_list:
                    avg_grad_norm_tensor = torch.stack(epoch_grad_norm_list).mean()

                avg_grad_norm = avg_grad_norm_tensor.item() if avg_grad_norm_tensor is not None else None

                avg_entropy, avg_pi_obj = None, None
                if pi_calculator and num_tokens_in_epoch > 0:
                    avg_entropy_tensor = epoch_entropy_sum / num_tokens_in_epoch
                    avg_entropy = avg_entropy_tensor.item()
                    if avg_grad_norm_tensor is not None:
                        _, avg_pi_obj = pi_calculator.calculate_pi(avg_entropy_tensor, avg_grad_norm_tensor)

                diagnostics = getattr(self.optimizer, 'diagnostics', None)

                epoch_metric = EpochMetric(
                    task_name=task_name, task_epoch=len(self.store.get_history_for_task(task_name)),
                    global_epoch=global_epoch,
                    avg_train_loss=avg_train_loss, task_metrics=task_metrics,
                    avg_pi_obj=avg_pi_obj, avg_entropy=avg_entropy,
                    grad_norm=avg_grad_norm, learning_rate=self.optimizer.param_groups[0]['lr'],
                    diagnostics=diagnostics,
                    epoch_time_s=epoch_time, peak_gpu_mem_mb=peak_gpu_mem_mb
                )
                self.store.add_epoch(epoch_metric)
                self._broadcast("on_epoch_end")

            self._broadcast("save")

            if scheduler:
                scheduler.step()

        self._broadcast("on_train_end")
