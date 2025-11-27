import time
from typing import Any

import torch

from utils.data import EpochMetric, MetricStore, StepMetric, TaskMetrics
from utils.metrics import PICalculator, compute_grad_norm

from .callbacks.base import Callback


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

    def _broadcast(self, event: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, event)(**kwargs)

    def fit(self, tasks: dict, train_loaders: dict, valid_loaders: dict,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None,
            optimizer_tags: dict, pi_config: dict | None, output_dir: str):

        pi_gamma = pi_config.get("gamma", 1.0) if pi_config else 1.0
        pi_alpha = pi_config.get("alpha", 1.0) if pi_config else 1.0
        pi_ema_beta = pi_config.get("ema_beta") if pi_config else None
        pi_calculator = PICalculator(gamma=pi_gamma, alpha=pi_alpha, ema_beta=pi_ema_beta)

        self._broadcast("on_train_begin", store=self.store, output_dir=output_dir)

        start_epoch = 0
        for cb in self.callbacks:
            checkpoint = cb.load(path=f"{output_dir}/latest_checkpoint.pt", model=self.model,
                                 optimizer=self.optimizer, scheduler=scheduler)
            if checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                self.store = checkpoint["store"]
                print(f"Resuming training from epoch {start_epoch}")
                break

        global_step = 0
        epochs = self.config["train"]["epochs"]
        task_names = list(tasks.keys())

        for global_epoch in range(start_epoch, epochs):
            self.model.train()

            for task_name in task_names:
                current_task = tasks[task_name]
                current_train_loader = train_loaders[task_name]
                task_epoch = len(self.store.get_history_for_task(task_name))

                self._broadcast("on_epoch_begin", epoch=global_epoch, total_steps=len(current_train_loader))

                epoch_start_time = time.time()
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                epoch_loss_sum = 0.0
                epoch_grad_norm_list = []
                epoch_entropy_sum = 0.0
                num_tokens_in_epoch = 0

                for step, batch in enumerate(current_train_loader):
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
                        task_name=task_name, global_step=global_step, task_epoch=task_epoch,
                        step_in_epoch=step, loss=loss_tensor.item(), learning_rate=self.optimizer.param_groups[0]['lr']
                    )
                    self.store.add_step(step_metric)
                    self._broadcast("on_step_end", step_metric=step_metric, total_steps=len(current_train_loader))

                    if pi_calculator and logits is not None:
                        with torch.no_grad():
                            probas = torch.softmax(logits, dim=-1)
                            batch_entropy_sum = -(probas * torch.log_softmax(logits, dim=-1)).sum()
                            epoch_entropy_sum += batch_entropy_sum.item()
                            num_tokens_in_epoch += logits.shape[0] * logits.shape[1]

                    epoch_loss_sum += loss_tensor.item()
                    epoch_grad_norm_list.append(compute_grad_norm(self.model))
                    global_step += 1

                self.model.eval()
                with torch.no_grad():
                    task_metrics_dict = current_task.validate_epoch(self.model, valid_loaders[task_name], self.criterion, self.device)
                task_metrics = TaskMetrics(metrics=task_metrics_dict)
                self.model.train()

                epoch_time = time.time() - epoch_start_time
                peak_gpu_mem_mb = None
                if self.device.type == 'cuda':
                    peak_gpu_mem_bytes = torch.cuda.max_memory_allocated()
                    peak_gpu_mem_mb = peak_gpu_mem_bytes / (1024 ** 2)

                avg_train_loss = epoch_loss_sum / len(current_train_loader)
                avg_grad_norm = sum(epoch_grad_norm_list) / len(epoch_grad_norm_list) if epoch_grad_norm_list else None

                avg_entropy, avg_pi_obj = None, None
                if pi_calculator and num_tokens_in_epoch > 0:
                    avg_entropy = epoch_entropy_sum / num_tokens_in_epoch
                    if avg_grad_norm is not None:
                        _, avg_pi_obj = pi_calculator.calculate_pi(torch.tensor(avg_entropy), avg_grad_norm)

                diagnostics = getattr(self.optimizer, 'diagnostics', None)

                epoch_metric = EpochMetric(
                    task_name=task_name, task_epoch=task_epoch, global_epoch=global_epoch,
                    avg_train_loss=avg_train_loss, task_metrics=task_metrics,
                    avg_pi_obj=avg_pi_obj, avg_entropy=avg_entropy,
                    grad_norm=avg_grad_norm, learning_rate=self.optimizer.param_groups[0]['lr'],
                    diagnostics=diagnostics,
                    epoch_time_s=epoch_time, peak_gpu_mem_mb=peak_gpu_mem_mb
                )
                self.store.add_epoch(epoch_metric)
                self._broadcast("on_epoch_end", store=self.store)

            self._broadcast("save", epoch=global_epoch, model=self.model, optimizer=self.optimizer,
                          scheduler=scheduler, store=self.store)

            if scheduler:
                scheduler.step()

        self._broadcast("on_train_end", store=self.store)
