import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from utils.data import EpochMetric, MetricStore, StepMetric, TaskMetrics
from utils.metrics import PICalculator, compute_grad_norm

from .callbacks.base import Callback
from .callbacks.context import TrainerContext


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


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

        self.adaptive_wd_config = self.config.get("adaptive_wd", {})
        self.adaptive_wd_enabled = self.adaptive_wd_config.get("enabled", False)
        if self.adaptive_wd_enabled:
            self.wd_mode = self.adaptive_wd_config.get("mode", "ipcwd")
            self.ema_beta = self.adaptive_wd_config.get("ema_beta", 0.98)

            self.ema_ppl = 0.0
            self.wd_gamma = self.adaptive_wd_config.get("gamma", 0.1)
            self.wd_base = self.adaptive_wd_config.get("base_wd", 1e-4)

            self.ema_loss = 0.0
            self.pcwd_alpha = self.adaptive_wd_config.get("alpha", 0.01)
            self.lambda_min = self.adaptive_wd_config.get("lambda_min", 0.01)
            self.lambda_max = self.adaptive_wd_config.get("lambda_max", 1.0)
            self.current_lambda = -1.0  # Uninitialized

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
            total_epochs=self.config["experiment"]["epochs"]
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
                min_train_loss = float('inf')
                min_loss_step = -1
                epoch_grad_norm_list = []
                epoch_entropy_sum = torch.tensor(0.0, device=self.device)
                num_tokens_in_epoch = 0

                for step, batch in enumerate(current_train_loader):
                    self.context.current_step_in_epoch = step
                    self._broadcast("on_step_begin")

                    # Execution Flow Control: Skip initial step if optimizer takes closure
                    skip_initial = optimizer_tags.get("d_1_step_takes_closure", False)
                    
                    print(f"[PROBE] Global Step {self.context.global_step}: skip_initial={skip_initial}, opt={self.optimizer.__class__.__name__}")

                    if not skip_initial:
                        logits, loss_tensor, _ = current_task.train_step(
                            model=self.model, batch=batch, criterion=self.criterion,
                            device=self.device,
                            needs_second_order=optimizer_tags.get("d_2_requires_second_order", False)
                        )
                        if loss_tensor is not None:
                            print(f"[PROBE] Loss obtained: {loss_tensor.item():.4f}")
                    else:
                        logits, loss_tensor = None, None

                    if self.adaptive_wd_enabled and loss_tensor is None and skip_initial:
                        print(f"[WARNING] adaptive_wd is bypassed because loss_tensor is None in closure mode!")

                    if self.adaptive_wd_enabled and loss_tensor is None and skip_initial:
                        # 探针：检查 closure 模式下 adaptive_wd 是否被跳过
                        pass

                    if self.adaptive_wd_enabled and loss_tensor is None and skip_initial:
                        print(f"[DEBUG] adaptive_wd is bypassed for {self.optimizer.__class__.__name__} (closure mode)")

                    if self.adaptive_wd_enabled and loss_tensor is None and skip_initial:
                        print(f"[DEBUG] Step {self.context.global_step}: adaptive_wd bypassed (closure mode)")

                    if self.adaptive_wd_enabled and loss_tensor is not None:
                        loss_item = loss_tensor.detach().item()
                        adaptive_wd = self.optimizer.param_groups[0]['weight_decay']

                        if self.wd_mode == "ipcwd":
                            ppl_batch = torch.exp(torch.tensor(loss_item)).item()
                            if self.ema_ppl == 0.0:
                                self.ema_ppl = ppl_batch
                            else:
                                self.ema_ppl = self.ema_beta * self.ema_ppl + (1 - self.ema_beta) * ppl_batch
                            adaptive_wd = self.wd_base * (1 + self.wd_gamma * self.ema_ppl)

                        elif self.wd_mode == "pcwd":
                            if self.current_lambda < 0:
                                self.current_lambda = self.optimizer.param_groups[0]['weight_decay']

                            if self.ema_loss == 0.0:
                                self.ema_loss = loss_item
                                delta_loss = 0.0
                            else:
                                prev_ema_loss = self.ema_loss
                                self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * loss_item
                                delta_loss = self.ema_loss - prev_ema_loss

                            control_signal = -torch.log(torch.abs(torch.tensor(delta_loss)) + 1e-8).item()

                            if delta_loss < 0:
                                self.current_lambda *= (1 + self.pcwd_alpha * control_signal)
                            else:
                                self.current_lambda /= (1 + self.pcwd_alpha * control_signal)

                            adaptive_wd = torch.clamp(torch.tensor(self.current_lambda), self.lambda_min, self.lambda_max).item()
                            self.current_lambda = adaptive_wd

                        for group in self.optimizer.param_groups:
                            group['weight_decay'] = adaptive_wd

                    self.optimizer.zero_grad()
                    
                    # Define closures for optimizers that need them
                    step_logits, step_loss = None, None

                    def base_closure():
                        nonlocal step_logits, step_loss
                        lgt, ls, _ = current_task.train_step(
                            model=self.model, batch=batch, criterion=self.criterion,
                            device=self.device,
                            needs_second_order=optimizer_tags.get("d_2_requires_second_order", False)
                        )
                        # Capture the first call's results for metrics
                        if step_logits is None:
                            step_logits = lgt
                        if step_loss is None:
                            step_loss = ls
                        return ls

                    if optimizer_tags.get("d_1_step_takes_closure", False):
                        if optimizer_tags.get("d_2_requires_bn_protection", False):
                            call_count = 0
                            def protected_closure():
                                nonlocal call_count
                                if call_count == 0:
                                    enable_running_stats(self.model)
                                else:
                                    disable_running_stats(self.model)
                                res = base_closure()
                                call_count += 1
                                return res
                            
                            step_output = self.optimizer.step(protected_closure)
                            # Ensure BN stats are enabled back
                            enable_running_stats(self.model)
                        else:
                            step_output = self.optimizer.step(base_closure)
                        
                        # Sync outer scope variables for metrics
                        logits, loss_tensor = step_logits, step_loss
                        if loss_tensor is None and isinstance(step_output, torch.Tensor):
                            loss_tensor = step_output
                    else:
                        # For standard optimizers, and those requiring just the loss tensor.
                        if not optimizer_tags.get("d_1_step_requires_loss_tensor", False):
                            print(f"[DEBUG] Manual backward for {self.optimizer.__class__.__name__}")
                            loss_tensor.backward(create_graph=optimizer_tags.get("d_2_requires_second_order", False))
                        
                        if optimizer_tags.get("d_1_step_requires_loss_tensor", False):
                             self.optimizer.step(loss_tensor)
                        else:
                             self.optimizer.step()

                    # 验证梯度是否存在
                    has_grad = any(p.grad is not None for group in self.optimizer.param_groups for p in group['params'])
                    if not has_grad:
                        print(f"[WARNING] No gradient detected after backward for task {task_name}!")

                    step_metric = StepMetric(
                        task_name=task_name, global_step=self.context.global_step,
                        task_epoch=len(self.store.get_history_for_task(task_name)),
                        step_in_epoch=step,
                        loss=loss_tensor.item() if loss_tensor is not None else 0.0,
                        learning_rate=self.optimizer.param_groups[0]['lr']
                    )
                    self.store.add_step(step_metric)
                    self._broadcast("on_step_end")

                    with torch.no_grad():
                        if pi_calculator and logits is not None:
                            probas = torch.softmax(logits, dim=-1)
                            batch_entropy = -(probas * torch.log_softmax(logits, dim=-1)).sum()
                            epoch_entropy_sum += batch_entropy
                            num_tokens_in_epoch += logits.numel()

                        if loss_tensor is not None:
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

                # Early stopping for grokking
                if task_metrics_dict.get("accuracy", 0.0) >= 99.9 or task_metrics_dict.get("loss", 1.0) < 0.001:
                    print(f"Early stopping triggered: {task_name} accuracy reached {task_metrics_dict.get('accuracy', 0.0)}% or loss reached {task_metrics_dict.get('loss', 1.0)}")
                    epoch_time = time.time() - epoch_start_time
                    avg_train_loss = (epoch_loss_sum / len(current_train_loader)).item()
                    epoch_metric = EpochMetric(
                        task_name=task_name, task_epoch=len(self.store.get_history_for_task(task_name)),
                        global_epoch=global_epoch,
                        avg_train_loss=avg_train_loss,
                        min_train_loss=min_train_loss if min_train_loss != float('inf') else None,
                        min_loss_step=min_loss_step if min_loss_step != -1 else None,
                        task_metrics=task_metrics,
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        epoch_time_s=epoch_time
                    )
                    self.store.add_epoch(epoch_metric)
                    self._broadcast("on_epoch_end")
                    self._broadcast("save")
                    self._broadcast("on_train_end")
                    return

                epoch_time = time.time() - epoch_start_time
                peak_gpu_mem_mb = None
                if self.device.type == 'cuda':
                    peak_gpu_mem_bytes = torch.cuda.max_memory_allocated()
                    peak_gpu_mem_mb = peak_gpu_mem_bytes / (1024 ** 2)

                avg_train_loss = (epoch_loss_sum / len(current_train_loader)).item()

                avg_grad_norm_tensor = None
                if epoch_grad_norm_list:
                    tensor_list = [t for t in epoch_grad_norm_list if isinstance(t, torch.Tensor)]
                    if tensor_list:
                        avg_grad_norm_tensor = torch.stack(tensor_list).mean()

                avg_grad_norm = avg_grad_norm_tensor.item() if avg_grad_norm_tensor is not None else None

                avg_entropy, avg_pi_obj = None, None
                if pi_calculator and num_tokens_in_epoch > 0:
                    avg_entropy_tensor = epoch_entropy_sum / num_tokens_in_epoch
                    avg_entropy = avg_entropy_tensor.item()
                    if avg_grad_norm is not None:
                        _, avg_pi_obj = pi_calculator.calculate_pi(avg_entropy_tensor, avg_grad_norm)

                diagnostics = getattr(self.optimizer, 'diagnostics', {})
                if diagnostics is not None:
                    import copy
                    diagnostics = copy.deepcopy(diagnostics)
                
                # 监控各参数组的平均范数
                for i, group in enumerate(self.optimizer.param_groups):
                    group_name = "muon" if group.get("use_muon") or group.get("is_rmsuon_group") else "adam"
                    norms = [p.norm().item() for p in group['params']]
                    if norms:
                        diagnostics[f"group_{i}_{group_name}_avg_norm"] = sum(norms) / len(norms)

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
