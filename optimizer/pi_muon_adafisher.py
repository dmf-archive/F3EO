from collections.abc import Callable as CallableABC
from typing import Any

import torch

from .ada_fisher import AdaFisherBackBone
from .muon import muon_update


class PI_Muon_AdaFisher(AdaFisherBackBone):

    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 1e-3,
                 beta: float = 0.9,
                 Lambda: float = 1e-3,
                 gamma: float = 0.8,
                 TCov: int = 100,
                 weight_decay: float = 0,
                 dist: bool = False,
                 muon_momentum: float = 0.95):

        super().__init__(model, lr, beta, Lambda, gamma, TCov, weight_decay, dist)
        self.muon_momentum = muon_momentum

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['muon_momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: None | CallableABC[[], torch.Tensor] = None, effective_gamma: float | None = None, pi_object=None):
        if closure is not None:
            raise NotImplementedError("Closure not supported for PI-Muon-AdaFisher.")

        # 使用 pi_object 或 effective_gamma 计算 pi_value
        pi_value = 0.5  # 默认值
        if pi_object is not None:
            pi_value = pi_object.raw_pi
        elif effective_gamma is not None:
            pi_value = torch.tanh(torch.tensor(effective_gamma)).item()

        lambda_t = 1.0 - pi_value

        for group in self.param_groups:
            idx_param, idx_module = 0, 0
            param_list, hyperparameters = group['params'], {
                "weight_decay": group['weight_decay'],
                "beta": group['beta'],
                "lr": group['lr']
            }

            for _ in range(len(self.modules)):
                if idx_param >= len(param_list) or param_list[idx_param].grad is None:
                    if idx_param < len(param_list) and param_list[idx_param].ndim > 1:
                        idx_module += 1
                    idx_param += 1
                    continue

                p = param_list[idx_param]
                grad = p.grad.data

                state = self.state[p]
                if 'muon_momentum_buffer' not in state:
                    state['muon_momentum_buffer'] = torch.zeros_like(p)

                m = self.modules[idx_module]
                if self._check_dim(param_list, idx_module, idx_param):
                    F_tilde = self._get_F_tilde(m)

                    # 正确：对 Fisher 自然梯度做 Muon 正交化

                    # 理论修正：g_muon 和 g_fisher 必须从同一个原始梯度 g 独立导出
                    g_muon_w = muon_update(
                        grad,
                        state['muon_momentum_buffer'],
                        beta=self.muon_momentum
                    ).reshape(grad.shape)

                    if isinstance(F_tilde, list):
                        F_tilde_w, F_tilde_b = F_tilde

                        g_fisher_w = grad / (F_tilde_w + self.Lambda)
                        g_update_w = (1 - lambda_t) * g_fisher_w + lambda_t * g_muon_w
                        self._apply_adamw_update(group, p, g_update_w)
                        idx_param += 1

                        if idx_param < len(param_list) and param_list[idx_param].grad is not None:
                            p_bias = param_list[idx_param]
                            grad_bias = p_bias.grad.data

                            # Bias terms are not orthogonalized by Muon
                            g_fisher_b = grad_bias / (F_tilde_b + self.Lambda)
                            self._apply_adamw_update(group, p_bias, g_fisher_b)
                            idx_param += 1
                    else:
                        g_fisher = grad / (F_tilde + self.Lambda)
                        g_update = (1 - lambda_t) * g_fisher + lambda_t * g_muon_w
                        self._apply_adamw_update(group, p, g_update)
                        idx_param += 1

                    idx_module += 1
                else:
                    # For params without Fisher info (e.g. embeddings, heads), just use AdamW
                    self._apply_adamw_update(group, p, grad)
                    idx_param += 1

        self.steps += 1

    def _apply_adamw_update(self, group: dict[str, Any], param: torch.nn.Parameter, effective_grad: torch.Tensor):
        state = self.state[param]

        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['beta'], 0.999

        state['step'] += 1

        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        exp_avg.mul_(beta1).add_(effective_grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(1e-8)
        step_size = group['lr'] / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
