from dataclasses import dataclass

import torch


@dataclass
class PIObject:
    raw_pi: float
    log_pi: float
    neg_exp_pi: float

class PICalculator:
    def __init__(self, gamma: float, alpha: float, ema_beta: float | None = None, eps: float = 1e-8):
        self.gamma = gamma
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.eps = eps
        self.exp_avg_pi = 0.0
        self.pi_step = 0

    def calculate_pi(self, entropy: torch.Tensor, grad_norm: float) -> tuple[PIObject, PIObject]:
        instant_pi_val = torch.exp(-(self.alpha * entropy + self.gamma * grad_norm)).item()

        def _create_pi_object(pi_value: float) -> PIObject:
            return PIObject(
                raw_pi=pi_value,
                log_pi=-torch.log(1.0 - torch.tensor(pi_value) + self.eps).item(),
                neg_exp_pi=-torch.exp(torch.tensor(pi_value)).item()
            )

        instant_pi_obj = _create_pi_object(instant_pi_val)

        if self.ema_beta is not None:
            self.pi_step += 1
            self.exp_avg_pi = self.exp_avg_pi * self.ema_beta + instant_pi_val * (1 - self.ema_beta)
            bias_correction = 1 - self.ema_beta ** self.pi_step
            smoothed_pi_val = self.exp_avg_pi / bias_correction
            smoothed_pi_obj = _create_pi_object(smoothed_pi_val)
            return instant_pi_obj, smoothed_pi_obj

        return instant_pi_obj, instant_pi_obj

def compute_grad_norm(model: torch.nn.Module, return_tensor: bool = False) -> float | torch.Tensor | None:
    grad_norms = [
        p.grad.detach().data.norm(2)
        for p in model.parameters()
        if p.grad is not None
    ]
    if not grad_norms:
        return None

    total_norm_sq = torch.stack(grad_norms).pow(2).sum()
    total_norm = total_norm_sq.sqrt()

    if return_tensor:
        return total_norm
    return total_norm.item()
