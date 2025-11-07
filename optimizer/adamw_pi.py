import torch
from torch.optim.optimizer import Optimizer


class AdamW_PI(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, amsgrad=False, maximize=False, gamma=0.1):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, gamma=gamma))
        self.pi_step = 0
        self.exp_avg_log_pi = 0.0
        self.last_log_pi = 0.0
        self.last_entropy = 0.0
        self.last_adaptive_wd = 0.0

    def step(self, closure=None, loss=None, logits=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if logits is not None:
            with torch.no_grad():
                probas = torch.softmax(logits, dim=-1)
                log_probas = torch.log_softmax(logits, dim=-1)
                entropy = -(probas * log_probas).sum(dim=-1).mean()
                grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in self.params if p.grad is not None))
                log_pi = -entropy + self.param_groups[0]['gamma'] * grad_norm
                beta1 = self.param_groups[0]['betas'][0]
                self.pi_step += 1
                self.exp_avg_log_pi = self.exp_avg_log_pi * beta1 + log_pi.item() * (1 - beta1)
                pi_norm = torch.exp(torch.clamp(torch.tensor(self.exp_avg_log_pi / (1 - beta1 ** self.pi_step)), max=0.0))
                adaptive_wd = self.param_groups[0]['weight_decay'] * torch.exp(self.param_groups[0]['gamma'] * pi_norm)
                for group in self.param_groups:
                    group['weight_decay'] = adaptive_wd.item()
                self.last_log_pi = self.exp_avg_log_pi / (1 - beta1 ** self.pi_step)
                self.last_entropy = entropy.item()
                self.last_adaptive_wd = adaptive_wd.item()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p)
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if group['amsgrad']:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / (1 - beta2 ** state['step']) ** 0.5).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / (1 - beta2 ** state['step']) ** 0.5).add_(group['eps'])
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group['weight_decay'] > 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])

        return loss, None