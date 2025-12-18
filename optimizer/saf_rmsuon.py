import torch
from torch.optim.optimizer import Optimizer


@torch.jit.script
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16) if G.dtype == torch.float32 else G

    if G.size(-2) > G.size(-1):
        X = X.mT

    # The input G is already scaled, so we just normalize for stability.
    # The NS iteration itself is scale-invariant up to a point.
    X = X.div(X.norm(p=2.0, dim=[-2, -1], keepdim=True).add(1e-7))

    for _ in range(steps):
        A = X.matmul(X.mT)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)


@torch.jit.script
def adamw_step_kernel(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    weight_decay: float,
    eps: float
):
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
    step_size = lr / bias_correction1

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    param.addcdiv_(exp_avg, denom, value=-step_size)


class SAF_RMSuon(Optimizer):
    def __init__(self, params, **kwargs):
        defaults = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'ns_steps': 5,
            'wd_temp': 1.0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_rmsuon_group = group.get('is_rmsuon_group', False)

            if is_rmsuon_group:
                self._saf_rmsuon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _saf_rmsuon_step(self, group: dict):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']
        wd_temp = group.get('wd_temp', 1.0)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            
            grad_norm = grad.norm()
            effective_wd = weight_decay * (1.0 + wd_temp * grad_norm)
            
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            state['step'] += 1
            step = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            # 1. Fisher Pre-whitening (AdaRMSuon logic)
            m_scaled = m_hat / (v_hat.sqrt() + eps)

            # 2. Energy Extraction
            energy = m_scaled.norm()

            # 3. Spectral Filtering
            original_shape = m_scaled.shape
            m_scaled_flat = m_scaled.view(m_scaled.size(0), -1) if p.ndim == 4 else m_scaled

            s_ortho = zeropower_via_newtonschulz5(m_scaled_flat, steps=ns_steps)

            if p.ndim == 4:
                s_ortho = s_ortho.view(original_shape)

            # 4. Energy Injection and Update
            update = energy * s_ortho

            if effective_wd != 0:
                p.mul_(1 - lr * effective_wd)

            p.add_(update, alpha=-lr)

    def _adamw_step(self, group: dict):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1
            step = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            adamw_step_kernel(
                p, grad, exp_avg, exp_avg_sq,
                beta1, beta2, step, lr, weight_decay, eps
            )