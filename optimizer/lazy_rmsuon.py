import torch
from typing import Optional, List, Tuple


@torch.jit.script
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16) if G.dtype == torch.float32 else G

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X.div(X.norm(p=2.0, dim=[-2, -1], keepdim=True).add(1e-7))

    for _ in range(steps):
        A = X.matmul(X.mT)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


@torch.jit.script
def rmsuon_statistics_kernel(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update statistics
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    m_hat = exp_avg / bias_correction1
    
    # Calculate energy (AdamW update norm)
    # This is the expensive part we want to skip in lazy mode
    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
    adam_update = m_hat.div(denom)
    energy = adam_update.norm()
    
    return m_hat, energy


@torch.jit.script
def rmsuon_statistics_kernel_lazy(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int
) -> torch.Tensor:
    # Only update statistics, do not compute energy
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    bias_correction1 = 1 - beta1 ** step
    m_hat = exp_avg / bias_correction1
    
    return m_hat


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


class LazyRMSuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        energy_sync_every: int = 10,  # New parameter for lazy sync
        aux_lr: Optional[float] = None,
        aux_betas: Optional[tuple] = None,
        aux_eps: Optional[float] = None,
        aux_weight_decay: Optional[float] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        aux_lr = aux_lr if aux_lr is not None else lr
        aux_betas = aux_betas if aux_betas is not None else betas
        aux_eps = aux_eps if aux_eps is not None else eps
        aux_weight_decay = aux_weight_decay if aux_weight_decay is not None else weight_decay

        all_params = []
        for p_or_group in params:
            if isinstance(p_or_group, dict):
                all_params.extend(p_or_group['params'])
            else:
                all_params.append(p_or_group)

        rmsuon_params = []
        adamw_params = []
        for p in all_params:
            if p.requires_grad:
                if p.ndim >= 2 and max(p.shape) < 10000:
                    rmsuon_params.append(p)
                else:
                    adamw_params.append(p)

        param_groups = []
        if rmsuon_params:
            param_groups.append({
                'params': rmsuon_params,
                'is_rmsuon_group': True,
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'ns_steps': ns_steps,
                'energy_sync_every': energy_sync_every,
            })
        if adamw_params:
            param_groups.append({
                'params': adamw_params,
                'is_rmsuon_group': False,
                'lr': aux_lr,
                'betas': aux_betas,
                'eps': aux_eps,
                'weight_decay': aux_weight_decay,
            })

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            is_rmsuon_group = group.get('is_rmsuon_group', False)
            ns_steps = group.get('ns_steps', 5)
            energy_sync_every = group.get('energy_sync_every', 10)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['cached_energy'] = torch.tensor(0.0, device=p.device) # Initialize cached energy

                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                if is_rmsuon_group and p.ndim >= 2:
                    # Decide whether to compute energy or use cache
                    should_sync_energy = (step % energy_sync_every == 0) or (step == 1)

                    if should_sync_energy:
                        m_hat, energy = rmsuon_statistics_kernel(
                            grad, exp_avg, exp_avg_sq, beta1, beta2, step, eps
                        )
                        state['cached_energy'].copy_(energy) # Update cache
                    else:
                        m_hat = rmsuon_statistics_kernel_lazy(
                            grad, exp_avg, exp_avg_sq, beta1, beta2, step
                        )
                        energy = state['cached_energy']

                    original_shape = m_hat.shape
                    if p.ndim == 4:
                        m_hat_flat = m_hat.view(m_hat.size(0), -1)
                    else:
                        m_hat_flat = m_hat

                    O = zeropower_via_newtonschulz5(m_hat_flat, steps=ns_steps)

                    if len(original_shape) == 4:
                        O = O.view(original_shape)

                    base_energy = O.norm().add_(1e-10)
                    scale = energy / base_energy
                    
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    
                    p.add_(O, alpha=-lr * scale)
                    
                else:
                    adamw_step_kernel(
                        p, grad, exp_avg, exp_avg_sq,
                        beta1, beta2, step, lr, weight_decay, eps
                    )

        return loss