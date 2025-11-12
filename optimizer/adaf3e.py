import torch
from torch.optim.optimizer import Optimizer


class AdaF3E(Optimizer):
    """
    Implements the AdaF3E algorithm (Scalar-Modulated version).
    This optimizer uses a scalar projection of the third-order information (g^T H g)
    to adaptively modulate the learning rate.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999, 0.9), # (beta1, beta2, beta_s for scalar modulation)
                 eps=1e-8,
                 weight_decay=0,
                 alpha=1.0, # Strength of the scalar modulation
                 meta_grad_clip_norm=1.0,
                 amsgrad=False):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        alpha=alpha, meta_grad_clip_norm=meta_grad_clip_norm)

        super(AdaF3E, self).__init__(params, defaults)
        # Initialize state for the scalar s
        self.state['s_ema'] = 0.0

    def step(self, closure=None, effective_gamma=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('AdaF3E does not support sparse gradients.')
                    if not p.grad.requires_grad:
                        raise RuntimeError('Gradient tensor does not have grad_fn. When calling loss.backward(), make sure the option create_graph is set to True.')
                    params_with_grad.append(p)
                    grads.append(p.grad)

        if not grads:
            return loss

        grad_norm_sq = sum(g.pow(2).sum() for g in grads)
        meta_grads = torch.autograd.grad(grad_norm_sq, params_with_grad, retain_graph=False, allow_unused=True)

        clip_value = self.param_groups[0]['meta_grad_clip_norm']
        if clip_value > 0:
            total_norm = torch.sqrt(sum(torch.norm(g.detach(), 2).pow(2) for g in meta_grads if g is not None))
            clip_coef = clip_value / (total_norm + 1e-6)
            if clip_coef < 1:
                clipped_meta_grads = []
                for g in meta_grads:
                    if g is not None:
                        clipped_meta_grads.append(g.mul(clip_coef))
                    else:
                        clipped_meta_grads.append(None)
                meta_grads = tuple(clipped_meta_grads)

        with torch.no_grad():
            for i, p in enumerate(params_with_grad):
                grad = grads[i]
                meta_grad = meta_grads[i]
                group = self.param_groups[0] # Assuming one group

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_curvature'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, exp_curvature = state['exp_avg'], state['exp_avg_sq'], state['exp_curvature']

                state['step'] += 1
                beta1, beta2, beta3 = group['betas']

                # PIWD: 使用 effective_gamma 动态调整权重衰减
                adaptive_weight_decay = group['weight_decay']
                if effective_gamma is not None and effective_gamma > 0:
                    multiplier = torch.exp(torch.tensor(effective_gamma)).item()
                    adaptive_weight_decay *= multiplier

                if adaptive_weight_decay > 0:
                    p.add_(p, alpha=-adaptive_weight_decay * group['lr'])

                # 注意：这里的 grad 不再包含 weight_decay，因为它已被手动应用
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # --- Scalar Modulation Logic ---
                s = 0.0
                if meta_grad is not None:
                    s = torch.dot(grad.view(-1), meta_grad.view(-1)).item()

                beta_s = group['betas'][2]
                self.state['s_ema'] = beta_s * self.state['s_ema'] + (1 - beta_s) * s

                # Modulation function: f(x) = exp(-alpha * x)
                modulation_factor = torch.exp(torch.tensor(-group['alpha'] * self.state['s_ema'])).item()

                # Modulate learning rate
                modulated_lr = group['lr'] * modulation_factor
                # ---

                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(exp_avg_sq, max_exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['eps'])

                step_size = modulated_lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
