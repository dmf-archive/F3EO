import torch
from torch.optim.optimizer import Optimizer


class F3EWD(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 maximize=False,
                 single_gpu=True,
                 orthogonalize=True,
                 meta_grad_clip_norm=1.0,
                 gamma=0.1):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize,
                        orthogonalize=orthogonalize, meta_grad_clip_norm=meta_grad_clip_norm,
                        gamma=gamma)
        
        self.single_gpu = single_gpu
        super(F3EWD, self).__init__(params, defaults)
        self.last_log_pi = 0.0
        self.last_adaptive_wd = 0.0
        self.pi_step = 0
        self.exp_avg_log_pi = 0.0

    def step(self, closure=None, loss=None, logits=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('F3EO-PIWD does not support sparse gradients.')
                    if p.grad.grad_fn is None and p.grad.requires_grad == False:
                        raise RuntimeError('Gradient tensor does not have grad_fn. When calling loss.backward(), make sure the option create_graph is set to True.')
                    params_with_grad.append(p)
                    grads.append(p.grad)

        if not grads:
            return loss, None

        # Compute grad norm once to avoid O(N) redundancy
        grad_norm_sq = sum(g.pow(2).sum() for g in grads)
        
        adaptive_weight_decay = self.param_groups[0]['weight_decay']
        entropy = None
        beta_multiplier = 1.0
        awd_multiplier = 1.0
        if logits is not None:
            with torch.no_grad():
                probas = torch.softmax(logits, dim=-1)
                log_probas = torch.log_softmax(logits, dim=-1)
                entropy_per_sample = -torch.sum(probas * log_probas, dim=-1)
                entropy = entropy_per_sample.mean()
                grad_norm_for_pi = torch.sqrt(grad_norm_sq)

            # alpha is now hardcoded to 1, gamma is the sole hyperparameter
            gamma = self.param_groups[0]['gamma']
            log_pi = -entropy + gamma * grad_norm_for_pi

            self.pi_step += 1
            beta1, _ = self.param_groups[0]['betas']
            self.exp_avg_log_pi = self.exp_avg_log_pi * beta1 + log_pi.item() * (1 - beta1)
            bias_correction_pi = 1 - beta1 ** self.pi_step
            smoothed_log_pi = self.exp_avg_log_pi / bias_correction_pi
            
            # pi_norm is exp(log(PI)), its range is (0, 1]
            # Clamping log_pi to prevent exp from creating large values if log_pi > 0
            pi_norm = torch.exp(torch.clamp(torch.tensor(smoothed_log_pi), max=0.0))

            # AWD multiplier: Monotonic exponential, driven by gamma
            awd_multiplier = torch.exp(gamma * pi_norm)

            # Beta multiplier: U-shaped cosh, centered at 0.5, driven by gamma
            beta_multiplier = torch.cosh(gamma * (pi_norm - 0.5))

            # Note: weight_decay from config is only a base value.
            # The effective weight decay is fully determined by the PI signal.
            base_wd = self.param_groups[0]['weight_decay']
            adaptive_weight_decay = base_wd * awd_multiplier
            
            self.last_log_pi = smoothed_log_pi
            self.last_adaptive_wd = adaptive_weight_decay.item() if isinstance(adaptive_weight_decay, torch.Tensor) else adaptive_weight_decay
        else:
            self.last_log_pi = 0.0
            self.last_adaptive_wd = adaptive_weight_decay

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
            param_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    meta_grad = meta_grads[param_idx]
                    first_grad = grads[param_idx]
                    param_idx += 1

                    if meta_grad is None:
                        effective_grad = first_grad
                    else:
                        if group['orthogonalize'] and first_grad is not None:
                            first_grad_flat = first_grad.reshape(-1)
                            meta_grad_flat = meta_grad.reshape(-1)
                            first_grad_dot = torch.dot(first_grad_flat, first_grad_flat)
                            if first_grad_dot > 0:
                                projection_scale = torch.dot(meta_grad_flat, first_grad_flat) / first_grad_dot
                                meta_grad = meta_grad - projection_scale * first_grad
                        
                        effective_grad = first_grad - beta_multiplier * meta_grad

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']

                    state['step'] += 1
                    beta1, beta2 = group['betas']

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    if adaptive_weight_decay > 0:
                        p.add_(p, alpha=-adaptive_weight_decay * group['lr'])

                    exp_avg.mul_(beta1).add_(effective_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1 - beta2)

                    if group['amsgrad']:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss, entropy
