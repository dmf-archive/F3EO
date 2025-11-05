import torch
from torch.optim.optimizer import Optimizer

class F3EO(Optimizer):
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
                 meta_grad_clip_norm=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize,
                        orthogonalize=orthogonalize, meta_grad_clip_norm=meta_grad_clip_norm)
        
        self.single_gpu = single_gpu
        super(F3EO, self).__init__(params, defaults)

    def step(self, closure=None):
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
                        raise RuntimeError('F3EO does not support sparse gradients.')
                    # 检查 p.grad 是否有 grad_fn，确保可以进行二阶反向传播
                    if p.grad.grad_fn is None and p.grad.requires_grad == False: # 增加 requires_grad 检查
                        raise RuntimeError('Gradient tensor does not have grad_fn. When calling loss.backward(), make sure the option create_graph is set to True.')
                    params_with_grad.append(p)
                    grads.append(p.grad)
        
        if not grads:
            return loss

        # 计算梯度的L2范数的平方
        grad_norm_sq = torch.sum(torch.stack([g.pow(2).sum() for g in grads]))
        
        # 计算 meta_grads (即 ∇θ ||g||²)
        meta_grads = torch.autograd.grad(grad_norm_sq, params_with_grad, retain_graph=False, allow_unused=True)

        # 对元梯度进行范数裁剪
        clip_value = self.param_groups[0]['meta_grad_clip_norm']
        if clip_value > 0:
            device = params_with_grad[0].device
            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in meta_grads if g is not None]), 2)
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

                    if meta_grad is None: # 如果 meta_grad 为 None，说明该参数没有二阶梯度，直接使用一阶梯度
                        effective_grad = first_grad
                    else:
                        # 正交化处理：确保 meta_grad 与 first_grad 正交
                        if group['orthogonalize'] and first_grad is not None:
                            first_grad_flat = first_grad.reshape(-1)
                            meta_grad_flat = meta_grad.reshape(-1)
                            first_grad_dot = torch.dot(first_grad_flat, first_grad_flat)
                            if first_grad_dot > 0:
                                projection_scale = torch.dot(meta_grad_flat, first_grad_flat) / first_grad_dot
                                meta_grad = meta_grad - projection_scale * first_grad
                        
                        # 最终有效梯度为一阶梯度加上正交化后的三阶修正
                        effective_grad = first_grad + meta_grad

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
                    
                    if group['weight_decay'] != 0:
                        p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                    exp_avg.mul_(beta1).add_(effective_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(effective_grad, effective_grad, value=1 - beta2)

                    if group['amsgrad']:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss