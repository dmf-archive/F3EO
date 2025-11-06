import torch
from torch.optim.optimizer import Optimizer


class F3EPIAdaX(Optimizer):
    """F3EPI-AdaX：解耦自适应的三阶优化器"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
        orthogonalize=True,
        meta_grad_clip_norm=1.0,
        alpha=1.0,
        gamma=0.1,
    ):
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            orthogonalize=orthogonalize,
            meta_grad_clip_norm=meta_grad_clip_norm,
            alpha=alpha,
            gamma=gamma,
        )
        super(F3EPIAdaX, self).__init__(params, defaults)
        self.last_log_pi = 0.0

    def step(self, closure=None, loss=None):
        eval_loss = None
        if closure is not None:
            with torch.enable_grad():
                eval_loss = closure()

        main_loss = loss if loss is not None else eval_loss

        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("F3EPIAdaX does not support sparse gradients.")
                    if p.grad.grad_fn is None and not p.grad.requires_grad:
                        raise RuntimeError(
                            "Gradient tensor does not have grad_fn. "
                            "When calling loss.backward(), make sure create_graph=True."
                        )
                    params_with_grad.append(p)
                    grads.append(p.grad)

        if not grads:
            return main_loss

        # 计算元梯度相关量
        grad_norm_sq_for_meta = sum(g.pow(2).sum() for g in grads)
        beta_complexity = 0.0
        if main_loss is not None:
            grad_norm_for_pi = torch.sqrt(
                sum(g.detach().pow(2).sum() for g in grads)
            )
            alpha = self.param_groups[0]["alpha"]
            gamma = self.param_groups[0]["gamma"]
            log_pi = -alpha * main_loss.detach() + alpha * gamma * grad_norm_for_pi
            beta_complexity = torch.tanh(log_pi)
            self.last_log_pi = log_pi.item()
        else:
            beta_complexity = 0.0
            self.last_log_pi = 0.0

        # 计算元梯度并裁剪
        meta_grads = torch.autograd.grad(
            grad_norm_sq_for_meta,
            params_with_grad,
            retain_graph=False,
            allow_unused=True,
        )

        clip_value = self.param_groups[0]["meta_grad_clip_norm"]
        if clip_value > 0:
            device = params_with_grad[0].device
            total_norm = torch.sqrt(
                sum(torch.norm(g.detach(), 2).pow(2) for g in meta_grads if g is not None)
            )
            clip_coef = clip_value / (total_norm + 1e-6)
            if clip_coef < 1:
                meta_grads = tuple(
                    g.mul_(clip_coef) if g is not None else None for g in meta_grads
                )

        with torch.no_grad():
            param_idx = 0
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = grads[param_idx]
                    meta_grad = meta_grads[param_idx]
                    param_idx += 1

                    # 正交化
                    if meta_grad is not None and group["orthogonalize"]:
                        grad_flat = grad.reshape(-1)
                        meta_flat = meta_grad.reshape(-1)
                        grad_dot = torch.dot(grad_flat, grad_flat)
                        if grad_dot > 0:
                            proj = torch.dot(meta_flat, grad_flat) / grad_dot
                            meta_grad = meta_grad - proj * grad

                    # 构造有效梯度
                    if meta_grad is not None:
                        effective_grad = grad + beta_complexity * meta_grad
                    else:
                        effective_grad = grad

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["meta_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    meta_exp_avg_sq = state["meta_exp_avg_sq"]
                    if group["amsgrad"]:
                        max_exp_avg_sq = state["max_exp_avg_sq"]

                    state["step"] += 1

                    # 权重衰减
                    if group["weight_decay"] != 0:
                        p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                    # 一阶梯度用完整 Adam
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # 三阶梯度用 RMSProp 式二阶矩
                    if meta_grad is not None:
                        meta_exp_avg_sq.mul_(beta2).addcmul_(meta_grad, meta_grad, value=1 - beta2)

                    # 分别计算并应用更新
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    # 一阶梯度 Adam 更新
                    denom_g = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group["eps"])
                    step_size_g = group["lr"] / bias_correction1
                    p.addcdiv_(exp_avg, denom_g, value=-step_size_g)

                    # 三阶梯度 RMSProp 更新
                    if meta_grad is not None:
                        denom_meta = (meta_exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group["eps"])
                        step_size_meta = group["lr"] * beta_complexity
                        p.addcdiv_(meta_grad, denom_meta, value=-step_size_meta)

        return main_loss