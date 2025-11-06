import torch
from torch.optim.optimizer import Optimizer


class F3EPIRMS(Optimizer):
    """F3EPI-RMS：撤掉 tanh 与一阶动量，仅保留二阶矩步长自适应"""

    def __init__(
        self,
        params,
        lr=1e-4,
        eps=1e-8,
        weight_decay=0,
        maximize=False,
        orthogonalize=True,
        meta_grad_clip_norm=1.0,
        alpha=1.0,
        gamma=0.1,
        beta2=0.999,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            orthogonalize=orthogonalize,
            meta_grad_clip_norm=meta_grad_clip_norm,
            alpha=alpha,
            gamma=gamma,
            beta2=beta2,
        )
        super(F3EPIRMS, self).__init__(params, defaults)
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
                        raise RuntimeError("F3EPIRMS does not support sparse gradients.")
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
            # 关键：撤掉 tanh，用硬 clip 给 PI 全杠杆
            beta_complexity = torch.clamp(log_pi, -1.0, 1.0)
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

        # 参数更新
        with torch.no_grad():
            param_idx = 0
            for group in self.param_groups:
                beta2 = group["beta2"]
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

                    # 有效梯度：g + β δ_meta
                    effective_grad = (
                        grad + beta_complexity * meta_grad
                        if meta_grad is not None
                        else grad
                    )

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["square_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    square_avg = state["square_avg"]
                    state["step"] += 1

                    # RMS 式二阶矩更新
                    square_avg.mul_(beta2).addcmul_(
                        effective_grad, effective_grad, value=1 - beta2
                    )

                    # 权重衰减
                    if group["weight_decay"] != 0:
                        p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                    # RMSProp 步长
                    avg = square_avg.sqrt().add_(group["eps"])
                    p.addcdiv_(effective_grad, avg, value=-group["lr"])

        return main_loss
