import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional

from .kfac import KFACOptimizer
from .muon import muon_update


class FIENA_FOG(KFACOptimizer):
    """
    FIENA-Full：在 Full KFAC + Muon 之上，注入 PI 门控的黎曼平滑三阶梯度
    g_final = g + λ · r_t，其中 r_t 是 PI 加权的 H·g 自然指数平均
    """

    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 muon_momentum=0.95,
                 lambda_=1e-4,      # 三阶正则强度
                 pi_beta=1.0,       # PI→权重斜率
                 hutch_samples=1):  # Hutchinson 样本数
        super().__init__(model, lr, momentum, stat_decay, damping, kl_clip,
                         weight_decay, TCov, TInv, batch_averaged)

        self.lambda_ = lambda_
        self.pi_beta = pi_beta
        self.hutch_samples = hutch_samples

        # 为每个参数维护三阶平滑缓冲区
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['fiena_buffer'] = torch.zeros_like(p)

    # ------------------------------------------------------------------
    # 1) 外部调用：传入当前批次的平均 PI
    # ------------------------------------------------------------------
    def set_pi(self, pi: float):
        """传入当前批次的平均 PI（0→1]）"""
        self.pi = pi

    # ------------------------------------------------------------------
    # 2) 计算 PI 门控权重 w_t = σ(β · PI)
    # ------------------------------------------------------------------
    def _pi_weight(self) -> float:
        return torch.sigmoid(torch.tensor(self.pi_beta * self.pi)).item()

    # ------------------------------------------------------------------
    # 3) Hutchinson 估计 H·g 并做 PI 加权黎曼平滑
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _hutchinson_pi_smooth(self, grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        返回平滑后的三阶梯度 r_t = EMA(H·g, w=1/Tr[ℱ])
        这里用对角 Fisher 近似 Tr[ℱ]，并用 PI 门控权重
        """
        # 用参数自身的 Fisher 对角线作为 Tr[ℱ] 代理
        state = self.state[param]
        if 'fisher_diag' not in state:
            # 冷启动：用梯度平方初始化
            state['fisher_diag'] = grad.pow(2).detach()
        # EMA 更新对角 Fisher
        state['fisher_diag'].lerp_(grad.pow(2), 1 - self.stat_decay)

        tr_f = state['fisher_diag'].sum().item() + 1e-8
        w = 1.0 / tr_f  # Fisher 加权：平坦方向权重高

        # PI 门控：高 PI → 更相信历史，低 PI → 更相信当前
        pi_w = self._pi_weight()
        w *= (1 - pi_w)  # 高 PI → 小权重，低 PI → 大权重

        # Hutchinson 估计 H·g
        hutch = torch.zeros_like(grad)
        for _ in range(self.hutch_samples):
            v = torch.randn_like(grad)
            gv = torch.dot(grad.flatten(), v.flatten())
            Hv = torch.autograd.grad(gv, param, retain_graph=True, create_graph=False)[0]
            hutch += Hv * gv
        hutch /= self.hutch_samples

        # 黎曼平滑：EMA(H·g, w)
        buffer = state['fiena_buffer']
        buffer.lerp_(hutch, w)
        return buffer

    # ------------------------------------------------------------------
    # 4) 重写 step：在 KFAC 自然梯度之后，注入 FIENA 三阶正则
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1) 先执行标准 KFAC 自然梯度计算
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v

        # 2) 对自然梯度注入 FIENA 三阶正则
        for m in self.modules:
            if m.weight.grad is None:
                continue
            # 获取当前自然梯度
            nat_grad_w = updates[m][0]
            nat_grad_b = updates[m][1] if len(updates[m]) > 1 else None

            # 计算三阶平滑项
            r_w = self._hutchinson_pi_smooth(
                m.weight.grad.data, m.weight
            )
            # 注入：g_nat ← g_nat + λ·r_t
            nat_grad_w += self.lambda_ * r_w.reshape(nat_grad_w.shape)

            if nat_grad_b is not None and m.bias is not None:
                r_b = self._hutchinson_pi_smooth(
                    m.bias.grad.data, m.bias
                )
                nat_grad_b += self.lambda_ * r_b.reshape(nat_grad_b.shape)

            # 写回梯度缓冲区
            m.weight.grad.data.copy_(nat_grad_w)
            if nat_grad_b is not None:
                m.bias.grad.data.copy_(nat_grad_b)

        # 3) KL-clipping 与 Muon 正交化（复用 FOG 流程）
        self._kl_clip_and_update_grad(updates, lr)

        # 4) 最终参数更新（复用 FOG 流程）
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)
                p.data.add_(d_p, alpha=-lr)

        self.steps += 1