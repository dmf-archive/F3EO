import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional


class PI_ZPD(Optimizer):
    """
    显式 PI-ZPD：用实时 PI 动态软化交叉熵损失
    L = w(PI) · CE，其中 w(PI) = σ(β · PI)
    基于 DiagFOG 实现，使用二阶优化而非 SGD
    """

    def __init__(self, param_groups, model=None, lr: float = 1e-3, beta: float = 1.0, 
                 stat_decay: float = 0.95, damping: float = 1e-3, muon_momentum: float = 0.95):
        # 使用 DiagFOG 的默认参数结构
        defaults = dict(lr=lr, beta=beta, damping=damping, muon_momentum=muon_momentum,
                       stat_decay=stat_decay, use_diag_fog=True)
        super().__init__(param_groups, defaults)
        
        # 初始化 DiagFOG 的核心组件
        self.model = model
        self.stat_decay = stat_decay
        self.damping = damping
        self.muon_momentum = muon_momentum
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        
        if self.model is not None:
            self._prepare_model()
            
        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        
        # 从 DiagFOG 导入必要的工具函数
        from .kfac_utils import update_running_stat
        from .muon import zeropower_via_newtonschulz5
        self._update_running_stat = update_running_stat
        self._zeropower_via_newtonschulz5 = zeropower_via_newtonschulz5

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % 10 == 0:  # TCov = 10
            a = input[0].data
            a = a.reshape(-1, a.size(-1))
            if module.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
            aa_diag = (a * a).sum(dim=0)
            if self.steps == 0:
                self.m_aa[module] = torch.ones_like(aa_diag)
            self._update_running_stat(aa_diag, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % 10 == 0:  # TCov = 10
            g = grad_output[0].data
            g = g.reshape(-1, g.size(-1))
            gg_diag = (g * g).sum(dim=0)
            if self.steps == 0:
                self.m_gg[module] = torch.ones_like(gg_diag)
            self._update_running_stat(gg_diag, self.m_gg[module], self.stat_decay)

    def _get_matrix_form_grad(self, m, classname):
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.reshape(m.weight.grad.data.size(0), -1)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        A_inv_diag = 1.0 / (self.m_aa[m] + damping)
        G_inv_diag = 1.0 / (self.m_gg[m] + damping)
        v = p_grad_mat * (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
        
        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v

    def set_pi(self, pi: float):
        """外部调用：传入当前批次的平均 PI"""
        self.pi = pi

    def _pi_weight(self) -> float:
        return torch.sigmoid(torch.tensor(self.defaults['beta'] * self.pi)).item()

    def _muon_update(self, grad, momentum_buffer):
        momentum_buffer.lerp_(grad, 1 - self.muon_momentum)
        update = grad.lerp_(momentum_buffer, self.muon_momentum)
        if update.ndim == 4:  # conv filters
            update = update.view(update.size(0), -1)
        elif update.ndim == 1:
            return update
        update = self._zeropower_via_newtonschulz5(update, steps=5)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 计算 PI 门控权重
        w = self._pi_weight()
        
        # 对每个参数组应用 DiagFOG 更新逻辑
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0)
            
            # 获取该组相关的模块
            group_modules = [m for m in self.modules if any(p in group['params'] for p in m.parameters())]
            
            updates = {}
            for m in group_modules:
                if not any(p.grad is not None for p in m.parameters()):
                    continue
                    
                classname = m.__class__.__name__
                p_grad_mat = self._get_matrix_form_grad(m, classname)
                v = self._get_natural_grad(m, p_grad_mat, self.damping)
                updates[m] = v
            
            # KL 裁剪
            if updates:
                vg_sum = 0
                for m in updates:
                    v = updates[m]
                    vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
                    if len(v) > 1 and m.bias is not None:
                        vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
                
                nu = min(1.0, (0.001 / (vg_sum + 1e-8))**0.5)  # kl_clip = 0.001
                
                for m in updates:
                    v = updates[m]
                    m.weight.grad.data.copy_(v[0])
                    m.weight.grad.data.mul_(nu)
                    if len(v) > 1 and m.bias is not None:
                        m.bias.grad.data.copy_(v[1])
                        m.bias.grad.data.mul_(nu)
            
            # Muon 正交化
            for m in group_modules:
                for p in m.parameters():
                    if p.grad is None or p.ndim < 2:
                        continue
                    state = self.state[p]
                    if 'muon_momentum_buffer' not in state:
                        state['muon_momentum_buffer'] = torch.zeros_like(p.grad)
                    
                    stat_grad = p.grad.data.clone()
                    muon_update = self._muon_update(stat_grad, state['muon_momentum_buffer'])
                    p.grad.data.copy_(muon_update.reshape(p.grad.data.size()))
            
            # 最终参数更新，应用 PI 门控
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                # 应用 PI 门控权重到学习率
                p.data.add_(d_p, alpha=-lr * w)
        
        self.steps += 1
        return loss