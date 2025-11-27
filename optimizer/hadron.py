import torch

from .kfac import KFACOptimizer
from .muon import muon_update


class Hadron(KFACOptimizer):
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
                 srm_gamma=0.0):

        super().__init__(model, lr, momentum, stat_decay, damping, kl_clip,
                         weight_decay, TCov, TInv, batch_averaged)

        self.muon_momentum = muon_momentum
        self.srm_gamma = srm_gamma
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['muon_momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("Closure not supported for Hadron.")

        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']

        natural_grads = {}

        for m in self.modules:
            if m.weight.grad is None:
                continue

            if self.steps % self.TInv == 0:
                self._update_inv(m)

            p_grad_mat = self._get_matrix_form_grad(m, m.__class__.__name__)
            g_nat_mat_list = self._get_natural_grad(m, p_grad_mat, damping)
            natural_grads[m] = g_nat_mat_list

        final_updates = {}
        vg_sum = 0
        for m in self.modules:
            if m.weight.grad is None:
                continue

            g_nat_w, g_nat_b = natural_grads[m][0], natural_grads[m][1] if len(natural_grads[m]) > 1 else None

            state_w = self.state[m.weight]
            hadron_update_w, rho_w = muon_update(
                g_nat_w,
                state_w['muon_momentum_buffer'],
                beta=self.muon_momentum,
                srm_gamma=self.srm_gamma
            )
            
            if not hasattr(self, 'diagnostics'):
                self.diagnostics = {}
            self.diagnostics[f"{m.__class__.__name__}_weight_rho"] = rho_w

            hadron_update_b = None
            if g_nat_b is not None and m.bias is not None:
                state_b = self.state[m.bias]
                hadron_update_b, _ = muon_update(
                    g_nat_b,
                    state_b['muon_momentum_buffer'],
                    beta=self.muon_momentum,
                    srm_gamma=self.srm_gamma
                )

            final_updates[m] = [hadron_update_w, hadron_update_b]

            vg_sum += (hadron_update_w.reshape(m.weight.grad.data.shape) * m.weight.grad.data * lr ** 2).sum().item()
            if hadron_update_b is not None:
                vg_sum += (hadron_update_b.reshape(m.bias.grad.data.shape) * m.bias.grad.data * lr ** 2).sum().item()

        vg_sum_safe = max(vg_sum, 1e-8)
        nu = min(1.0, (self.kl_clip / vg_sum_safe)**0.5)

        for m in self.modules:
            if m.weight.grad is None:
                continue

            hadron_update_w, hadron_update_b = final_updates[m]

            m.weight.grad.data.copy_(hadron_update_w.reshape(m.weight.grad.data.shape))
            m.weight.grad.data.mul_(nu)

            if hadron_update_b is not None:
                m.bias.grad.data.copy_(hadron_update_b.reshape(m.bias.grad.data.shape))
                m.bias.grad.data.mul_(nu)

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
