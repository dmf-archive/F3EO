from collections.abc import Callable as CallableABC
from math import sqrt

import torch

from .kfac import KFACOptimizer
from .muon import muon_update


class PI_Muon_KFAC(KFACOptimizer):

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
                 muon_momentum=0.95):

        super().__init__(model, lr, momentum, stat_decay, damping, kl_clip,
                         weight_decay, TCov, TInv, batch_averaged)
        self.muon_momentum = muon_momentum

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['muon_momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: None | CallableABC[[], torch.Tensor] = None, effective_gamma: float | None = None, pi_object=None):
        if closure is not None:
            raise NotImplementedError("Closure not supported for PI-Muon-KFAC.")

        pi_value = 0.5  # Default value
        if pi_object is not None:
            pi_value = pi_object.raw_pi
        elif effective_gamma is not None:
            pi_value = torch.tanh(torch.tensor(effective_gamma)).item()

        lambda_t = 1.0 - pi_value
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']

        kfac_updates = {}
        muon_updates = {}

        for m in self.modules:
            if m.weight.grad is None:
                continue

            state_w = self.state[m.weight]
            if 'muon_momentum_buffer' not in state_w:
                state_w['muon_momentum_buffer'] = torch.zeros_like(m.weight)

            muon_grad_w = muon_update(
                m.weight.grad.data,
                state_w['muon_momentum_buffer'],
                beta=self.muon_momentum
            ).reshape(m.weight.grad.data.shape)

            muon_grad_b = None
            if m.bias is not None and m.bias.grad is not None:
                state_b = self.state[m.bias]
                if 'muon_momentum_buffer' not in state_b:
                    state_b['muon_momentum_buffer'] = torch.zeros_like(m.bias)
                muon_grad_b = muon_update(
                    m.bias.grad.data,
                    state_b['muon_momentum_buffer'],
                    beta=self.muon_momentum
                ).reshape(m.bias.grad.data.shape)

            muon_updates[m] = [muon_grad_w, muon_grad_b]

            if self.steps % self.TInv == 0:
                self._update_inv(m)

            p_grad_mat = self._get_matrix_form_grad(m, m.__class__.__name__)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            kfac_updates[m] = v

        final_updates = {}
        vg_sum = 0
        for m in self.modules:
            if m.weight.grad is None:
                continue

            kfac_update = kfac_updates[m]
            muon_update_vals = muon_updates[m]

            g_fisher_w = kfac_update[0]
            g_muon_w = muon_update_vals[0]
            g_update_w = (1 - lambda_t) * g_fisher_w + lambda_t * g_muon_w

            g_update_b = None
            if len(kfac_update) > 1 and len(muon_update_vals) > 1:
                g_fisher_b = kfac_update[1]
                g_muon_b = muon_update_vals[1]
                if g_fisher_b is not None and g_muon_b is not None:
                    g_update_b = (1 - lambda_t) * g_fisher_b + lambda_t * g_muon_b

            final_updates[m] = [g_update_w, g_update_b]

            vg_sum += (g_update_w * m.weight.grad.data * lr ** 2).sum().item()
            if g_update_b is not None:
                vg_sum += (g_update_b * m.bias.grad.data * lr ** 2).sum().item()

        nu = min(1.0, sqrt(self.kl_clip / (vg_sum + 1e-8)))

        for m in self.modules:
            if m.weight.grad is None:
                continue

            g_update_w, g_update_b = final_updates[m]
            m.weight.grad.data.copy_(g_update_w)
            m.weight.grad.data.mul_(nu)
            if g_update_b is not None:
                m.bias.grad.data.copy_(g_update_b)
                m.bias.grad.data.mul_(nu)

        self._step(None)
        self.steps += 1
