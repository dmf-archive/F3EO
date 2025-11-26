import torch

from .kfac import KFACOptimizer
from .muon import muon_update


class Hadron(KFACOptimizer):
    """
    Hadron Optimizer: Full KFAC + Muon operator composition.

    This optimizer combines KFAC and Muon in a sequential, operator-composition
    manner rather than a linear combination. It uses KFAC to compute a
    natural gradient direction in the statistical manifold, and then uses
    the Muon orthogonalization step to project this direction onto the
    Stiefel manifold, finding the nearest structure-preserving update.

    The pipeline is: g -> g_nat = F_emp⁻¹g -> g_update = Ortho(g_nat)
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

        # 1. Compute KFAC-preconditioned gradient (g_nat) for each module
        for m in self.modules:
            if m.weight.grad is None:
                continue

            if self.steps % self.TInv == 0:
                self._update_inv(m)

            p_grad_mat = self._get_matrix_form_grad(m, m.__class__.__name__)
            g_nat_mat_list = self._get_natural_grad(m, p_grad_mat, damping)
            natural_grads[m] = g_nat_mat_list

        # 2. Apply Muon orthogonalization to the natural gradients
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
            
            # Store diagnostic
            if not hasattr(self, 'diagnostics'):
                self.diagnostics = {}
            self.diagnostics[f"{m.__class__.__name__}_weight_rho"] = rho_w

            hadron_update_b = None
            if g_nat_b is not None and m.bias is not None:
                state_b = self.state[m.bias]
                # Bias terms are 1D, muon_update handles them by skipping orthogonalization
                hadron_update_b, _ = muon_update(
                    g_nat_b,
                    state_b['muon_momentum_buffer'],
                    beta=self.muon_momentum,
                    srm_gamma=self.srm_gamma
                )

            final_updates[m] = [hadron_update_w, hadron_update_b]

            # For KL clipping, we need the dot product of the final update and the *original* gradient
            vg_sum += (hadron_update_w.reshape(m.weight.grad.data.shape) * m.weight.grad.data * lr ** 2).sum().item()
            if hadron_update_b is not None:
                vg_sum += (hadron_update_b.reshape(m.bias.grad.data.shape) * m.bias.grad.data * lr ** 2).sum().item()

        # 3. KL clipping and gradient update
        # Add numerical protection for vg_sum
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

        # 4. Final parameter update using the modified gradients
        # The _step() from parent KFACOptimizer is a momentum SGD, which is
        # redundant and harmful after the Muon update. We replace it with a
        # direct application of the final computed gradient.
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr) # AdamW-style decay
                p.data.add_(d_p, alpha=-lr)

        self.steps += 1
