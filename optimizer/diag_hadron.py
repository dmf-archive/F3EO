import math

import torch
import torch.optim as optim

from .kfac_utils import update_running_stat

from .muon import zeropower_via_newtonschulz5


class DiagHadron(optim.Optimizer):
    def __init__(self, param_groups, model=None, lr=1e-3, stat_decay=0.95, TCov=10, TInv=100, muon_momentum=0.95, kl_clip=0.001, **kwargs):
        defaults = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        super().__init__(param_groups, defaults)

        self.param_groups_diag_hadron = []
        self.param_groups_adam = []

        for group in self.param_groups:
            if group.get('use_diag_hadron', False):
                self.param_groups_diag_hadron.append(group)
            else:
                self.param_groups_adam.append(group)

        self.model = model
        self.stat_decay = stat_decay
        self.TCov = TCov
        self.TInv = TInv
        self.muon_momentum = muon_momentum
        self.kl_clip = kl_clip

        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []

        if self.model is not None:
            self._prepare_model()

        self.steps = 0
        self.m_aa, self.m_gg = {}, {}

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            a = input[0].data
            a = a.reshape(-1, a.size(-1))
            if module.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
            aa_diag = (a * a).sum(dim=0)
            if self.steps == 0:
                self.m_aa[module] = torch.ones_like(aa_diag)
            update_running_stat(aa_diag, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            g = grad_output[0].data
            g = g.reshape(-1, g.size(-1))
            gg_diag = (g * g).sum(dim=0)
            if self.steps == 0:
                self.m_gg[module] = torch.ones_like(gg_diag)
            update_running_stat(gg_diag, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

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

    def _kl_clip_and_update_grad(self, updates, lr):
        vg_sum = 0
        for m in updates.keys():
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()

        nu = min(1.0, math.sqrt(self.kl_clip / (vg_sum + 1e-8)))

        for m in updates.keys():
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _muon_update(self, grad, momentum_buffer):
        momentum_buffer.lerp_(grad, 1 - self.muon_momentum)
        update = grad.lerp_(momentum_buffer, self.muon_momentum)
        if update.ndim == 4:
            update = update.view(update.size(0), -1)
        elif update.ndim == 1:
            return update
        update = zeropower_via_newtonschulz5(update, steps=5)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return update

    def step(self, closure=None):
        for group in self.param_groups:
            if group.get('use_diag_hadron', False):
                lr = group['lr']
                damping = group.get('damping', 0.001)
                weight_decay = group.get('weight_decay', 0)

                updates = {}
                diag_hadron_modules_in_group = [m for m in self.modules if any(p is p_in_group for p_in_group in group['params'] for p in m.parameters())]

                for m in diag_hadron_modules_in_group:
                    if not any(p.grad is not None for p in m.parameters()):
                        continue
                    p_grad_mat = self._get_matrix_form_grad(m, m.__class__.__name__)
                    v = self._get_natural_grad(m, p_grad_mat, damping)
                    updates[m] = v

                if updates:
                    self._kl_clip_and_update_grad(updates, lr)
                    for m in diag_hadron_modules_in_group:
                        if m not in updates:
                            continue
                        for p in m.parameters():
                            if p.grad is None or p.ndim < 2:
                                continue
                            state = self.state[p]
                            if 'muon_momentum_buffer' not in state:
                                state['muon_momentum_buffer'] = torch.zeros_like(p.grad)
                            stat_grad = p.grad.data.clone()
                            muon_update = self._muon_update(stat_grad, state["muon_momentum_buffer"])
                            p.grad.data.copy_(muon_update.reshape(p.grad.data.size()))

                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(p.data, alpha=weight_decay)
                    p.data.add_(d_p, alpha=-lr)

            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    state['step'] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    step_size = group['lr'] / bias_correction1

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

        self.steps += 1
