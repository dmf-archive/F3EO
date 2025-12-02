import torch
import math
from torch.optim.optimizer import Optimizer
from torch.nn.functional import unfold

from .kfac_utils import update_running_stat
from .muon import zeropower_via_newtonschulz5

class KFACRMSuon(Optimizer):
    def __init__(self, param_groups, model=None, **kwargs):
        if model is None:
            raise ValueError("KFACRMSuon requires a model to be passed in.")
        
        defaults = dict(lr=kwargs.get('lr', 1e-3), 
                        betas=kwargs.get('betas', (0.9, 0.999)),
                        eps=kwargs.get('eps', 1e-8),
                        weight_decay=kwargs.get('weight_decay', 0),
                        stat_decay=kwargs.get('stat_decay', 0.95), 
                        TCov=kwargs.get('TCov', 10), 
                        damping=kwargs.get('damping', 0.001), 
                        ns_steps=kwargs.get('ns_steps', 5))
        super().__init__(param_groups, defaults)

        self.param_groups_kfac = [pg for pg in self.param_groups if pg.get('use_kfac_rmsuon', False)]
        self.param_groups_adam = [pg for pg in self.param_groups if not pg.get('use_kfac_rmsuon', False)]
        
        self.model = model
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self._prepare_model()

        self.steps = 0
        self.m_aa, self.m_gg = {}, {}

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.param_groups_kfac[0].get('TCov', 10) == 0:
            a = input[0].data
            
            if isinstance(module, torch.nn.Conv2d):
                a = unfold(a, module.kernel_size, padding=module.padding, stride=module.stride)
                a = a.permute(0, 2, 1).contiguous().view(-1, a.size(1))
            else: # Linear
                if a.ndim > 2:
                    a = a.reshape(-1, a.size(-1))

            if module.bias is not None:
                a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)
            
            aa_diag = (a * a).sum(dim=0) / a.size(0)
            if self.steps == 0 or module not in self.m_aa:
                self.m_aa[module] = torch.ones_like(aa_diag)
            update_running_stat(aa_diag, self.m_aa[module], self.param_groups_kfac[0].get('stat_decay', 0.95))

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.param_groups_kfac[0].get('TCov', 10) == 0:
            g = grad_output[0].data
            if isinstance(module, torch.nn.Conv2d):
                g = g.permute(0, 2, 3, 1).contiguous().view(-1, g.size(1))
            else: # Linear
                if g.ndim > 2:
                    g = g.reshape(-1, g.size(-1))
            
            gg_diag = (g * g).sum(dim=0) / g.size(0)
            if self.steps == 0 or module not in self.m_gg:
                self.m_gg[module] = torch.ones_like(gg_diag)
            update_running_stat(gg_diag, self.m_gg[module], self.param_groups_kfac[0].get('stat_decay', 0.95))

    def _prepare_model(self):
        kfac_params = {p for group in self.param_groups_kfac for p in group['params']}
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules and any(p in kfac_params for p in module.parameters()):
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _get_matrix_form_grad(self, m):
        grad = m.weight.grad.data
        if m.__class__.__name__ == 'Conv2d':
            grad = grad.reshape(grad.size(0), -1)
        if m.bias is not None and m.bias.grad is not None:
            grad = torch.cat([grad, m.bias.grad.data.view(-1, 1)], 1)
        return grad

    def _kfac_rmsuon_step(self, group):
        lr = group['lr']
        weight_decay = group['weight_decay']
        damping = group['damping']
        ns_steps = group['ns_steps']
        beta1, _ = group['betas']
        
        group_modules = [m for m in self.modules if any(p is p_in_group for p_in_group in group['params'] for p in m.parameters())]

        for m in group_modules:
            if m.weight.grad is None:
                continue

            grad_mat = self._get_matrix_form_grad(m)
            
            param_state = self.state[m.weight]
            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.zeros_like(grad_mat)
            
            buf = param_state['momentum_buffer']
            buf.mul_(beta1).add_(grad_mat, alpha=1 - beta1)
            m_hat = buf

            # Using the correct logic from my second attempt, which user confirmed
            A_inv_diag = 1.0 / (self.m_aa.get(m) + damping)
            G_inv_diag = 1.0 / (self.m_gg.get(m) + damping)

            if m.bias is not None:
                 A_inv_diag_ = A_inv_diag[:-1]
            else:
                 A_inv_diag_ = A_inv_diag
            
            preconditioned_grad = m_hat * (G_inv_diag.unsqueeze(1) * A_inv_diag_.unsqueeze(0))

            energy = preconditioned_grad.norm()
            s_ortho = zeropower_via_newtonschulz5(preconditioned_grad, steps=ns_steps)
            update = energy * s_ortho

            if m.bias is not None and m.bias.grad is not None:
                update_w = update[:, :-1].reshape(m.weight.shape)
                update_b = update[:, -1:].reshape(m.bias.shape)
                
                if weight_decay != 0:
                    m.weight.data.add_(m.weight.data, alpha=-weight_decay * lr)
                    m.bias.data.add_(m.bias.data, alpha=-weight_decay * lr)
                
                m.weight.data.add_(update_w, alpha=-lr)
                m.bias.data.add_(update_b, alpha=-lr)
            else:
                update_w = update.reshape(m.weight.shape)
                if weight_decay != 0:
                    m.weight.data.add_(m.weight.data, alpha=-weight_decay * lr)
                m.weight.data.add_(update_w, alpha=-lr)

    def _adam_step(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            step_size = group['lr'] / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group.get('eps', 1e-8))

            p.data.addcdiv_(exp_avg, denom, value=-step_size)
            if group['weight_decay'] != 0:
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

    def step(self, closure=None):
        for group in self.param_groups_kfac:
            self._kfac_rmsuon_step(group)
        
        for group in self.param_groups_adam:
            self._adam_step(group)

        self.steps += 1
