import math
import torch
import torch.optim as optim

from .kfac_utils import update_running_stat
from .muon import muon_update


class BlockHadron(optim.Optimizer):
    def __init__(self,
                 param_groups,
                 model=None,
                 lr=1e-3,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 muon_momentum=0.95,
                 block_size=64,
                 srm_gamma=0.0):

        defaults = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

        self.param_groups_block_hadron = []
        self.param_groups_adam = []

        for group in self.param_groups:
            if group.get('use_block_hadron', False):
                self.param_groups_block_hadron.append(group)
            else:
                self.param_groups_adam.append(group)

        self.model = model
        self.stat_decay = stat_decay
        self.damping = damping
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        self.muon_momentum = muon_momentum
        self.block_size = block_size
        self.srm_gamma = srm_gamma

        self.known_modules = {'Linear'}
        self.modules = []
        
        if self.model is not None:
            self._prepare_model()

        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.A_inv, self.G_inv = {}, {}

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            a = input[0].data.reshape(-1, module.in_features)
            if module.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
            
            for i in range(0, a.shape[1], self.block_size):
                block = a[:, i:i+self.block_size]
                cov_block = block.t() @ (block / block.size(0))
                
                if self.steps == 0:
                    if module not in self.m_aa:
                        self.m_aa[module] = {}
                    self.m_aa[module][i] = torch.ones_like(cov_block)
                
                update_running_stat(cov_block, self.m_aa[module][i], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            g = grad_output[0].data.reshape(-1, module.out_features)
            
            for i in range(0, g.shape[1], self.block_size):
                block = g[:, i:i+self.block_size]
                cov_block = block.t() @ (block / block.size(0))
                
                if self.steps == 0:
                    if module not in self.m_gg:
                        self.m_gg[module] = {}
                    self.m_gg[module][i] = torch.ones_like(cov_block)
                
                update_running_stat(cov_block, self.m_gg[module][i], self.stat_decay)

    def _update_inv(self, module):
        if module in self.m_aa and module in self.m_gg:
            self.A_inv[module] = {i: torch.inverse(self.m_aa[module][i] + self.damping * torch.eye(self.m_aa[module][i].size(0), device=self.m_aa[module][i].device)) for i in self.m_aa[module]}
            self.G_inv[module] = {i: torch.inverse(self.m_gg[module][i] + self.damping * torch.eye(self.m_gg[module][i].size(0), device=self.m_gg[module][i].device)) for i in self.m_gg[module]}

    def _get_matrix_form_grad(self, m):
        p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat):
        v = torch.zeros_like(p_grad_mat)
        
        for i_g, G_inv_block in self.G_inv[m].items():
            for i_a, A_inv_block in self.A_inv[m].items():
                grad_block = p_grad_mat[i_g:i_g+self.block_size, i_a:i_a+self.block_size]
                v_block = G_inv_block @ grad_block @ A_inv_block
                v[i_g:i_g+self.block_size, i_a:i_a+self.block_size] = v_block
        
        if m.bias is not None:
            v_w = v[:, :-1].view(m.weight.grad.data.size())
            v_b = v[:, -1:].view(m.bias.grad.data.size())
            return [v_w, v_b]
        else:
            return [v.view(m.weight.grad.data.size())]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups_adam:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: raise RuntimeError('AdamW does not support sparse gradients')
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

        for group in self.param_groups_block_hadron:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            natural_grads = {}
            block_hadron_modules = [m for m in self.modules if any(p is p_in_group for p_in_group in group['params'] for p in m.parameters())]

            if self.steps % self.TInv == 0:
                for m in block_hadron_modules:
                    self._update_inv(m)

            for m in block_hadron_modules:
                if m.weight.grad is None: continue
                if m not in self.A_inv or m not in self.G_inv: continue

                p_grad_mat = self._get_matrix_form_grad(m)
                g_nat_list = self._get_natural_grad(m, p_grad_mat)
                natural_grads[m] = g_nat_list

            final_updates = {}
            vg_sum = 0
            for m, g_nat_list in natural_grads.items():
                g_nat_w, g_nat_b = g_nat_list[0], g_nat_list[1] if len(g_nat_list) > 1 else None
                
                state_w = self.state[m.weight]
                if 'muon_momentum_buffer' not in state_w:
                    state_w['muon_momentum_buffer'] = torch.zeros_like(m.weight)
                
                hadron_update_w, _ = muon_update(g_nat_w, state_w['muon_momentum_buffer'], beta=self.muon_momentum, srm_gamma=self.srm_gamma)
                
                hadron_update_b = None
                if g_nat_b is not None:
                    state_b = self.state[m.bias]
                    if 'muon_momentum_buffer' not in state_b:
                        state_b['muon_momentum_buffer'] = torch.zeros_like(m.bias)
                    hadron_update_b, _ = muon_update(g_nat_b, state_b['muon_momentum_buffer'], beta=self.muon_momentum, srm_gamma=self.srm_gamma)

                final_updates[m] = [hadron_update_w, hadron_update_b]
                vg_sum += (hadron_update_w * m.weight.grad.data * lr**2).sum().item()
                if hadron_update_b is not None:
                    vg_sum += (hadron_update_b * m.bias.grad.data * lr**2).sum().item()
            
            nu = min(1.0, math.sqrt(self.kl_clip / (vg_sum + 1e-8)))

            for m, updates in final_updates.items():
                update_w, update_b = updates
                m.weight.grad.data.copy_(update_w)
                m.weight.grad.data.mul_(nu)
                if update_b is not None:
                    m.bias.grad.data.copy_(update_b)
                    m.bias.grad.data.mul_(nu)

            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad.data
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)
                p.data.add_(d_p, alpha=-lr)

        self.steps += 1