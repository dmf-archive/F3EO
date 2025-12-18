import torch
from torch.optim.optimizer import Optimizer


@torch.jit.script
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16) if G.dtype == torch.float32 else G

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X.div(X.norm(p=2.0, dim=[-2, -1], keepdim=True).add(1e-7))

    for _ in range(steps):
        A = X.matmul(X.mT)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)


class FS_AdaRMSuon(Optimizer):
    def __init__(self, params, rho=0.05, **kwargs):
        defaults = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'ns_steps': 5,
        }
        super().__init__(params, defaults)
        self.rho = rho

    def step(self, closure):
        if closure is None:
            raise ValueError("SAM-based optimizers require a closure to re-evaluate the loss.")

        with torch.enable_grad():
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                is_rmsuon_group = group.get('is_rmsuon_group', False)
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    state['g_orig'] = p.grad.clone()

                    if not is_rmsuon_group:
                        continue

                    if 'exp_avg_sq' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Standard SAM perturbation
                    grad = state['g_orig']
                    grad_norm = grad.norm(p=2.0)
                    
                    if grad_norm > 0:
                        e_w = (grad / grad_norm) * self.rho
                        p.add_(e_w)
                        state['e_w'] = e_w
                    
                    # Update v_hat based on original gradient for the second step's use
                    state['exp_avg_sq'].mul_(group['betas'][1]).addcmul_(state['g_orig'], state['g_orig'], value=1 - group['betas'][1])

        with torch.enable_grad():
            closure()

        with torch.no_grad():
            for group in self.param_groups:
                is_rmsuon_group = group.get('is_rmsuon_group', False)
                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    if 'e_w' in state:
                        p.sub_(state['e_w'])

                    g_sam = p.grad

                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1
                    step = state['step']
                    exp_avg = state['exp_avg']
                    
                    exp_avg.mul_(beta1).add_(g_sam, alpha=1 - beta1)

                    if is_rmsuon_group:
                        ns_steps = group['ns_steps']
                        exp_avg_sq = state['exp_avg_sq']
                        
                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        m_hat = exp_avg / bias_correction1
                        v_hat = exp_avg_sq / bias_correction2

                        m_scaled = m_hat / (v_hat.sqrt() + eps)
                        energy = m_scaled.norm()

                        original_shape = m_scaled.shape
                        m_scaled_flat = m_scaled.view(m_scaled.size(0), -1) if p.ndim == 4 else m_scaled
                        
                        s_ortho = zeropower_via_newtonschulz5(m_scaled_flat, steps=ns_steps)

                        if p.ndim == 4:
                            s_ortho = s_ortho.view(original_shape)

                        update = energy * s_ortho

                        if weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)

                        p.add_(update, alpha=-lr)
                    else: # Fallback to standard AdamW for non-RMSuon groups
                        exp_avg_sq = state.get('exp_avg_sq')
                        if exp_avg_sq is None:
                             exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)
                             state['exp_avg_sq'] = exp_avg_sq
                        
                        exp_avg_sq.mul_(beta2).addcmul_(g_sam, g_sam, value=1 - beta2)
                        
                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step
                        
                        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                        step_size = lr / bias_correction1

                        if weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)
                        
                        p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
