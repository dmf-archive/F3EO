import torch
from torch.optim.optimizer import Optimizer
from .ada_rmsuon import zeropower_via_newtonschulz5

class ARSOptimizer(Optimizer):
    def __init__(self, params, **kwargs):
        defaults = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'ns_steps': 5,
            'rho': 0.05,
            'k': 1,
            'alpha': 0.7,
            'adaptive': True,
        }
        for k, v in kwargs.items():
            if k in defaults:
                defaults[k] = v
        super().__init__(params, defaults)

    def step(self, closure):
        if closure is None:
            raise ValueError("ARS requires a closure.")

        # 0. Global Step Tracking
        first_p = self.param_groups[0]['params'][0]
        state_p = self.state[first_p]
        if 'step' not in state_p: state_p['step'] = 0
        state_p['step'] += 1
        global_step = state_p['step']
        
        k = self.param_groups[0]['k']
        is_sync_step = (global_step % k == 1) or (k <= 1)

        if is_sync_step:
            # 1. Probing Step (Side-effect free for optimizer states)
            with torch.enable_grad():
                loss = closure()
                loss.backward()
            
            with torch.no_grad():
                for group in self.param_groups:
                    rho = group['rho']
                    adaptive = group['adaptive']
                    eps_val = group['eps']
                    beta2 = group['betas'][1]
                    
                    for p in group['params']:
                        if p.grad is None: continue
                        state = self.state[p]
                        
                        # Initialize states if needed, but don't update them yet
                        if 'exp_avg_sq' not in state:
                            state['exp_avg_sq'] = torch.zeros_like(p)
                        
                        # Use existing v_hat for perturbation calculation
                        v_hat = state['exp_avg_sq'] / (1 - beta2 ** max(1, global_step - 1) + 1e-12)
                        
                        g_nat = p.grad / (v_hat.sqrt() + eps_val)
                        if adaptive:
                            g_nat.mul_(p.abs())
                        
                        norm = g_nat.norm() + 1e-12
                        perturb = g_nat * (rho / norm)
                        
                        state['last_eps'] = perturb
                        state['g_base'] = p.grad.clone() # Save g1 for flatness logic
                        p.add_(perturb)

            # 2. Adversarial Step (The real update)
            self.zero_grad()
            with torch.enable_grad():
                loss_adv = closure()
                loss_adv.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None: continue
                        state = self.state[p]
                        p.sub_(state['last_eps']) # Restore theta
                        
                        # Now p.grad is g2. We update states using g2.
                        beta1, beta2 = group['betas']
                        
                        # Update v (Second Moment)
                        state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                        
                        if k > 1:
                            g_base = state['g_base']
                            g_adv = p.grad
                            dot = (g_adv * g_base).sum()
                            base_norm_sq = (g_base * g_base).sum() + 1e-12
                            # Orthogonal residual: v = g_adv - proj_g_base(g_adv)
                            state['flatness_v'] = g_adv - (dot / base_norm_sq) * g_base
        else:
            # Non-sync step: Standard update with injected shear force
            with torch.enable_grad():
                loss = closure()
                loss.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    alpha = group['alpha']
                    beta2 = group['betas'][1]
                    for p in group['params']:
                        if p.grad is None: continue
                        state = self.state[p]
                        
                        # Update v (Second Moment)
                        if 'exp_avg_sq' not in state:
                            state['exp_avg_sq'] = torch.zeros_like(p)
                        state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                        if 'flatness_v' in state:
                            v = state['flatness_v']
                            g_norm = p.grad.norm()
                            v_norm = v.norm() + 1e-12
                            # Inject shear force
                            p.grad.add_(v, alpha=alpha * (g_norm / v_norm))

        # 3. Final AdaRMSuon Update
        with torch.no_grad():
            self._ada_rmsuon_update(global_step)
        
        return loss

    @torch.no_grad()
    def _ada_rmsuon_update(self, global_step):
        for group in self.param_groups:
            is_rmsuon = group.get('is_rmsuon_group', False)
            beta1, beta2 = group['betas']
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            ns_steps = group.get('ns_steps', 5)

            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                # Update m (Momentum) using current p.grad (which might be g2 or corrected g)
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                
                m_hat = exp_avg / (1 - beta1 ** global_step)
                v_hat = state['exp_avg_sq'] / (1 - beta2 ** global_step)

                if is_rmsuon:
                    m_scaled = m_hat / (v_hat.sqrt() + eps)
                    energy = m_scaled.norm()
                    
                    m_flat = m_scaled.view(m_scaled.size(0), -1) if p.ndim == 4 else m_scaled
                    s_ortho = zeropower_via_newtonschulz5(m_flat, steps=ns_steps)
                    if p.ndim == 4: s_ortho = s_ortho.view(m_scaled.shape)
                    
                    update = energy * s_ortho
                    if wd != 0: p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    denom = v_hat.sqrt().add_(eps)
                    if wd != 0: p.mul_(1 - lr * wd)
                    p.addcdiv_(m_hat, denom, value=-lr)
