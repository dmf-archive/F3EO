import torch
from torch.optim.optimizer import Optimizer
from .ada_rmsuon import zeropower_via_newtonschulz5

class ARGOptimizer(Optimizer):
    def __init__(self, params, **kwargs):
        defaults = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'ns_steps': 5,
            'alpha_gam': 0.1,   # Strength of the flatness penalty
            'rho': 0.05,       # Perturbation radius
            'adaptive': True,  # Adaptive GAM (scaling by parameter norm)
        }
        for k, v in kwargs.items():
            if k in defaults:
                defaults[k] = v
        super().__init__(params, defaults)

    def step(self, closure):
        if closure is None:
            raise ValueError("ARG requires a closure.")

        # 1. First-order sampling at theta
        with torch.enable_grad():
            loss = closure()
            # We need create_graph=True to differentiate through the gradients for GAM
            loss.backward(create_graph=True)
            
            params_with_grad = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad and p.grad is not None:
                        params_with_grad.append(p)
            
            if not params_with_grad:
                return loss

            # Save g1 and compute f = nabla ||g1_nat||^2
            g1 = [p.grad.clone() for p in params_with_grad]
            
            # Update step count and second moments for pre-whitening
            first_p = params_with_grad[0]
            state_p = self.state[first_p]
            if 'step' not in state_p: state_p['step'] = 0
            global_step = state_p['step'] + 1
            
            grad_norm_sq = 0.0
            for i, p in enumerate(params_with_grad):
                state = self.state[p]
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                _, beta2 = self.param_groups[0]['betas']
                # IMPORTANT: Update second moments without tracking gradients
                with torch.no_grad():
                    state['exp_avg_sq'].mul_(beta2).addcmul_(g1[i], g1[i], value=1 - beta2)
                
                v_hat = state['exp_avg_sq'] / (1 - beta2 ** global_step)
                
                # Natural gradient component
                g_nat = g1[i] / (v_hat.sqrt() + self.param_groups[0]['eps'])
                grad_norm_sq += g_nat.pow(2).sum()

            # Compute f = nabla_theta ||g_nat||^2
            f = torch.autograd.grad(grad_norm_sq, params_with_grad, retain_graph=False)
            f_dict = {id(p): grad_f for p, grad_f in zip(params_with_grad, f)}

        # 2. Perturbation: theta_adv = theta + rho * f / ||f||
        rho = self.param_groups[0]['rho']
        adaptive = self.param_groups[0]['adaptive']
        
        f_total_norm = torch.sqrt(sum(grad_f.pow(2).sum() for grad_f in f) + 1e-12)
        scale = rho / f_total_norm
        
        with torch.no_grad():
            for p in params_with_grad:
                perturb = f_dict[id(p)] * scale
                if adaptive:
                    perturb.mul_(p.abs())
                p.add_(perturb)
                self.state[p]['last_f_perturb'] = perturb

        # 3. Second-order sampling at theta_adv
        self.zero_grad()
        with torch.enable_grad():
            loss_adv = closure()
            loss_adv.backward(create_graph=True)
            # g2 is now in p.grad
            grad_norm_sq_adv = 0.0
            for p in params_with_grad:
                v_hat = self.state[p]['exp_avg_sq'] / (1 - self.param_groups[0]['betas'][1] ** global_step)
                g_nat_adv = p.grad / (v_hat.sqrt() + self.param_groups[0]['eps'])
                grad_norm_sq_adv += g_nat_adv.pow(2).sum()
            
            # Compute h_adv = nabla_theta_adv ||g_nat_adv||^2
            h_adv = torch.autograd.grad(grad_norm_sq_adv, params_with_grad)
            h_adv_dict = {id(p): h for p, h in zip(params_with_grad, h_adv)}

        # 4. Restore theta and combine gradients
        alpha_gam = self.param_groups[0]['alpha_gam']
        with torch.no_grad():
            for i, p in enumerate(params_with_grad):
                p.sub_(self.state[p]['last_f_perturb'])
                p.grad = g1[i].add(h_adv_dict[id(p)], alpha=alpha_gam)

        # 5. Execute AdaRMSuon Update Logic
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
                
                state['step'] = global_step
                exp_avg = state['exp_avg']
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
