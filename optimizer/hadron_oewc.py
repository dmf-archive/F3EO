import torch

from .hadron import Hadron
from .kfac_utils import update_running_stat


class HadronOEWC(Hadron):
    """
    Hadron Optimizer with Online Elastic Weight Consolidation (OEWC).

    This version implements OEWC correctly:
    1. It uses the short-term Fisher inverse for natural gradient preconditioning
       to adapt to the current task.
    2. It uses the long-term Fisher accumulation as a quadratic penalty term
       (a-la EWC) to regularize parameter updates and prevent catastrophic
       forgetting.
    """
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 lifelong_decay=0.999,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 ewc_lambda=0.1,  # New hyperparameter for EWC strength
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 muon_momentum=0.95):

        super().__init__(model, lr, momentum, stat_decay, damping, kl_clip,
                         weight_decay, TCov, TInv, batch_averaged, muon_momentum)

        self.lifelong_decay = lifelong_decay
        self.ewc_lambda = ewc_lambda

        self.lifelong_initialized = False
        self.m_aa_lifelong, self.m_gg_lifelong = {}, {}
        self.star_params = {}

        # Store initial parameters as the first "star" parameters
        self.store_star_parameters()

    def store_star_parameters(self):
        """Stores the current parameters as the 'star' parameters for EWC."""
        self.star_params = {p: p.data.clone() for p in self.model.parameters() if p.requires_grad}

    def _init_lifelong_accumulators(self):
        """Lazily initialize lifelong accumulators based on the short-term ones."""
        for m in self.modules:
            if m in self.m_aa and m in self.m_gg:
                self.m_aa_lifelong[m] = self.m_aa[m].clone()
                self.m_gg_lifelong[m] = self.m_gg[m].clone()
        self.lifelong_initialized = True

    def _update_lifelong_stats(self):
        """Update the long-term running statistics of the Fisher matrices."""
        if not self.lifelong_initialized:
            self._init_lifelong_accumulators()

        for m in self.modules:
            if m in self.m_aa_lifelong:
                 update_running_stat(self.m_aa[m], self.m_aa_lifelong[m], self.lifelong_decay)
                 update_running_stat(self.m_gg[m], self.m_gg_lifelong[m], self.lifelong_decay)

    @torch.no_grad()
    def step(self, closure=None):
        # Update lifelong stats after short-term stats are updated by hooks
        if self.steps % self.TCov == 0:
            self._update_lifelong_stats()

        # Call the parent Hadron's step method to get the natural gradient update
        # This will compute the update based on the *short-term* Fisher
        super().step(closure)

        # Now, apply the EWC penalty as an additional gradient modification
        if self.ewc_lambda > 0:
            for m in self.modules:
                if m.weight.grad is None or m not in self.m_aa_lifelong:
                    continue

                # Get the parameter deviation from the star-parameters
                w_dev = m.weight.data - self.star_params[m.weight]

                # Apply the quadratic penalty using the long-term Fisher
                # The EWC gradient is F_long * (theta - theta*)
                # For KFAC, this is (G_long ⊗ A_long) * dev
                # which can be computed as G_long @ dev @ A_long
                ewc_grad_w = self.m_gg_lifelong[m] @ w_dev @ self.m_aa_lifelong[m]
                m.weight.grad.data.add_(ewc_grad_w, alpha=self.ewc_lambda)

                if m.bias is not None:
                    b_dev = m.bias.data - self.star_params[m.bias]
                    # For bias, the EWC grad is just a scaling
                    ewc_grad_b = self.m_gg_lifelong[m].diagonal().mean() * b_dev
                    m.bias.grad.data.add_(ewc_grad_b, alpha=self.ewc_lambda)
