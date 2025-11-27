import torch

from .hadron import Hadron
from .kfac_utils import update_running_stat


class HadronOEWC(Hadron):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 lifelong_decay=0.999,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 ewc_lambda=0.1,
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

        self.store_star_parameters()

    def store_star_parameters(self):
        self.star_params = {p: p.data.clone() for p in self.model.parameters() if p.requires_grad}

    def _init_lifelong_accumulators(self):
        for m in self.modules:
            if m in self.m_aa and m in self.m_gg:
                self.m_aa_lifelong[m] = self.m_aa[m].clone()
                self.m_gg_lifelong[m] = self.m_gg[m].clone()
        self.lifelong_initialized = True

    def _update_lifelong_stats(self):
        if not self.lifelong_initialized:
            self._init_lifelong_accumulators()

        for m in self.modules:
            if m in self.m_aa_lifelong:
                 update_running_stat(self.m_aa[m], self.m_aa_lifelong[m], self.lifelong_decay)
                 update_running_stat(self.m_gg[m], self.m_gg_lifelong[m], self.lifelong_decay)

    @torch.no_grad()
    def step(self, closure=None):
        if self.steps % self.TCov == 0:
            self._update_lifelong_stats()

        super().step(closure)

        if self.ewc_lambda > 0:
            for m in self.modules:
                if m.weight.grad is None or m not in self.m_aa_lifelong:
                    continue

                w_dev = m.weight.data - self.star_params[m.weight]

                ewc_grad_w = self.m_gg_lifelong[m] @ w_dev @ self.m_aa_lifelong[m]
                m.weight.grad.data.add_(ewc_grad_w, alpha=self.ewc_lambda)

                if m.bias is not None:
                    b_dev = m.bias.data - self.star_params[m.bias]
                    ewc_grad_b = self.m_gg_lifelong[m].diagonal().mean() * b_dev
                    m.bias.grad.data.add_(ewc_grad_b, alpha=self.ewc_lambda)
