import torch
from torch.optim import AdamW


class AdamW_PI(AdamW):
    """
    A lightweight wrapper for AdamW that incorporates a PI-based adaptive weight decay.
    """
    def __init__(self, params, **kwargs):
        # The 'gamma' parameter is now implicitly handled by the external effective_gamma
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None, effective_gamma=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
            effective_gamma: An externally computed signal to modulate weight decay.
        """
        # Calculate the adaptive weight decay multiplier
        multiplier = 1.0
        if effective_gamma is not None:
            multiplier = torch.exp(torch.tensor(effective_gamma)).item()

        # Temporarily set the adaptive weight decay for this step
        original_wds = [group['weight_decay'] for group in self.param_groups]
        for group in self.param_groups:
            group['weight_decay'] = group['weight_decay'] * multiplier

        # Call the original AdamW step function
        loss = super().step(closure)

        # Restore original weight decay values to avoid state pollution
        for i, group in enumerate(self.param_groups):
            group['weight_decay'] = original_wds[i]

        return loss
