from dataclasses import dataclass
from typing import Callable

import torch

from .ada_hessian import Adahessian
from .diag_fog import DiagFOG
from .diag_kfac import DiagKFACOptimizer
from .diag_kfac_muon import DiagKFACMuonOptimizer
from .fiena_fog import FIENA_FOG
from .fog import FOG
from .kfac import KFACOptimizer
from .muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam
from .pi_zpd import PI_ZPD


@dataclass
class OptimizerMetadata:
    cls: type[torch.optim.Optimizer]
    requires_model: bool = False
    requires_second_order: bool = False
    expects_param_groups: bool = False
    constructor_takes_model: bool = False


def _create_muon_optimizer(params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, None]:
    muon_groups = []
    for group in params:
        if group.get('use_diag_fog', True):
            muon_groups.append({
                'params': group['params'], 'use_muon': True,
                'lr': config.get('lr', 0.02), 'momentum': config.get('momentum', 0.95),
                'weight_decay': config.get('weight_decay', 0.1)
            })
        else:
            muon_groups.append({
                'params': group['params'], 'use_muon': False, 'lr': 1e-4,
                'betas': (0.9, 0.95), 'eps': 1e-10,
                'weight_decay': config.get('weight_decay', 0.1)
            })
    optimizer = SingleDeviceMuonWithAuxAdam(muon_groups)
    return optimizer, {}, None


def _configure_aux_adamw_groups(params: list[dict], config: dict):
    adam_lr = config.pop("adam_lr", 1e-4)
    adam_wd = config.pop("adam_weight_decay", 0.01)
    adam_betas = config.pop("adam_betas", (0.9, 0.95))
    for group in params:
        if not group.get('use_diag_fog', False):
            group.setdefault('lr', adam_lr)
            group.setdefault('weight_decay', adam_wd)
            group.setdefault('betas', adam_betas)


OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata | Callable] = {
    "AdamW": OptimizerMetadata(cls=torch.optim.AdamW, expects_param_groups=True),
    "AdaHessian": OptimizerMetadata(cls=Adahessian, requires_second_order=True),
    "Muon": _create_muon_optimizer,
    "KFAC": OptimizerMetadata(cls=KFACOptimizer, requires_model=True, constructor_takes_model=True),
    "FOG": OptimizerMetadata(cls=FOG, requires_model=True, expects_param_groups=True),
    "DiagKFAC": OptimizerMetadata(cls=DiagKFACOptimizer, requires_model=True, expects_param_groups=True),
    "DiagKFACMuon": OptimizerMetadata(cls=DiagKFACMuonOptimizer, requires_model=True, constructor_takes_model=True),
    "DiagFOG": OptimizerMetadata(cls=DiagFOG, requires_model=True, expects_param_groups=True),
    "PI_ZPD": OptimizerMetadata(cls=PI_ZPD, requires_model=True, expects_param_groups=True),
    "FIENA_FOG": OptimizerMetadata(cls=FIENA_FOG, requires_model=True, requires_second_order=True),
}


def get_optimizer(name: str, params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, dict | None]:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    entry = OPTIMIZER_REGISTRY[name]
    if not isinstance(entry, OptimizerMetadata):
        return entry(params, **config)

    meta = entry
    opt_config = config.copy()
    tags = {
        "requires_second_order": meta.requires_second_order,
        "accepts_pi_signal": name in ["PI_ZPD", "FIENA_FOG"],
    }

    if name in ["DiagFOG", "DiagKFAC", "FOG"]:
        _configure_aux_adamw_groups(params, opt_config)

    init_params = params if meta.expects_param_groups else next(iter(params))['params']

    if meta.requires_model and "model" not in opt_config:
        raise ValueError(f"Optimizer '{name}' requires a 'model' instance.")

    if meta.constructor_takes_model:
        model = opt_config.pop('model')
        optimizer = meta.cls(model, **opt_config)
    else:
        if not meta.requires_model:
            opt_config.pop('model', None)
        optimizer = meta.cls(init_params, **opt_config)

    pi_config = None
    if tags["accepts_pi_signal"]:
        pi_keys = ["gamma", "ema_beta", "alpha"]
        pi_config = {k: config[k] for k in pi_keys if k in config}

    return optimizer, tags, pi_config
