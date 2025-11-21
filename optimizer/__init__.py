from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class OptimizerMetadata:
    cls_name: str
    module_name: str
    requires_model: bool = False
    requires_second_order: bool = False
    expects_param_groups: bool = False
    constructor_takes_model: bool = False
    requires_loss_for_step: bool = False

def _import_optimizer(module_name: str, class_name: str) -> type[torch.optim.Optimizer]:
    from importlib import import_module
    module = import_module(f".{module_name}", package="optimizer")
    return getattr(module, class_name)

def _create_muon_optimizer(params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, None]:
    from .muon import SingleDeviceMuonWithAuxAdam
    muon_groups = []
    for group in params:
        if group.get('use_diag_hadron', True):
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

def _create_nuclear_muon_optimizer(params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, None]:
    from .nuclear_muon import SingleDeviceNuclearMuonWithAuxAdam

    # EXACTLY replicate Muon's parameter grouping logic, just change the flag name
    nuclear_groups = []
    for group in params:
        if group.get('use_diag_hadron', True):  # Same logic as Muon
            nuclear_groups.append({
                'params': group['params'], 'use_nuclear_muon': True,
                'lr': config.get('lr', 0.02), 'momentum': config.get('momentum', 0.95),
                'weight_decay': config.get('weight_decay', 0.1)
            })
        else:
            nuclear_groups.append({
                'params': group['params'], 'use_nuclear_muon': False, 'lr': 1e-4,
                'betas': (0.9, 0.95), 'eps': 1e-10,
                'weight_decay': config.get('weight_decay', 0.1)
            })
    optimizer = SingleDeviceNuclearMuonWithAuxAdam(nuclear_groups)
    return optimizer, {}, None

def _configure_aux_adamw_groups(params: list[dict], config: dict):
    adam_lr = config.pop("adam_lr", 1e-4)
    adam_wd = config.pop("adam_weight_decay", 0.01)
    adam_betas = config.pop("adam_betas", (0.9, 0.95))
    for group in params:
        if not group.get('use_diag_hadron', False):
            group.setdefault('lr', adam_lr)
            group.setdefault('weight_decay', adam_wd)
            group.setdefault('betas', adam_betas)


OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata | Callable] = {
    "AdamW": OptimizerMetadata(cls_name="AdamW", module_name="torch.optim", expects_param_groups=True),
    "AdaHessian": OptimizerMetadata(cls_name="Adahessian", module_name="ada_hessian", requires_second_order=True),
    "Muon": _create_muon_optimizer,
    "KFAC": OptimizerMetadata(cls_name="KFACOptimizer", module_name="kfac", requires_model=True, constructor_takes_model=True),
    "Hadron": OptimizerMetadata(cls_name="Hadron", module_name="hadron", requires_model=True, constructor_takes_model=True),
    "DiagKFAC": OptimizerMetadata(cls_name="DiagKFACOptimizer", module_name="diag_kfac", requires_model=True, expects_param_groups=True),
    "DiagKFACMuon": OptimizerMetadata(cls_name="DiagKFACMuonOptimizer", module_name="diag_kfac_muon", requires_model=True, constructor_takes_model=True),
    "DiagHadron": OptimizerMetadata(cls_name="DiagHadron", module_name="diag_hadron", requires_model=True, expects_param_groups=True),
    "NuclearMuon": _create_nuclear_muon_optimizer,
    "SSK": OptimizerMetadata(cls_name="SSK", module_name="ssk", requires_model=True, constructor_takes_model=True),
}


def get_optimizer(name: str, params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, dict | None]:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    entry = OPTIMIZER_REGISTRY[name]
    if not isinstance(entry, OptimizerMetadata):
        return entry(params, **config)

    meta = entry

    if meta.module_name == "torch.optim":
        OptimizerClass = getattr(torch.optim, meta.cls_name)
    else:
        OptimizerClass = _import_optimizer(meta.module_name, meta.cls_name)

    opt_config = config.copy()
    tags = {
        "requires_second_order": meta.requires_second_order,
        "requires_loss_for_step": meta.requires_loss_for_step,
        "accepts_pi_signal": name in ["PI_ZPD", "FIENA_FOG"],
    }

    if name in ["DiagHadron", "DiagKFAC", "Hadron", "SSK"]:
        _configure_aux_adamw_groups(params, opt_config)

    init_params = params if meta.expects_param_groups else next(iter(params))['params']

    if meta.requires_model and "model" not in opt_config:
        raise ValueError(f"Optimizer '{name}' requires a 'model' instance.")

    if meta.constructor_takes_model:
        model = opt_config.pop('model')
        optimizer = OptimizerClass(model, **opt_config)
    else:
        if not meta.requires_model:
            opt_config.pop('model', None)
        optimizer = OptimizerClass(init_params, **opt_config)

    pi_config = None
    if tags["accepts_pi_signal"]:
        pi_keys = ["gamma", "ema_beta", "alpha"]
        pi_config = {k: config[k] for k in pi_keys if k in config}

    return optimizer, tags, pi_config
