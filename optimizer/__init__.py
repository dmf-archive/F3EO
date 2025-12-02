from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

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
    is_sam: bool = False

OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata] = {
    "AdamW": OptimizerMetadata(cls_name="AdamW", module_name="torch.optim", expects_param_groups=True),
    "AdaHessian": OptimizerMetadata(cls_name="Adahessian", module_name="ada_hessian", requires_second_order=True),
    "Muon": OptimizerMetadata(cls_name="SingleDeviceMuonWithAuxAdam", module_name="muon", expects_param_groups=True),
    "KFAC": OptimizerMetadata(cls_name="KFACOptimizer", module_name="kfac", requires_model=True, constructor_takes_model=True),
    "Hadron": OptimizerMetadata(cls_name="Hadron", module_name="hadron", requires_model=True, constructor_takes_model=True),
    "DiagKFAC": OptimizerMetadata(cls_name="DiagKFACOptimizer", module_name="diag_kfac", requires_model=True, expects_param_groups=True),
    "DiagKFACMuon": OptimizerMetadata(cls_name="DiagKFACMuonOptimizer", module_name="diag_kfac_muon", requires_model=True, constructor_takes_model=True),
    "DiagHadron": OptimizerMetadata(cls_name="DiagHadron", module_name="diag_hadron", requires_model=True, expects_param_groups=True),
    "RMSuon": OptimizerMetadata(cls_name="RMSuon", module_name="rmsuon", expects_param_groups=True),
    "AdaSuon": OptimizerMetadata(cls_name="AdaSuon", module_name="adasuon", expects_param_groups=True),
    "LazyRMSuon": OptimizerMetadata(cls_name="LazyRMSuon", module_name="lazy_rmsuon", expects_param_groups=True),
    "AdaRMSuon": OptimizerMetadata(cls_name="AdaRMSuon", module_name="ada_rmsuon", expects_param_groups=True),
    "AdaMuon": OptimizerMetadata(cls_name="AdaMuon", module_name="ada_muon", expects_param_groups=True),
    "KFACRMSuon": OptimizerMetadata(cls_name="KFACRMSuon", module_name="kfac_rmsuon", expects_param_groups=True, requires_model=True),
}

def _import_optimizer(module_name: str, class_name: str) -> type[torch.optim.Optimizer]:
    if module_name == "torch.optim":
        return getattr(torch.optim, class_name)
    module = import_module(f".{module_name}", package="optimizer")
    return getattr(module, class_name)

def _create_specialized_param_groups(
    params: list[torch.nn.Parameter],
    optimizer_name: str,
    config: dict
) -> list[dict]:
    """
    Automatically splits parameters into groups for specialized optimizers
    like Muon, RMSuon, etc., and a default AdamW group for the rest.
    """
    special_params = []
    adamw_params = []

    # Define the condition for a parameter to be handled by the special optimizer
    if optimizer_name in ["Muon", "RMSuon", "AdaSuon", "LazyRMSuon", "AdaRMSuon", "AdaMuon", "KFACRMSuon"]:
        is_special_param = lambda p: p.ndim >= 2 and max(p.shape) < 10000
        if optimizer_name == "KFACRMSuon":
            special_group_flag = 'use_kfac_rmsuon'
        elif 'RMSuon' in optimizer_name or 'AdaSuon' in optimizer_name or 'AdaMuon' in optimizer_name:
            special_group_flag = 'is_rmsuon_group'
        else:
            special_group_flag = 'use_muon'
    else:
        # Extend with other optimizer conditions if needed
        return [{'params': params}]

    for p in params:
        if p.requires_grad:
            if is_special_param(p):
                special_params.append(p)
            else:
                adamw_params.append(p)

    param_groups = []
    if special_params:
        special_config = {
            'params': special_params,
            special_group_flag: True,
            'lr': config.get("lr", 1e-3),
            'weight_decay': config.get("weight_decay", 0.1),
        }
        if optimizer_name == "Muon":
            special_config['momentum'] = config.get("momentum", 0.95)
        elif "RMSuon" in optimizer_name or "AdaSuon" in optimizer_name or "AdaRMSuon" in optimizer_name or "AdaMuon" in optimizer_name:
            special_config['betas'] = config.get("betas", (0.9, 0.999))
            special_config['eps'] = config.get("eps", 1e-8)
        elif optimizer_name == "KFACRMSuon":
            special_config['stat_decay'] = config.get("stat_decay", 0.95)
            special_config['damping'] = config.get("damping", 0.001)
            special_config['TCov'] = config.get("TCov", 10)
            special_config['ns_steps'] = config.get("ns_steps", 5)
            special_config['ns_steps'] = config.get("ns_steps", 5)
            if optimizer_name == "LazyRMSuon":
                special_config['energy_sync_every'] = config.get("energy_sync_every", 10)

        param_groups.append(special_config)

    if adamw_params:
        adam_config = {
            'params': adamw_params,
            special_group_flag: False,
            'lr': config.get("adam_lr", config.get("lr", 1e-3)),
            'betas': config.get("adam_betas", (0.9, 0.999)),
            'eps': config.get("adam_eps", 1e-8),
            'weight_decay': config.get("adam_weight_decay", 0.01),
        }
        param_groups.append(adam_config)

    return param_groups


def get_optimizer(name: str, params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, dict | None]:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    meta = OPTIMIZER_REGISTRY[name]
    OptimizerClass = _import_optimizer(meta.module_name, meta.cls_name)

    opt_config = config.copy()
    tags = {
        "requires_second_order": meta.requires_second_order,
        "requires_loss_for_step": meta.requires_loss_for_step,
        "accepts_pi_signal": name in ["PI_ZPD", "FIENA_FOG"],
    }

    # If optimizer expects param groups, we might need to auto-create them
    if meta.expects_param_groups and name in ["Muon", "RMSuon", "AdaSuon", "LazyRMSuon", "AdaRMSuon", "AdaMuon", "KFACRMSuon"]:
        # Flatten the initial param list
        all_params = [p for group in params for p in group['params']]
        init_params = _create_specialized_param_groups(all_params, name, opt_config)
    elif meta.expects_param_groups:
        init_params = params
    else:
        init_params = next(iter(params))['params']


    if meta.requires_model and "model" not in opt_config:
        raise ValueError(f"Optimizer '{name}' requires a 'model' instance.")

    if meta.constructor_takes_model:
        model = opt_config.pop('model')
        optimizer = OptimizerClass(model, **opt_config)
    else:
        if not meta.requires_model:
            opt_config.pop('model', None)

        # Clean up config for optimizers that get structured groups
        if meta.expects_param_groups and name in ["Muon", "RMSuon", "AdaSuon", "LazyRMSuon", "AdaRMSuon", "AdaMuon", "KFACRMSuon"]:
             if name == "KFACRMSuon":
                 optimizer = OptimizerClass(init_params, model=opt_config.pop('model'), **opt_config)
             else:
                 optimizer = OptimizerClass(init_params)
        else:
             optimizer = OptimizerClass(init_params, **opt_config)


    pi_config = None
    if tags["accepts_pi_signal"]:
        pi_keys = ["gamma", "ema_beta", "alpha"]
        pi_config = {k: config[k] for k in pi_keys if k in config}

    return optimizer, tags, pi_config
