from dataclasses import dataclass
from importlib import import_module

import torch


@dataclass
class OptimizerMetadata:
    cls_name: str
    module_name: str
    expects_param_groups: bool = False
    # Constructor Dependencies
    d_0_requires_model_instance: bool = False
    # Step Method Signature
    d_1_step_requires_closure_eval_loss: bool = False
    d_1_step_requires_closure_eval_logits: bool = False
    d_1_step_requires_loss_tensor: bool = False
    # Gradient Calculation Control
    d_2_backward_handles_itself: bool = False
    d_2_backward_requires_second_order: bool = False


OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata] = {
    "AdamW": OptimizerMetadata(cls_name="AdamW", module_name="torch.optim", expects_param_groups=True),
    "AdaHessian": OptimizerMetadata(cls_name="Adahessian", module_name="ada_hessian", d_2_backward_requires_second_order=True),
    "Muon": OptimizerMetadata(cls_name="SingleDeviceMuonWithAuxAdam", module_name="muon", expects_param_groups=True),
    "KFAC": OptimizerMetadata(cls_name="KFACOptimizer", module_name="kfac", d_0_requires_model_instance=True),
    "Hadron": OptimizerMetadata(cls_name="Hadron", module_name="hadron", d_0_requires_model_instance=True),
    "DiagKFAC": OptimizerMetadata(cls_name="DiagKFACOptimizer", module_name="diag_kfac", d_0_requires_model_instance=True, expects_param_groups=True),
    "DiagHadron": OptimizerMetadata(cls_name="DiagHadron", module_name="diag_hadron", d_0_requires_model_instance=True, expects_param_groups=True),
    "RMSuon": OptimizerMetadata(cls_name="RMSuon", module_name="rmsuon", expects_param_groups=True),
    "AdaSuon": OptimizerMetadata(cls_name="AdaSuon", module_name="adasuon", expects_param_groups=True),
    "AdaRMSuon": OptimizerMetadata(cls_name="AdaRMSuon", module_name="ada_rmsuon", expects_param_groups=True),
    "FS_AdaRMSuon": OptimizerMetadata(cls_name="FS_AdaRMSuon", module_name="fs_ada_rmsuon", expects_param_groups=True, d_1_step_requires_closure_eval_loss=True),
    "SAF_RMSuon": OptimizerMetadata(cls_name="SAF_RMSuon", module_name="saf_rmsuon", expects_param_groups=True, d_1_step_requires_closure_eval_logits=True, d_2_backward_requires_second_order=True),
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
    special_params = []
    adamw_params = []

    if optimizer_name in ["Muon", "RMSuon", "AdaSuon", "LazyRMSuon", "AdaRMSuon", "IG_AdaRMSuon", "FS_AdaRMSuon", "SAF_RMSuon"]:
        is_special_param = lambda p: p.ndim >= 2 and max(p.shape) < 10000
        if 'RMSuon' in optimizer_name or 'AdaSuon' in optimizer_name or 'IG_AdaRMSuon' in optimizer_name or 'SAF_RMSuon' in optimizer_name:
            special_group_flag = 'is_rmsuon_group'
        else:
            special_group_flag = 'use_muon'
    else:
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
        elif "RMSuon" in optimizer_name or "AdaSuon" in optimizer_name or "AdaRMSuon" in optimizer_name or "IG_AdaRMSuon" in optimizer_name or "FS_AdaRMSuon" in optimizer_name or "SAF_RMSuon" in optimizer_name:
            special_config['betas'] = config.get("betas", (0.9, 0.999))
            special_config['eps'] = config.get("eps", 1e-8)
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
        "d_0_requires_model_instance": meta.d_0_requires_model_instance,
        "d_1_step_requires_closure_eval_loss": meta.d_1_step_requires_closure_eval_loss,
        "d_1_step_requires_closure_eval_logits": meta.d_1_step_requires_closure_eval_logits,
        "d_1_step_requires_loss_tensor": meta.d_1_step_requires_loss_tensor,
        "d_2_backward_handles_itself": meta.d_2_backward_handles_itself,
        "d_2_backward_requires_second_order": meta.d_2_backward_requires_second_order,
        "accepts_pi_signal": name in ["PI_ZPD", "FIENA_FOG"],
    }

    if meta.expects_param_groups and name in ["Muon", "RMSuon", "AdaSuon", "AdaRMSuon", "FS_AdaRMSuon", "SAF_RMSuon"]:
        all_params = [p for group in params for p in group['params']]
        init_params = _create_specialized_param_groups(all_params, name, opt_config)
    elif meta.expects_param_groups:
        init_params = params
    else:
        init_params = next(iter(params))['params']


    if meta.d_0_requires_model_instance and "model" not in opt_config:
        raise ValueError(f"Optimizer '{name}' requires a 'model' instance.")

    # KFAC is a special case where it takes the model directly.
    if name in ["KFAC", "Hadron", "DiagKFAC", "DiagHadron"]:
        model = opt_config.pop('model')
        optimizer = OptimizerClass(model, **opt_config)
    else:
        if meta.d_0_requires_model_instance:
             # For other optimizers that might need the model but not in constructor
            opt_config.pop('model', None)

        if meta.expects_param_groups and name in ["Muon", "RMSuon", "AdaSuon", "AdaRMSuon", "FS_AdaRMSuon", "SAF_RMSuon"]:
            optimizer = OptimizerClass(init_params, **opt_config)
        else:
            optimizer = OptimizerClass(init_params, **opt_config)


    pi_config = None
    if tags["accepts_pi_signal"]:
        pi_keys = ["gamma", "ema_beta", "alpha"]
        pi_config = {k: config[k] for k in pi_keys if k in config}

    return optimizer, tags, pi_config
