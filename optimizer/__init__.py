from dataclasses import dataclass, field
from importlib import import_module
from enum import Enum, auto
import torch

class GroupingStrategy(Enum):
    NONE = auto()
    MUON = auto()    # 使用 'use_muon' 标志位
    RMSUON = auto()  # 使用 'is_rmsuon_group' 标志位

@dataclass
class OptimizerMetadata:
    cls_name: str
    module_name: str
    grouping: GroupingStrategy = GroupingStrategy.NONE
    expects_param_groups: bool = False
    # 对应 Trainer 中的 tags 契约
    d_0_requires_model_instance: bool = False
    d_1_step_takes_closure: bool = False
    d_1_step_requires_loss_tensor: bool = False
    d_2_requires_second_order: bool = False
    d_2_requires_bn_protection: bool = False
    d_2_backward_handles_itself: bool = False
    pi_aware: bool = False
    # 专用参数组需要从全局 config 继承的额外键
    extra_config_keys: list[str] = field(default_factory=list)

OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata] = {
    "AdamW": OptimizerMetadata(
        cls_name="AdamW", module_name="torch.optim", 
        expects_param_groups=True
    ),
    "AdaHessian": OptimizerMetadata(
        cls_name="Adahessian", module_name="ada_hessian", 
        d_2_requires_second_order=True
    ),
    "Muon": OptimizerMetadata(
        cls_name="SingleDeviceMuonWithAuxAdam", module_name="muon",
        grouping=GroupingStrategy.MUON, expects_param_groups=True,
        extra_config_keys=["momentum", "betas", "eps", "ns_steps"]
    ),
    "RMSuon": OptimizerMetadata(
        cls_name="RMSuon", module_name="rmsuon", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        extra_config_keys=["betas", "eps"]
    ),
    "AdaRMSuon": OptimizerMetadata(
        cls_name="AdaRMSuon", module_name="ada_rmsuon", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        extra_config_keys=["betas", "eps"]
    ),
    "ARS": OptimizerMetadata(
        cls_name="ARSOptimizer", module_name="ars", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        d_1_step_takes_closure=True, d_2_requires_bn_protection=True,
        extra_config_keys=["betas", "eps", "rho", "k", "alpha"]
    ),
    "ARG": OptimizerMetadata(
        cls_name="ARGOptimizer", module_name="arg", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        d_1_step_takes_closure=True, d_2_requires_second_order=True, d_2_requires_bn_protection=True,
        extra_config_keys=["betas", "eps", "rho"]
    ),
    "KFAC": OptimizerMetadata(
        cls_name="KFACOptimizer", module_name="kfac", 
        d_0_requires_model_instance=True
    ),
    "DiagHadron": OptimizerMetadata(
        cls_name="DiagHadron", module_name="diag_hadron", 
        d_0_requires_model_instance=True, expects_param_groups=True
    ),
    "LARS": OptimizerMetadata(
        cls_name="LARSOptimizer", module_name="lars",
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        d_1_step_takes_closure=True, d_2_requires_bn_protection=True,
        extra_config_keys=["betas", "eps", "rho", "k", "alpha", "adaptive_alpha"]
    ),
    "ARS2-Neo": OptimizerMetadata(
        cls_name="SingleDeviceARS2Neo", module_name="ars2_neo",
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        d_1_step_takes_closure=True, d_2_requires_bn_protection=True,
        extra_config_keys=["betas", "eps", "rho", "k", "alpha", "adaptive", "ns_steps"]
    ),
}

def _import_optimizer(module_name: str, class_name: str) -> type[torch.optim.Optimizer]:
    if module_name == "torch.optim":
        return getattr(torch.optim, class_name)
    module = import_module(f".{module_name}", package="optimizer")
    return getattr(module, class_name)

def _create_specialized_param_groups(params: list[torch.nn.Parameter], meta: OptimizerMetadata, config: dict) -> list[dict]:
    is_special = lambda p: p.ndim >= 2 and max(p.shape) < 10000
    flag_name = "use_muon" if meta.grouping == GroupingStrategy.MUON else "is_rmsuon_group"
    
    special_params = [p for p in params if p.requires_grad and is_special(p)]
    adam_params = [p for p in params if p.requires_grad and not is_special(p)]
    
    groups = []
    if special_params:
        grp = {
            'params': special_params,
            flag_name: True,
            'lr': config.get("lr", 1e-3),
            'weight_decay': config.get("weight_decay", 0.1),
        }
        for k in meta.extra_config_keys:
            if k in config: grp[k] = config[k]
        groups.append(grp)
        
    if adam_params:
        groups.append({
            'params': adam_params,
            flag_name: False,
            'lr': config.get("adam_lr", config.get("lr", 1e-3)),
            'betas': config.get("adam_betas", (0.9, 0.999)),
            'eps': config.get("adam_eps", 1e-8),
            'weight_decay': config.get("adam_weight_decay", 0.01),
        })
    return groups

def get_optimizer(name: str, params: list[dict], **config) -> tuple[torch.optim.Optimizer, dict, dict | None]:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    meta = OPTIMIZER_REGISTRY[name]
    opt_cls = _import_optimizer(meta.module_name, meta.cls_name)
    
    tags = {
        "d_0_requires_model_instance": meta.d_0_requires_model_instance,
        "d_1_step_takes_closure": meta.d_1_step_takes_closure,
        "d_1_step_requires_loss_tensor": meta.d_1_step_requires_loss_tensor,
        "d_2_requires_second_order": meta.d_2_requires_second_order,
        "d_2_requires_bn_protection": meta.d_2_requires_bn_protection,
        "d_2_backward_handles_itself": meta.d_2_backward_handles_itself,
        "accepts_pi_signal": meta.pi_aware or name in ["PI_ZPD", "FIENA_FOG"],
    }

    # 参数初始化逻辑还原
    opt_config = config.copy()
    if meta.grouping != GroupingStrategy.NONE:
        flag_name = "use_muon" if meta.grouping == GroupingStrategy.MUON else "is_rmsuon_group"
        # 如果已经手动分好组了，则尊重手动分组
        if any(flag_name in g for g in params):
            init_params = params
        else:
            all_params = [p for g in params for p in g['params']]
            init_params = _create_specialized_param_groups(all_params, meta, opt_config)
        
        for key in ["adam_lr", "adam_betas", "adam_eps", "adam_weight_decay"]:
            opt_config.pop(key, None)
    elif meta.expects_param_groups:
        init_params = params
    else:
        init_params = next(iter(params))['params']
    
    model_instance = opt_config.pop('model', None)

    if meta.d_0_requires_model_instance:
        if model_instance is None: raise ValueError(f"Optimizer '{name}' requires a 'model' instance.")
        optimizer = opt_cls(model_instance, **opt_config)
    else:
        optimizer = opt_cls(init_params, **opt_config)

    pi_config = {k: config[k] for k in ["gamma", "ema_beta", "alpha"] if k in config} if tags["accepts_pi_signal"] else None
    return optimizer, tags, pi_config
