from dataclasses import dataclass

import torch

from .ada_hessian import Adahessian
from .diag_fog import DiagFOG
from .diag_kfac import DiagKFACOptimizer
from .diag_kfac_muon import DiagKFACMuonOptimizer
from .fiena_fog import FIENA_FOG
from .fog import FOG
from .kfac import KFACOptimizer
from .muon import SingleDeviceMuon
from .pi_zpd import PI_ZPD


@dataclass
class OptimizerMetadata:
    cls: type[torch.optim.Optimizer]
    requires_model: bool = False
    requires_second_order: bool = False
    expects_param_groups: bool = False


OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata] = {
    "AdamW": OptimizerMetadata(cls=torch.optim.AdamW),
    "AdaHessian": OptimizerMetadata(cls=Adahessian, requires_second_order=True),
    "Muon": OptimizerMetadata(cls=SingleDeviceMuon),
    "KFAC": OptimizerMetadata(cls=KFACOptimizer, requires_model=True),
    "FOG": OptimizerMetadata(cls=FOG, requires_model=True),
    "DiagKFAC": OptimizerMetadata(cls=DiagKFACOptimizer, requires_model=True),
    "DiagKFACMuon": OptimizerMetadata(cls=DiagKFACMuonOptimizer, requires_model=True),
    "DiagFOG": OptimizerMetadata(cls=DiagFOG, requires_model=True, expects_param_groups=True),
    "PI_ZPD": OptimizerMetadata(cls=PI_ZPD, requires_model=True, expects_param_groups=True),
    "FIENA_FOG": OptimizerMetadata(cls=FIENA_FOG, requires_model=True, requires_second_order=True),
}


def get_optimizer(name: str, params, **config):
    """
    统一优化器工厂，返回 (optimizer, tags, pi_config)
    - optimizer: 优化器实例
    - tags: 优化器能力标签 (dict)
    - pi_config: PI 计算相关配置 (dict or None)
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    meta = OPTIMIZER_REGISTRY[name]
    tags = {
        "requires_second_order": meta.requires_second_order,
        "accepts_pi_signal": name in ["PI_ZPD", "FIENA_FOG"],
    }

    opt_config = config.copy()

    if meta.requires_model:
        if "model" not in opt_config:
            raise ValueError(f"{name} optimizer requires 'model' parameter in config")
        model = opt_config.pop("model")

        if meta.requires_model:
            if "model" not in config:
                raise ValueError(f"{name} optimizer requires 'model' parameter in config")

            # Pass model and params correctly
            if meta.expects_param_groups:
                 opt = meta.cls(params, **opt_config)
            else:
                 opt = meta.cls(model, **opt_config)

        else:
            opt = meta.cls(params, **opt_config)
    else:
        opt = meta.cls(params, **opt_config)

    pi_config = None
    if tags["accepts_pi_signal"]:
        pi_config = {k: opt_config.pop(k) for k in list(opt_config.keys()) if k in ["gamma", "ema_beta", "alpha"]}

    return opt, tags, pi_config
