def get_optimizer(name: str, params, **config):
    """
    统一优化器工厂，返回 (optimizer, tags, pi_config)
    - optimizer: 优化器实例
    - tags: 优化器能力标签 (dict)
    - pi_config: PI 计算相关配置 (dict or None)
    """
    tags = {
        "requires_second_order": False,
        "accepts_pi_signal": False,
    }
    pi_config = None

    if name == "AdamW":
        import torch
        opt = torch.optim.AdamW(params, **config)

    elif name == "AdaHessian":
        from .ada_hessian import Adahessian
        opt = Adahessian(params, **config)
        tags["requires_second_order"] = True

    elif name == "AdaFisher":
        from .ada_fisher import AdaFisher
        if "model" not in config:
            raise ValueError("AdaFisher optimizer requires 'model' parameter in config")
        model = config.pop("model")
        opt = AdaFisher(model, **config)
        tags["requires_second_order"] = True

    # PI 感知型优化器
    elif name in ["F3EPI", "AdamW_PI", "F3EWD"]:
        pi_config = {k: config.pop(k) for k in list(config.keys()) if k in ["gamma", "ema_beta", "alpha"]}
        tags["accepts_pi_signal"] = True

        if name == "F3EPI":
            from .F3EPI import F3EPI
            opt = F3EPI(params, **config)
            tags["requires_second_order"] = True
        elif name == "AdamW_PI":
            from .adamw_pi import AdamW_PI
            opt = AdamW_PI(params, **config)
        elif name == "F3EWD":
            from .F3EWD import F3EWD
            opt = F3EWD(params, **config)
            tags["requires_second_order"] = True

    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return opt, tags, pi_config
