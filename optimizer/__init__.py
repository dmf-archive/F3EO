def get_optimizer(name: str, params, **config):
    """
    统一优化器工厂，返回 (optimizer, tags_dict)
    tags_dict 提供能力标签，供任务脚本通用处理：
      - requires_second_order : bool  是否需要 create_graph=True
      - passes_loss_to_step   : bool  是否将 loss 张量传给 step(...)
    """
    tags = {
        "requires_second_order": False,
        "accepts_pi_signal": False,
    }

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

    # F3E 系列：全部需要二阶梯度
    elif name == "F3EPI":
        from .F3EPI import F3EPI
        opt = F3EPI(params, **config)
        tags["requires_second_order"] = True
        tags["accepts_pi_signal"] = True # F3EPI is now PI-aware

    elif name == "AdamW_PI":
        from .adamw_pi import AdamW_PI
        opt = AdamW_PI(params, **config)
        tags["accepts_pi_signal"] = True

    elif name == "F3EWD":
        from .F3EWD import F3EWD
        opt = F3EWD(params, **config)
        tags["requires_second_order"] = True
        tags["accepts_pi_signal"] = True

    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return opt, tags
