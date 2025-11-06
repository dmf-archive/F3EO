def get_optimizer(name: str, params, **config):
    """
    统一优化器工厂，返回 (optimizer, tags_dict)
    tags_dict 提供能力标签，供任务脚本通用处理：
      - requires_second_order : bool  是否需要 create_graph=True
      - passes_loss_to_step   : bool  是否将 loss 张量传给 step(...)
    """
    tags = {
        "requires_second_order": False,
        "passes_loss_to_step": False,
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

    # F3E 系列：全部需要二阶梯度，部分需要传 loss
    elif name == "F3EO":
        from .F3EO import F3EO
        opt = F3EO(params, **config)
        tags["requires_second_order"] = True

    elif name == "F3EL":
        from .F3EL import F3EL
        opt = F3EL(params, **config)
        tags["requires_second_order"] = True
        tags["passes_loss_to_step"] = True

    elif name == "F3EW":
        from .F3EW import F3EW
        opt = F3EW(params, **config)
        tags["requires_second_order"] = True

    elif name == "F3EPI":
        from .F3EPI import F3EPI
        opt = F3EPI(params, **config)
        tags["requires_second_order"] = True
        tags["passes_loss_to_step"] = True

    elif name == "F3EPIRMS":
        from .F3EPI_RMS import F3EPIRMS
        opt = F3EPIRMS(params, **config)
        tags["requires_second_order"] = True
        tags["passes_loss_to_step"] = True

    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return opt, tags
