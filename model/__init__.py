def get_model(name: str, num_classes: int = 10, **kwargs):
    """按需延迟加载模型，避免不必要的依赖"""
    if name == "resnet18":
        from .resnet import resnet18
        return resnet18(num_classes=num_classes)
    elif name == "resnet18_cifar":
        from .resnet_cifar import ResNet18 as resnet18_cifar
        return resnet18_cifar(num_classes=num_classes)
    elif name == "resnet18_mnist":
        from .resnet_mnist import ResNet18_mnist
        return ResNet18_mnist(num_classes=num_classes)
    elif name == "swin_t":
        from .swin import swin_t
        return swin_t(num_classes=num_classes, **kwargs)
    elif name == "nano_gpt":
        from .nano_gpt import MiniGPT1 as nano_gpt
        return nano_gpt(vocabulary_size=num_classes, **kwargs)
    else:
        return None
