from .nano_gpt import MiniGPT1 as nano_gpt
from .resnet import resnet18
from .resnet_cifar import ResNet18 as resnet18_cifar
from .resnet_mnist import ResNet18_mnist
from .swin import swin_t


def get_model(name: str, num_classes: int = 10, **kwargs):
    return (
        resnet18(num_classes=num_classes) if name == "resnet18" else
        resnet18_cifar(num_classes=num_classes) if name == "resnet18_cifar" else
        ResNet18_mnist(num_classes=num_classes) if name == "resnet18_mnist" else
        swin_t(num_classes=num_classes, **kwargs) if name == "swin_t" else
        nano_gpt(vocabulary_size=num_classes, **kwargs) if name == "nano_gpt" else
        None
    )
