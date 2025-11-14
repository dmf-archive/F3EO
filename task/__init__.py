# 任务模块初始化
from .cifar10 import Cifar10Task
from .wikitext2 import Wikitext2Task
from .mnist_cl import MnistClTask
from .fashion_cl import FashionClTask

__all__ = ['Cifar10Task', 'Wikitext2Task', 'MnistClTask', 'FashionClTask']