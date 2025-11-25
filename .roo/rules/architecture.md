# F3EO-Bench: 轻量级三阶优化器评测框架

```ascii
f3eo-bench/
├── README.md      # 30 秒上手命令
├── pyproject.toml # 只留 torch rich tqdm
├── optimizer/
│   ├── __init__.py
│   ├── f3eo.py    # 三阶核心，<150 行
│   └── adahessian.py  # 二阶对照，直接抄官方
├── model/
│   ├── __init__.py
│   ├── resnet.py # ResNet-18
│   ├── vit.py   # Transformer
│   └── nano-gpt.py   # MNIST/Fashion 头
├── data/ # 自动创建的数据集缓存
├── config/
│   ├── cifar10.toml
│   ├── wikitext2.toml
│   └── cl_stream.toml
├── task/ # 不同任务具体的训练调度器
│   ├── cifar10.py
│   ├── wikitext2.py
│   └── cl_stream.py   # 持续学习 MNIST→Fashion
├── outputs/# gitignored，自动生成
│   ├── report/      # markdown report
│   └── checkpoints/   # 只存 best.pt
└──  scripts/
   ├── train.py   # 统一入口、rich log print
   └── notebook/
      └── loss_landscape.ipynb  # 损失地形可视化（参考 adafisher）
```

## 实验流水线（多种配置，一条命令）

```bash
python -m scripts/train.py --config config/cifar10.toml
```

## 挂载新优化器流程

为保证框架的可扩展性，添加一个新的优化器需要遵循以下三个步骤。这个流程确保了优化器能够被正确地实例化、配置，并与需要模型实例的二阶方法（如 KFAC）兼容。

1. **创建优化器实现**: 在 `optimizer/` 目录下创建一个新的 Python 文件（例如 `my_optimizer.py`），并在其中实现你的优化器类。

2. **在工厂函数中注册**: 打开 [`optimizer/__init__.py`](optimizer/__init__.py)，在 `get_optimizer` 工厂函数中，为你的新优化器添加一个 `elif` 分支，用于导入和实例化它。
