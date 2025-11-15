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

---

## **重构计划 v2.0：高内聚、低耦合的训练管线**

**目标**: 彻底解耦优化器配置、训练逻辑和可观测性，建立一个清晰、可扩展、易于维护的实验框架。

### **第一阶段：优化器注册与配置革命 (WP-OPT)**

此阶段旨在根除硬编码，创建一个灵活、声明式的优化器系统。

1. **任务 1.1: 实现基于注册表的优化器工厂**

   - **问题**: `optimizer/__init__.py` 中冗长的 `if/elif` 链难以维护和扩展。
   - **方案**:
     1. 在 `optimizer/__init__.py` 中创建一个名为 `OPTIMIZER_REGISTRY` 的全局字典。
     2. 为每个优化器创建一个 `dataclass` 或 `dict` 用于存储其元数据，例如：`cls` (类本身), `requires_model: bool`, `supports_param_groups: bool`。
     3. 使用装饰器模式，在每个优化器文件（如 `optimizer/diag_fog.py`）的顶部直接将其注册到 `OPTIMIZER_REGISTRY` 中。
     4. 重构 `get_optimizer` 函数，使其通过查阅 `OPTIMIZER_REGISTRY` 来动态实例化优化器并获取其能力标签，彻底消除 `if/elif` 结构。
   - **涉及文件**: `optimizer/__init__.py` (核心重构), 所有 `optimizer/*.py` 文件。

2. **任务 1.2: 参数分组责任转移**
   - **问题**: `scripts/train.py` 中存在针对特定模型的参数分组逻辑，造成了不必要的耦合。
   - **方案**:
     1. 在 `task/base.py` 中定义一个新的抽象方法 `get_param_groups(self, model) -> list[dict]`。
     2. 在具体的 `Task` 类（如 `task/wikitext2.py`）中实现此方法，根据传入的 `model` 实例，返回为该模型量身定制的参数分组列表（例如，区分隐藏层、嵌入层等）。
     3. 重构 `scripts/train.py` 中的 `create_optimizer` 逻辑，使其调用当前 `task` 实例的 `get_param_groups` 方法来获取参数，而不是直接使用 `model.parameters()`。
   - **涉及文件**: `task/base.py`, `task/wikitext2.py`, `task/cifar10.py`, `scripts/train.py`。

### **第二阶段：训练逻辑与可观测性分离 (WP-OBS)**

此阶段旨在将“如何训练”与“如何观察训练”彻底分离。

1. **任务 2.1: 引入回调式（Callback）可观测系统**

   - **问题**: 日志、检查点等观测逻辑散布在主训练循环中，难以增删或替换。
   - **方案**:
     1. 创建一个 `Callback` 基类 (`utils/callbacks/base.py`)，定义 `on_train_begin`, `on_epoch_end`, `on_step_end` 等钩子（hooks）。
     2. 将 `utils/observers` 目录重命名为 `utils/callbacks`，并重构其中的所有类（`ConsoleLogger`, `MDLogger`, `CheckpointSaver`），使其继承自 `Callback` 基类。
   - **涉及文件**: `utils/observers/*` -> `utils/callbacks/*` (重构), `utils/callbacks/base.py` (新文件)。

2. **任务 2.2: 抽象 `Trainer` 核心**
   - **问题**: `scripts/train.py` 文件过于臃肿，承担了所有职责。
   - **方案**:
     1. 创建一个新的 `Trainer` 类 (`utils/trainer.py`)。其构造函数接收 `model`, `optimizer`, `criterion`, 数据加载器, 以及一个 `callbacks` 列表。
     2. 将完整的训练/验证循环逻辑封装到 `Trainer.fit()` 方法中。在循环的各个关键节点（如 epoch 开始、step 结束），`Trainer` 将遍历并调用所有注册回调的相应钩子方法。
     3. `scripts/train.py` 的职责被极大简化，仅负责组装所有组件（模型、任务、优化器、回调）并将其传递给 `Trainer`，最后调用 `trainer.fit()`。
   - **涉及文件**: `utils/trainer.py` (新文件), `scripts/train.py` (大幅简化)。

---

此计划将通过 `Orchestrator` 模式分阶段执行。
