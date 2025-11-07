# AdaFisher 论文基线性能数据

本文件记录了从 AdaFisher 论文 (arXiv:2405.16397) 中提取的关键优化器性能基线，以便于未来参考。

## CIFAR-10 / ResNet-18

实验设置：

- **批量大小**: 256
- **训练周期**: ~200 epochs (以 AdaFisher 的训练时间为准)
- **指标**: 最终验证准确率 (%)

| Optimizer  | Accuracy (%) |
| ---------- | ------------ |
| Adam       | 94.85 ± 0.1  |
| AdaHessian | 95.44 ± 0.1  |
| AdaFisher  | 96.25 ± 0.2  |

_数据来源: 论文 Table 2_

## Wikitext-2 / GPT-1 (small)

实验设置：

- **批量大小**: 32
- **训练周期**: ~50 epochs (以 AdaFisher 的训练时间为准)
- **指标**: 最终验证集困惑度 (Perplexity)

| Optimizer  | Perplexity |
| ---------- | ---------- |
| AdamW      | 100.93     |
| AdaHessian | 118.01     |
| AdaFisherW | 89.26      |

_数据来源: 论文 Table 5_

## 关键发现：AdaFisher 论文的实验配置问题

通过对 AdaFisher 原始代码库 (`ref/Adafisher/`) 的严格分析，我们发现了其论文结果可复现性的一个根本性问题：

### 实现与调用的不匹配

**问题核心**: 在 `ref/Adafisher/Language_Model/run_exp.py` 中，AdaFisherW 优化器被实例化为：

```python
optimizer = AdaFisherW(model, lr=args.lr, gammas=[args.gamma1, args.gamma2], ...)
```

然而，其 `AdaFisherBackBone` 基类的构造函数签名却是：

```python
def __init__(self, ..., gamma: float = 0.8, ...)
```

**后果**: 意图传递的列表参数 `gammas=[0.92, 0.008]` 被忽略，优化器回退到使用默认的 `gamma = 0.8`。这意味着论文声称使用了 `gamma2=0.008` 这一极低值，但实际运行中并未生效。

**影响**: 如果 `gamma` 真的被设置为 `0.008`，其指数移动平均（EMA）更新公式 `current *= gamma * 1e-1; current += (gamma * 1e-2) * new` 将导致 Kronecker 因子估计剧烈波动，极有可能使训练发散。论文结果与实际运行配置不符，存在严重的可复现性问题。

### 当前实验状态

**AdamW 基线 (进行中)**:

- **配置**: `lr=1e-5`, `weight_decay=0.1`, 60 epochs
- **当前进度**: Epoch 6, Valid PPL ~5M
- **观察**: 学习率极低，收敛极其缓慢，远未达到 AdaFisher 论文中 AdamW 的 100.93 PPL 基线。

**下一步**: 待 AdamW 实验完成后，将启动使用我们推导出的**真实**超参数（`gamma=0.8`）的 AdaFisherW 对照组实验，以建立可靠的性能基线。

## 待办笔记

- **优化器输出目录命名**: 当前 [`scripts/train.py`](scripts/train.py) 和 [`utils/training_monitor.py`](utils/training_monitor.py) 使用优化器名称而非实验配置名称来创建输出文件夹（用于保存检查点和摘要）。这极不利于调参和实验管理，必须修复为使用实验配置名称。
