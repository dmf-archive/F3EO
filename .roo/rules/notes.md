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
