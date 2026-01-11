# F3EO-Bench: 轻量级先进优化器评估框架

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/F3EO)

> **F3EO-Bench** 是一个用于原型设计和评估神经网络优化器的研究框架，核心理念是 **能量-几何解耦 (Energy-Geometry Decoupling)**。

## 1. 核心理念：能量-几何解耦

现代深度学习优化面临着统计自适应性（学多快）和结构稳定性（学什么）之间的权衡。F3EO-Bench 致力于研究解耦这两个关注点的优化器。我们的旗舰优化器 **AdaRMSuon** 将这一原则付诸实践：

1. **统计算子 (能量)**: 使用 AdamW 的二阶矩修正动量的标量范数 (‖m̂ / √v̂‖) 来确定更新幅度。这充当了自由能下降速率的计算廉价代理。
2. **结构算子 (几何)**: 采用 Muon 的 Newton-Schulz 迭代来寻找正交更新方向 (O_t)。**AdaRMSuon** 进一步引入了 **预白化 (Pre-whitening)** 技术，在投影前使用 v_t 对梯度进行白化，确保更新沿着黎曼流形的测地线进行。

最终更新是这两个算子的复合：`g_update = scale * O_t`。这解耦了“多快”与“去哪”，提供了一条通往极小值的鲁棒且高效的路径。

## 2. 进阶演化：从几何到拓扑 (ARS)

虽然 AdaRMSuon 收敛极快，但它倾向于陷入尖锐的局部极小值（过拟合）。为了解决这个问题，我们引入了 **ARS (AdaRMSuon Regularized Search)**，它在几何滑行的基础上增加了拓扑层面的平坦度约束。

- **流形感知 SAM**: ARS 不在欧氏空间进行球形扰动，而是在由 v_t 定义的黎曼流形上计算对抗方向，寻找“平坦”的测地线区域。
- **Lazy Mode 与剪切力注入**: 为了避免 SAM 带来的双倍计算开销，我们实现了 Lazy Mode (k > 1)。在非扰动步骤中，我们注入一个与基础梯度正交的“剪切力” (v_flat)，持续推动模型离开尖锐区域。
- **强度补偿机制**: 实验表明，在 Lazy Mode 下，必须通过增大注入强度 (α) 来补偿低频修正带来的偏差。配置 k=5, α=0.1 实现了训练速度与泛化性能的最佳平衡。

## 3. 关键实验结果

### 3.1 Wikitext-2 语言建模

我们在 Wikitext-2 (`line mode`) 上验证了这些优化器。该模式保留了句子边界，最大化了输入的上下文完整性。

| 优化器         | 核心机制                     | Epoch 1 PPL | Best PPL  | Final PPL | 说明                         |
| :------------- | :--------------------------- | :---------- | :-------- | :-------- | :--------------------------- |
| Muon           | SGD + Newton-Schulz          | 233.30      | 161.09    | 161.09    | 基线                         |
| RMSuon         | Adam + Newton-Schulz         | 146.52      | 99.07     | 134.11    | 早期版本                     |
| AdaRMSuon      | Pre-white + Energy           | **133.68**  | 83.88     | 87.61     | 收敛最快，但过拟合           |
| ARS (Sync)     | Manifold SAM (ρ=0.05)        | 156.62      | 83.70     | 83.70     | 泛化好                       |
| **ARS (Sync)** | **Manifold SAM (ρ=0.1)**     | 159.13      | **80.94** | **80.94** | **最佳质量**                 |
| **ARS (Lazy)** | **Shear Force (k=5, α=0.1)** | 158.21      | 82.10     | 82.10     | **最佳平衡 (~1.5x Speedup)** |

结果显示：

1. **ARS (Sync, ρ=0.1)** 取得了最佳的泛化性能 (PPL 80.94)，证明了流形感知扰动在寻找平坦极小值方面的有效性。
2. **ARS (Lazy)** 通过强度补偿机制，以仅比 Sync 模式高 1.1 PPL 的代价，实现了约 1.5 倍的训练加速，是实际应用中的最佳选择。

### 3.2 Grokking 现象加速实验

我们在模加法任务上验证了优化器对 Grokking（顿悟）现象的加速效果。该任务中，模型需要学习模运算的内在规律。

| 优化器    | 拟合速度  | 顿悟时刻| 收敛时刻| 最终性能 | 状态  |
| :-------- | :-------- | :------ | :------ | :------- | :---- |
| **AdamW** | ~Epoch 140 | **Epoch 228** | Epoch 556 | 100.0% | ✅ 标准 Grokking |
| **AdaRMSuon** | **Epoch 28** | **Epoch 54** | **Epoch 300** | 99.9% | 🚀 **极速 Grokking** |
| **ARS** | Epoch 17 | **Epoch 100** | Epoch 290 | 99.1% | 🚀 **稳健 Grokking** |

**核心结论**：**AdaRMSuon** 将 Grokking 现象的发生时间相比 AdamW 基准提前了 **4 倍以上** (Epoch 228 → Epoch 54)，有力证明了"能量-几何解耦"与"流形平坦度约束"在加速模型泛化相变中的关键作用。

## 4. 快速开始

### 4.1. 安装

```bash
# 推荐使用 uv
uv sync
```

### 4.2. 复现关键实验

```bash
# 1. 运行 ARS Sync Mode (最佳质量)
python -m scripts.train --config config/wikitext2_line_mode_ars_rho_0.1.toml

# 2. 运行 ARS Lazy Mode (最佳平衡)
python -m scripts.train --config config/wikitext2_line_mode_ars_rho_0.1_k5_alpha0.1.toml
```

比较 `outputs/` 目录下生成的 summary 文件中的 `Eval Perplexity`。

## 5. 框架结构

该框架专为快速原型设计和清晰评估而设计。

- **Optimizers**: 位于 [`optimizer/`](optimizer/)。参见 [`optimizer/rmsuon.py`](optimizer/rmsuon.py) 和 [`optimizer/ars.py`](optimizer/ars.py)。
- **Models**: 标准架构位于 [`model/`](model/)。
- **Tasks**: 训练和评估逻辑定义在 [`task/`](task/) 中。`line mode` 实现位于 [`task/wikitext2_line.py`](task/wikitext2_line.py)。
- **Configs**: 实验配置通过 [`config/`](config/) 中的 TOML 文件管理。
- **Outputs**: 所有结果、日志和检查点都保存到 [`outputs/`](outputs/)。

## 引用

如果您在研究中使用了 F3EO-Bench，请引用相关的理论工作：

```bibtex
@software{f3eo_bench_2025,
  author = {Rui, L.},
  title = {F3EO-Bench: A Lightweight Framework for Advanced Optimizer Evaluation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dmf-archive/F3EO},
  note = {A research framework for optimizers based on the Energy-Geometry Decoupling principle.}
}
```
