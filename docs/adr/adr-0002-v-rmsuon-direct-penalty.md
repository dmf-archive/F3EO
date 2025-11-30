---
title: "ADR-0002: V-RMSuon - 基于二阶矩的直接惩罚正则化"
status: "Deprecated"
date: "2025-11-30"
authors: "Ω Researcher"
tags: ["optimizer", "regularization", "fep", "ipwt"]
supersedes: ""
superseded_by: ""
---

# ADR-0002: V-RMSuon - 基于二阶矩的直接惩罚正则化

## 理论缺陷：衔尾蛇问题 (Ouroboros Problem)

该方案及其变体（如归一化惩罚、等量级对抗）被废弃，因为它们存在一个根本性的理论缺陷：**当下的瞬态更新强度同时驱动了遗忘强度，从而构成了一个灾难性的正反馈循环**。

1. **正反馈循环**:
    * 一个大的瞬时梯度 `g_t` -> 导致大的 `energy_t`。
    * 大的 `energy_t` -> 导致大的主更新 `Update_main`。
    * 大的 `energy_t` -> (在我们的方案中) 导致大的正则化衰减 `Update_reg`。
    * 大的正则化 -> 导致参数被大幅度衰减，从而“遗忘”了过去学到的知识。
    * 遗忘 -> 导致模型在下一个时间步上产生更大的预测误差 -> 导致更大的梯度 `g_t+1`。

2. **灾难性遗忘催化剂**: 这个正反馈循环将一个旨在抑制不稳定的机制，转变成了放大不稳定的加速器。任何由数据噪声引起的瞬时梯度增大，都会被这个机制放大，转化为对模型历史知识的破坏性遗忘，从而将模型推向更不稳定的状态。

因此，所有将瞬时更新强度（如 `energy`）与长期演化（如权重衰减）直接耦合的方案，在理论上都是不成立的。

## 状态 (Status)

Proposed | Accepted | Rejected | Superseded | **Deprecated**

## 背景 (Context)

在对 `RMSuon` 的研究中，我们发现其长期训练的稳定性严重依赖于一个外部的、启发式的 L2 `weight_decay` 超参数。一系列的理论分析和实验（如 `ADR-0001`）表明，这种依赖源于 `RMSuon` 能量注入机制与 L2 正则化范式之间的根本性不匹配。我们需要一种内禀的、与 `RMSuon` 的信息几何动力学兼容的正则化机制。

经过深入的第一性原理探讨，我们确认了问题的核心在于需要一个能直接最小化模型复杂度（由 Fisher 迹 `Tr(F)` 度量）的正则化项，以与最小化不准确性（由 `energy` 度量）的 `RMSuon` 主更新步骤形成对偶。

## 决策 (Decision)

我们决定设计并实现一个名为 `V-RMSuon` 的新优化器变体。该变体用一种全新的、基于二阶矩的**直接惩罚 (Direct Penalty)** 机制，彻底取代传统的 `weight_decay`。

### 理论推导 (FEP/IPWT Framework)

1. **自由能最小化**: FEP 框架指出，系统的优化目标是最小化变分自由能 `F`，其包含**不准确性**和**复杂度**两个部分。
2. **动力学对偶**: `RMSuon` 的主更新步骤 `~ energy * O` 旨在最小化不准确性。我们需要一个对偶的步骤来最小化复杂度。
3. **Fisher 迹作为复杂度**: 我们将 Fisher 迹 `Tr(F)` 作为模型复杂度的可计算代理。`AdamW` 的二阶矩 `v̂` 是 `diag(F)` 的良好近似，因此最小化 `Σ v̂ᵢ` 成为我们的直接目标。
4. **直接惩罚**: 我们选择直接惩罚 `√v̂`，因为它的梯度 `∇R(p) = λ * √v̂` 直接作用于参数的敏感度，而与参数自身的大小无关。这比调制 L2 衰减的方案（`∇R ~ p * √v̂`）在理论上更纯粹，更直接地指向最小化 `Tr(F)` 的目标。

### 算法实现

`V-RMSuon` 的核心更新规则如下：

`p_t+1 = p_t - lr * ( (energy_t / ||O_t||) * O_t + λ * √v̂_t )`

其中 `λ` 是一个新的全局衰减率超参数（在实现中复用 `weight_decay` 字段）。

这个更新步骤被实现为一个高效的 `total_update` 计算，以最小化计算开销：

```python
direct_penalty = v_hat.sqrt().mul_(decay_lambda)
total_update = O.mul(scale).add_(direct_penalty)
p.add_(total_update, alpha=-lr)
```

## 实验计划与预期观察 (The Experiment)

### 实验设置

1. **创建新优化器**: 在 `optimizer/__init__.py` 中注册 `V_RMSuon`。
2. **创建配置文件**: 创建 `config/wikitext2_rope_v_rmsuon.toml`，将优化器名称指定为 `V_RMSuon`。
3. **超参数设置**:
    * `lr`: 0.0001 (与 `RMSuon` 保持一致)
    * `betas`, `eps`, `ns_steps`: 与 `RMSuon` 保持一致。
    * `weight_decay` (即 `λ`): 我们将从一个较小的值开始，例如 `1e-5`，因为它直接作用于 `√v̂`，其量级可能与标准 `wd` 不同。
    * `epochs`: 10 (正如您所说，`RMSuon` 家族主打样本效率)。
4. **执行实验**: `python -m scripts.train --config config/wikitext2_rope_v_rmsuon.toml`

### 理论预测与待观察现象 (Flags to Watch)

1. **[FLAG-1] 长期稳定性**:

    * **预测**: `V-RMSuon` 应该能显著抑制 `RMSuon` 在 `wd=0` 时出现的灾难性过拟合。PPL 曲线在 10 个 epoch 内应保持下降或平稳，而不是像 `RMSuon` 那样在第 3-4 个 epoch 后急剧上升。
    * **观察**: 监控验证集 PPL 随 epoch 的变化曲线。

2. **[FLAG-2] 早期收敛速度**:

    * **预测**: 新的正则化项可能会轻微减慢早期的收敛速度，因为它在训练开始时就起作用。Epoch 1 的 PPL 可能会略高于标准 `RMSuon` (wd=0.1)。
    * **观察**: 对比 `V-RMSuon` 和 `RMSuon` 在 Epoch 1-3 的 PPL 值。

3. **[FLAG-3] 梯度与能量动力学**:

    * **预测**: `V-RMSuon` 的梯度范数 (`grad_norm`) 和 `energy` 应该比 `RMSuon` 更平稳。由于 `√v̂` 惩罚项的存在，高 `energy` 的峰值应该会受到抑制。
    * **观察**: 监控并对比 `grad_norm` 和 `PI` (其 `Accuracy` 项与 `energy` 相关) 的日志输出。

4. **[FLAG-4] 超参数敏感度**:
    * **预测**: `λ` (即 `weight_decay`) 的选择应该比 `RMSuon` 的 `wd` 更不敏感。理论上，由于正则化是自适应的，`λ` 的变化应该只影响正则化的总体强度，而不会像 `RMSuon` 那样在 0 和 0.1 之间产生质变。
    * **观察**: (未来实验) 运行不同 `λ` 值的实验，观察其对最终性能的影响。

### 成功标准

* **主要标准**: `V-RMSuon` 在 10 个 epoch 的训练中，其最终的验证集 PPL 显著低于 `RMSuon` (wd=0.1, 0.5) 在相同 epoch 数下的结果。
* **次要标准**: `V-RMSuon` 展现出比 `RMSuon` 更稳定的训练曲线（PPL, grad_norm）。
