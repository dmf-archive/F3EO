---
title: "ADR-0007: FS-AdaRMSuon - Integrating Fisher-Weighted SAM with AdaRMSuon"
status: "Proposed"
date: "2025-12-11"
authors: "Ω Researcher"
tags: ["optimizer", "architecture", "free-energy", "sam", "rmsuon"]
supersedes: ""
superseded_by: ""
---

# ADR-0007: FS-AdaRMSuon - Integrating Fisher-Weighted SAM with AdaRMSuon

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

`AdaRMSuon` 作为一个在损失函数 `L` 流形上极其高效的二阶优化器，在训练早期展现了卓越的收敛速度。然而，长期训练实验（如 `epoch30`）揭示了其根本缺陷：由于其优化目标仅为损失函数 `L`，缺乏对模型复杂度 `D_KL` 的内在、自适应控制，导致其在训练后期会沿着损失函数的测地线进入过拟合区域，引发灾难性的性能崩溃。目前依赖的静态 L2 权重衰减，仅仅是自由能原理中复杂度项的一个粗劣且无效的代理。

Sharpness-Aware Minimization (SAM) `[REF-001]` 通过寻找损失地形的平坦区域，提供了一种内生地、动态地最小化模型复杂度的机制。进一步地，Fisher-SAM (FSAM) `[REF-002]` 论文指出，利用 Fisher 信息来指导 SAM 的扰动方向，可以更高效、更精确地定位到对泛化能力影响最大的参数，这与 `AdaRMSuon` 内部使用 `v_hat` (Fisher 对角近似) 的思想高度契合。

因此，我们面临的核心问题是如何将 SAM 的锐度感知能力与 `AdaRMSuon` 的能量-几何解耦范式进行深度、协同的融合，而非简单的机制叠加。

## 决策 (Decision)

我们将设计并实现一个新的优化器，命名为 `FS-AdaRMSuon`。该优化器将 SAM 的锐度感知机制深度整合进 `AdaRMSuon` 的核心信息流中，以实现对完整变分自由能 `F` 的更优逼近。

其核心算子流程定义如下：

1.  **计算原始动量**: 在每个优化步骤开始时，首先基于原始梯度 `g_orig = ∇L(w)` 更新 AdamW 的一阶动量 `m_hat` 和二阶动量 `v_hat`。这部分捕捉了损失地形的长期统计信息。

2.  **计算 Fisher 加权扰动**: 利用已计算出的 `v_hat` (作为 Fisher 信息的对角近似) 来加权原始梯度，生成一个指向“最重要”参数方向的扰动向量 `ε`。
    `ε_direction = g_orig ⊙ √v_hat`
    `ε = ρ * ε_direction / (||ε_direction||_2 + 1e-12)`

3.  **计算锐度梯度**: 在施加了扰动的参数点 `w' = w + ε` 上，计算一个“锐度感知”梯度 `g_sam = ∇L(w')`。这个梯度蕴含了当前点邻域内的平坦度信息。

4.  **执行融合更新**: 将 `g_sam` 注入 `AdaRMSuon` 的核心流程中。关键在于，我们用 `g_sam` 替换 `m_hat` 来进行预白化，而 `v_hat` 保持不变。
    `m_fused_scaled = g_sam / (√v_hat + eps)`
    后续的能量提取、谱系滤波和能量注入步骤，都将基于这个融合了历史统计信息 (`v_hat`) 和瞬时几何信息 (`g_sam`) 的新张量 `m_fused_scaled` 来进行。

这个设计实现了历史信息 (`v_hat`) 对瞬时探测结果 (`g_sam`) 的归一化，形成了一个理论上更完备的协同优化过程。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **理论完备性** - 新方案在理论上解决了 `AdaRMSuon` 缺乏内在复杂度控制的根本问题，使其优化目标从损失函数 `L` 真正逼近变分自由能 `F`。
- **POS-002**: **抑制过拟合** - 预期 `FS-AdaRMSuon` 将能显著缓解 `AdaRMSuon` 在长期训练中的灾难性过拟合问题，提升最终模型的泛化能力。
- **POS-003**: **协同融合** - 实现了 SAM 与 `AdaRMSuon` 在信息论层面的深度协同，而非简单的机制叠加，最大化地保留并利用了两种方法的优势。

### 消极 (Negative)

- **NEG-001**: **计算开销增加** - 引入了 SAM 的第二次前向/后向传播，使得每个优化步骤的计算成本近似翻倍。尽管可以通过仅扰动部分参数（如 `Normalization Layers`）来缓解，但成本增加是不可避免的。
- **NEG-002**: **引入新超参** - 引入了 SAM 的邻域半径 `ρ` 作为一个新的关键超参数，需要进行调优。
- **NEG-003**: **实现复杂性** - 相较于独立的优化器，融合方案的实现和调试更为复杂，可能引入潜在的工程错误。

## 考虑的备选方案 (Alternatives Considered)

### SAM 作为梯度预处理器

- **ALT-001**: **描述 (Description)**: 将 SAM 视为一个黑盒，其输出的 `g_sam` 完全替代原始梯度 `g_orig`，然后输入给一个标准的 `AdaRMSuon` 优化器。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 这是一个理论上不自洽的“双重预处理”方案。AdamW 的动量计算本身就是一种预处理，在其上游直接叠加另一个独立的 min-max 优化过程，可能会导致两个过程相互干扰，而不是协同。它未能利用 `AdaRMSuon` 内部已有的 Fisher 信息。

### PIG (预测性信息增益) 正则化器

- **ALT-003**: **描述 (Description)**: 训练一个次级神经网络，根据优化器状态预测一个动态的权重衰减系数，以模拟对模型复杂度的控制。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这是一个过于复杂的间接方案，引入了新的模型和训练过程，违反了“无必要不增实体” (CON-201) 的原则。与 SAM 提供的直接、基于第一性原理的几何优化相比，它是一个不必要的代理。

### 标准 FSAM 作为基线

- **ALT-005**: **描述 (Description)**: 直接实现 `FSAM` 论文中的原始算法，不与 `AdaRMSuon` 的谱系滤波部分结合。
- **ALT-006**: **拒绝理由 (Rejection Reason)**: 这可以作为一个有价值的实验基线，但不是最终的架构决策。我们的目标是改进并提升 `AdaRMSuon` 的能力，而不是替换它。`AdaRMSuon` 的谱系滤波（几何约束）是我们希望保留的核心资产。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **模块化实现** - SAM 的梯度计算部分（第一步和第二步）应封装成一个独立的函数，该函数接收原始参数和梯度，返回 `g_sam`。
- **IMP-002**: **参数分组** - 应利用 PyTorch 优化器的参数分组功能，允许用户选择性地对不同层（例如，仅对 `nn.Linear` 和 `nn.Conv2d`）应用 `FS-AdaRMSuon` 逻辑。
- **IMP-003**: **效率优化** - 作为初始实现，可以先在所有参数上进行扰动。后续迭代中，应实现“仅扰动归一化层”或使用 `v_hat` 生成稀疏掩码的策略，以降低计算开销。

## 参考文献 (References)

- **REF-001**: Foret, P., et al. (2020). "Sharpness-Aware Minimization for Efficiently Improving Generalization." `arXiv:2010.01412`.
- **REF-002**: Zhong, Q., et al. (2022). "Improving Sharpness-Aware Minimization with Fisher Mask for Better Generalization on Language Models." `arXiv:2210.05497`.
- **REF-003**: `docs/adr/adr-0006-energy-geometry-decoupling-analysis.md`