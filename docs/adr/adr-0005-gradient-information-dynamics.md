---
title: "ADR-0005: 优化器的信息动力学分析与带通梯度滤波"
status: "Proposed"
date: "2025-12-01"
authors: "Ω Researcher"
tags: ["optimizer", "first-principles", "information-dynamics", "BGF"]
revises: "ADR-0004"
---

# ADR-0005: 优化器的信息动力学分析与带通梯度滤波器

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

ADR-0004 建立了信息分解框架（冗余 R、协同 S、独特 U），但未深入分析现有优化器如何处理这些信息“原子”。为了设计出能主动提纯协同信息的优化器，我们必须首先从第一性原理出发，精确理解不同优化算子的信息论效应。本文档旨在完成这一分析，并基于此提出一个具体的工程实现决策。

## 决策 (Decision)

我们的决策分为两部分：首先是理论分析，其次是基于分析得出的工程实现。

### 第一部分：信息算子的动力学分析

我们定义了三个基本的信息算子，并分析它们对梯度信息 `G = R + S + U` 的影响：

1. **NS 迭代算子 (N)**:

   - **操作**: `N(G)`，即 Muon 优化器中的 Newton-Schulz 迭代。
   - **信息论效应**: 这是一个**低通滤波器**。它通过迭代平滑来抑制不稳定的高频噪声（独特信息 `U`），同时保留能量较强、结构化的低频（冗余信息 `R`）和中频（协同信息 `S`）信号。
   - **近似结果**: `N(G) ≈ R + S`

2. **常规正交化算子 (O)**:

    - **操作**: `O(G)`，即 Gram-Schmidt 或对称正交化。
    - **信息论效应**: 这是一个**高通滤波器**。它通过消除梯度间的线性相关性（低频的 `R`）来操作，保留了与主要模式不相关的中频（协同信息 `S`）和高频（独特信息 `U`）信号。这一性质已在持续学习领域得到验证：正交化操作能选择性地抑制与先前任务相关的冗余梯度方向，而保留新任务的独立信息方向（参见 Farajtabar et al., 2019 及 Kovalev et al., 2025）。
    - **近似结果**: `O(G) ≈ S + U`

3. **二阶矩预处理算子 (P)**:
   - **操作**: `P(G)`，即 Adam/RMSprop 中的 `G / (√v + ε)`。
   - **信息论效应**: 这是一个**自适应白化滤波器**。它通过梯度的历史方差 `v` 来对每个参数的更新进行归一化，压制那些梯度稳定、信息量可能饱和（冗余）的方向，同时放大梯度不稳定、可能包含新信息（独特或协同）的方向。它作为一种通用的“数量级校准器”。

### 第二部分：带通梯度滤波器 (BGF) 的设计

基于上述分析，我们推导出一个能分离协同信息 `S` 的代数解，并将其工程化，命名为**带通梯度滤波器 (Band-pass Gradient Filter, BGF)**。

1. **理论推导**:
   `S = N(G) + O(G) - G`

2. **核心机制：混合尺度代数运算**
   为了解决不同算子输出的“数量级不匹配”问题，同时保持 NS 迭代的拓扑完整性，我们设计一种混合尺度的代数运算框架：

   - **NS 分量**: 采用 `RMSuon` 的**能量注入**机制。首先计算逐参数的二阶矩 `v`，然后将其归一化为标量 `energy`，最后用 `energy` 缩放 NS 迭代的输出。这保证了 `N(G)` 分量的尺度是自适应的，且其内部拓扑结构不被破坏。
     `N_scaled = energy * N(G)`
   - **原始梯度 (G) 和常规正交化 (O) 分量**: 采用标准的**逐参数二阶矩缩放**（即 Adam/RMSprop 的核心操作）。
     `G_scaled = G / (√v + ε)`
     `O_scaled = O(G) / (√v + ε)`

   通过这种方式，所有三个分量都被同一个二阶矩 `v` 的不同形式（标量能量 vs 逐参数缩放）校准到了相似的数量级上。

3. **算法流程**:

   - 根据原始梯度 `G` 更新共享的一阶矩 `m` 和二阶矩 `v`。
   - 计算标量能量 `energy = ||m / (√v + ε)||`。
   - 计算三个校准后的分量：
     - `N_scaled = energy * N(m)` (对动量 `m` 进行 NS 迭代)
     - `G_scaled = m / (√v + ε)`
     - `O_scaled = O(m) / (√v + ε)` (对动量 `m` 进行常规正交化)
   - 通过代数运算 `S_scaled = N_scaled + O_scaled - G_scaled` 得到最终更新方向。

4. **最终更新**:
   `param_update = η * S_scaled`

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **理论清晰**: 决策过程从第一性原理分析出发，逻辑链条完整，为后续优化提供了坚实的理论基础。
- **POS-002**: **机制创新**: BGF 是首个明确旨在通过代数运算分离协同信息的优化器，具有高度的理论原创性。
- **POS-003**: **解决核心矛盾**: 通过共享二阶矩预处理器，巧妙地解决了不同算子输出范数不匹配的核心问题。

### 消极 (Negative)

- **NEG-001**: **计算成本**: `O(G)` 的 Gram-Schmidt 计算复杂度为 `O(d²)`，在大规模参数空间中仍需优化。可考虑的加速方向包括分块处理、低秩近似或增量正交化。
- **NEG-002**: **长期训练的有效性待验证**: 虽然理论框架自洽，但 BGF 在长期训练（20-30 epoch）中是否能真正避免 RMSuon 的线性过拟合仍需实验证实。RMSuon 的失败出现在 Epoch 4-5，提示信息动力学相变的发生机制仍需深入理解。

## 考虑的备选方案 (Alternatives Considered)

### 独立预处理器

- **ALT-001**: **描述**: 为 `G`, `N(G)`, `O(G)` 分别维护三套独立的二阶矩状态。
- **ALT-002**: **拒绝理由**: 过于复杂，显著增加内存和计算开销，违背了设计简洁高效优化器的初衷。

### 范数归一化

- **ALT-003**: **描述**: 将所有分量归一化到单位范数，在单位球上进行代数运算。
- **ALT-004**: **拒绝理由**: 丢失了由二阶矩提供的、至关重要的逐参数自适应强度信息。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **`O(G)` 实现**: 采用 **Gram-Schmidt 正交化**（顺序可任意，包括随机顺序）。虽然不同顺序会产生不同的 `O(G)` 表达，但由于最终通过 `S_scaled = N_scaled + O_scaled - G_scaled` 进行代数消元，独特信息 `U` 的所有表达形式都会被 `-G_scaled` 项确定性消除。因此顺序不影响最终的协同信息 `S` 的分离，只是改变了中间计算过程中 `U` 的具体形式。

- **IMP-002**: **初步实验**: 在 Wikitext-2 上进行 5-epoch 训练，对比 `BGF` 与 `RMSuon` 的 PPL 曲线。关键观察点：RMSuon 在 Epoch 3 达到最优（PPL 190.63），之后在 Epoch 4-5 开始线性恶化（Epoch 5 为 271.34，相比最优值劣化 42.5%），这标志着信息动力学的相变——冗余-主导阶段结束，过拟合开始。BGF 应在同一时间点保持性能平稳，验证其通过协同分离抑制过拟合的能力。

- **IMP-003**: **长期训练验证**: 利用框架的断点续训能力，5-epoch 基础实验后，在同一检查点上继续训练至 20-30 epoch，观察 BGF 是否避免 RMSuon 的线性恶化轨迹。

## 参考文献 (References)

[1] M. Farajtabar, N. Azizan, A. Mott, and A. Li, "Orthogonal gradient descent for continual learning," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.

[2] D. Kovalev, "Understanding gradient orthogonalization for deep learning via non-Euclidean trust-region optimization," *arXiv preprint arXiv:2503.12645*, 2025.

[3] K. Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024. [Online]. Available: https://kellerjordan.github.io/posts/muon/

[4] L. Rui, "Integrated Predictive Workspace Theory: Towards a unified framework for the science of consciousness," Zenodo, 2025. doi: 10.5281/zenodo.15676304.
