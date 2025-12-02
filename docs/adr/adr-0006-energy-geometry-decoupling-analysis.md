---
title: "ADR-0006: 能量-几何解耦范式中能量源的架构决策"
status: "Accepted"
date: "2025-12-02"
authors: "Ω Researcher"
tags: ["architecture", "decision", "optimizer", "fep", "so-ofe"]
supersedes: ""
superseded_by: ""
---

# ADR-0006: 能量-几何解耦范式中能量源的架构决策

## 状态 (Status)

**Accepted**

## 背景 (Context)

在 F3EO 项目中，我们致力于开发一种基于自由能原理（FEP）和整合预测工作空间理论（IPWT）的新型优化器。我们提出了“能量-几何解耦”范式，试图将参数更新分解为“统计能量”（`energy`，决定步长/强度）和“几何结构”（`geometry`，决定方向）。

为了确定最佳的 `energy` 来源，我们面临两个主要选择：

1. **Adam 二阶矩 (vₜ)**: 基于梯度历史的统计量，被视为经验 Fisher 矩阵对角线的“黑箱”估计。
2. **KFAC 积 (A ⊗ G)**: 基于输入协方差 (A) 和输出梯度协方差 (G) 的结构化分解，理论上更能捕捉参数间的相关性。

我们需要通过实验验证哪种能量源更能准确量化观测自由能的下降速率，从而指导优化器在参数流形上进行高效的“测地线滑行”。

## 决策 (Decision)

我们决定采用 **Adam 二阶矩 (vₜ)** 作为 `Ada-RMSuon` 优化器中 `energy` 的计算基础，并确立 **AdamW (统计能量) + Muon (几何约束)** 为核心架构范式。

这一决策基于以下实验证据和理论分析：

1. **性能优势**: 在 Wikitext-2 Line Mode 任务中，基于 Adam 二阶矩的 `Ada-RMSuon` (PPL 87.61) 显著优于基于 KFAC 的 `KFAC-RMSuon` (PPL 100.36)。
2. **FEP 一致性**: 在自由能原理框架下，梯度的模长直接量化了参数分布需要漂移的“距离”以最小化自由能（意外）。Adam 的二阶矩 vₜ 忠实地记录了这种“漂移需求”的历史统计量。
3. **结构化假设失效**: KFAC 的核心假设 F ≈ A ⊗ G 在 Transformer 等复杂非线性架构中往往失效，引入了结构化偏差，导致 `energy` 估计失准。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **收敛速度提升**: `Ada-RMSuon` 展现出极快的初始收敛速度，在 4 个 epoch 内即达到最佳 PPL。
- **POS-002**: **计算效率**: 相比于需要复杂 hooks 和矩阵分解的 KFAC，基于 Adam 的实现计算成本更低，更易于扩展到大规模模型。
- **POS-003**: **理论自洽性**: 确立了梯度作为“唯一真理”的地位，避免了因错误的结构化假设而引入的偏差，符合 SOO-OFE 的设计哲学。

### 消极 (Negative)

- **NEG-001**: **过拟合风险**: 虽然收敛极快，但实验观察到在训练后期（Epoch 5+）出现过拟合反弹。这意味着当前的几何约束（Muon 正交化）虽然强力，但可能缺乏动态适应性。
- **NEG-002**: **正则化耦合**: 现有的 `weight_decay` 策略与谱更新机制存在复杂的相互作用，可能需要针对性的调整。

## 考虑的备选方案 (Alternatives Considered)

### Ada-Muon (纯几何约束)

- **ALT-001**: **描述 (Description)**: 仅应用 Muon 的谱正交化，不进行基于 `energy` 的步长缩放。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 实验表明，其收敛速度和Muon并无显著优势，且后期依然出现过拟合趋势。

### KFAC-RMSuon (结构化能量源)

- **ALT-003**: **描述 (Description)**: 使用 KFAC 的对角近似 (F_diag ≈ diag(A) ⊗ diag(G)) 来计算 `energy`。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 实验性能显著劣于 `Ada-RMSuon`。理论分析指出，Transformer 的非线性复杂性破坏了 KFAC 的 Kronecker 积假设，导致能量估计偏差。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **能量计算**: `energy` 应计算为白化后动量的 Frobenius 范数：`energy = ||m / sqrt(v)||_F`。
- **IMP-002**: **数据预处理**: 强烈建议使用 `Line Mode`（按句/段打包）进行文本数据预处理，以避免因上下文不连续引入的噪声，这对于二阶优化器尤为敏感。
- **IMP-003**: **监控指标**: 需持续监控 `PPL` (困惑度) 和 `Grad Norm`。`PPL` 的反弹是过拟合的早期预警。

## 参考文献 (References)

- **REF-001**: [RMSuon.md](../../.roo/rules/RMSuon.md) - RMSuon 优化器核心文档
- **REF-002**: [AdaFisher](https://arxiv.org/abs/2405.16397) - 对比分析的外部基准
- **REF-003**: [Hadron.md](../../.roo/rules/Hadron.md) - 早期关于 KFAC/Hadron 的实验记录
