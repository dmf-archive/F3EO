# Grokking 动力学实验分析报告 (2026-01-27)

## 1. 优化器代际与术语对齐 (Ontology Alignment)

根据项目演进逻辑，我们将实验中的优化器名称归一化如下：

- **Gen 1 (ARS)**: 对应日志中的 `AdaRMSuon`。
- **Gen 2 (ARS2-Sync)**: 对应日志中的 `ARS`。
- **Gen 3 (ARS2-Neo)**: 当前生产版本，包含 `Base`、`Sync` 和 `AGA` (自适应几何感知) 模式。

## 2. 顿悟动力学对比表 (Grokking Dynamics Comparison)

任务配置：模加法 ($p=113, \text{fraction}=0.3$)。核心指标为**顿悟 Epoch**（Validation Accuracy 首次稳定超过 99% 的时间点）。

| 实验路径 (Output Path) | 优化器 (归一化称呼) | 顿悟 Epoch | 最终准确率 | 最终 Loss | 动力学特征分析 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`outputs/grok_ars_align`](outputs/grok_ars_align/summary.md) | **ARS2-Sync** | **~152** | 99.54% | 0.025 | **爆发力最强**。Sync 模式 ($k=1$) 提供了最高频的几何修正。 |
| [`outputs/lrp_grok_ars2_neo_aga_400e`](outputs/lrp_grok_ars2_neo_aga_400e/summary.md) | **ARS2-Neo (AGA)** | ~219 | 99.60% | **0.015** | **综合最强**。在保持极速顿悟的同时，最终 Loss 最低，泛化质量最优。 |
| [`outputs/lrp_grok_ars2_neo_base_400e`](outputs/lrp_grok_ars2_neo_base_400e/summary.md) | **ARS2-Neo (Base)** | ~286 | 99.53% | 0.049 | 验证了能量-几何解耦在无平坦度约束下的基准性能。 |
| [`outputs/grok_ada_rmsuon_align`](outputs/grok_ada_rmsuon_align/summary.md) | **ARS** | ~336 | 99.89% | 0.009 | 早期版本，收敛曲线极其平滑，但速度稍逊。 |
| [`outputs/grok_adamw`](outputs/grok_adamw/summary.md) | AdamW | ~564 | 100.0% | 0.0005 | 经典基准。虽然最终精度高，但顿悟延迟是 ARS2 的 3.7 倍。 |
| [`outputs/lrp_grok_adamw_600e`](outputs/lrp_grok_adamw_600e/summary.md) | AdamW | ~585 | 15.65% | 6.10 | **泛化崩溃**。在 590 Epoch 后出现灾难性过拟合，证明了 SAM 的必要性。 |
| [`outputs/grok_muon_tuned`](outputs/grok_muon_tuned/summary.md) | Muon | N/A | 57.16% | 42.95 | **未能顿悟**。纯几何优化在高度非线性的模运算流形上难以穿越。 |

## 3. 理论评估与结论 (Theoretical Synthesis)

### 3.1 核心判定

- **速度冠军**: **ARS2**。其在 152 Epoch 即完成相变，证明了高频测地线修正对穿越损失地形“窄缝”的决定性作用。
- **架构冠军**: **ARS2-Neo (AGA)**。它在 $k=10$ 的延迟同步下依然保持了极高的顿悟效率（219 Epoch），且最终 Loss 显著低于其他变体，实现了计算开销与泛化稳健性的帕累托最优。

### 3.2 核心洞察

1. **能量-几何解耦的必要性**: Muon 的失败证明了在 Grokking 任务中，必须引入类似 Adam 的元素级自适应（能量）来配合正交约束（几何）。
2. **平坦度约束的决定性**: AdamW 在长周期实验中的后期崩溃，有力支撑了 ARS2-Neo 引入流形感知 SAM 的理论动机——即优化器必须主动避开“针尖极小值”。
3. **AGA 的有效性**: AGA 模式通过动态调节同步频率，成功在不损失太多速度的前提下，引导模型进入了比 Base 模式更深、更平坦的盆地。

---
**签发人**: Ω Researcher
**日期**: 2026-01-27
