# CIFAR-10 Long-Range Experimental Plan (50 Epochs)

> **Status**: Pending
> **Note**: 待原子化训练框架 (`training_refactor_plan.md`) 重构完毕后执行。
> **Goal**: 在 50 Epochs 内验证 ARS2-Neo 的平坦度约束 (SAM) 与几何优化 (Manifold Optimization) 的协同效应，并确定 Lazy Mode 的最佳效率平衡点。
> **Constraint**: 最小化算力消耗，跳过已知劣势的 AdamW 对照组。

## 1. 实验设计哲学

本实验采用 **深度优先二分搜索 (DFS-Binary Search)** 策略，旨在通过最少的实验次数定位最优超参数区间，并同步完成消融实验。

1. **Baseline Skip**: 已知 Muon 优于 AdamW，故直接以 Muon 为基准线。
2. **ρ-DFS (Rho Search)**: 在 Sync 模式下寻找最佳扰动半径 ρ。由于几何流形曲率对 ρ 敏感，我们将从锚点出发向两端探测。
3. **k-DFS (Lazy Efficiency)**: 在确定 ρ_opt 后，固定该参数，通过二分搜索扫描 k 值与注入强度 α 的组合。

## 2. 实验矩阵 (Experiment Matrix)

所有实验基于 ResNet-18，Batch Size 256，Epochs 50。

### Phase 1: 几何与平坦度搜索 (Rho Search, k=1)

**目标**: 确立性能下限 (Base) 与平坦度增益上限 (Sync Best)。

| ID | Config Name | Optimizer | Params | 科学目标 |
| :--- | :--- | :--- | :--- | :--- |
| **E1-Muon** | `cifar10_muon_50e` | Muon | Standard | **Baseline**: 纯几何优化的基准线。 |
| **E1-Base** | `cifar10_ars2_neo_base_50e` | ARS2-Neo | k=0 | **Ablation**: 验证能量解耦架构的独立有效性。 |
| **E1-S01** | `cifar10_ars2_neo_sync_rho01_50e` | ARS2-Neo | k=1, ρ=0.1 | **Anchor**: 初始锚点，预期稳健增益。 |
| **E1-S005** | `cifar10_ars2_neo_sync_rho005_50e` | ARS2-Neo | k=1, ρ=0.05 | **Lower Bound**: 测试极小扰动的边界。 |
| **E1-S02** | `cifar10_ars2_neo_sync_rho02_50e` | ARS2-Neo | k=1, ρ=0.2 | **Mid-High**: 二分搜索探测点。 |
| **E1-S05** | `cifar10_ars2_neo_sync_rho05_50e` | ARS2-Neo | k=1, ρ=0.5 | **Upper Bound**: 测试强扰动下的流形稳定性。 |

### Phase 2: AGA 阈值动力学探测 (AGA Threshold Search, ρ=ρ_opt)

**目标**: 跳过手工 $k$ 和 $\alpha$，通过搜索几何一致性阈值 $L$ 实现全自动效率平衡。

| ID | Config Name | Optimizer | Params | 科学目标 |
| :--- | :--- | :--- | :--- | :--- |
| **E2-AGA-L05** | `cifar10_aga_L05` | ARS2-Neo (AGA) | $L=0.05$ | **High Fidelity**: 严格几何约束，预期平均 $k \approx 2\sim3$。 |
| **E2-AGA-L10** | `cifar10_aga_L10` | ARS2-Neo (AGA) | $L=0.10$ | **Balance**: 黄金分割点，验证平均 $k$ 是否逼近 $e$。 |
| **E2-AGA-L20** | `cifar10_aga_L20` | ARS2-Neo (AGA) | $L=0.20$ | **Extreme Lazy**: 宽松约束，测试流形平稳性的上限。 |

## 3. 执行队列 (AGA 动力学逻辑)

1. **Step 1: 锚点确立**
   - 运行 `E1-Muon`, `E1-Base`, `E1-S01`。
   - *Decision*: 若 `S01 > Base`，说明 SAM 有效，进入 ρ-DFS。

2. **Step 2: ρ-DFS (深度优先二分搜索)**
   - 运行 `E1-S05`。
   - **Case A**: 若 `S05 > S01` -> 运行 `E1-S02` (探测 0.1~0.5 之间)；若 `S02` 依然好，考虑继续向上探测。
   - **Case B**: 若 `S01 > S05` -> 运行 `E1-S005` (探测更小扰动)；若 `S01` 依然最优，则 ρ_opt = 0.1。

3. **Step 3: AGA 曲线拟合**
   - 运行 `E2-AGA-L05`, `L10`, `L20`。
   - **数据采集**: 记录每个 Epoch 的 `avg_k` 和 `avg_alpha`。
   - **分析**:
     - 绘制 $Accuracy = f(L)$ 和 $Compute\_Saved = g(L)$。
     - 验证 $k_{eff}$ 是否随训练阶段（初期 vs 后期）呈现对数增长。
     - 寻找是否存在 $k \approx 2.718$ 的性能/效率奇点。

## 4. 资源估算

- **单次实验时长**: 约 30 分钟。
- **总实验时长**: 约 6 小时。
