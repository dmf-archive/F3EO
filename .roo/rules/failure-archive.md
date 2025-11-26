# F3EO 失败与反模式档案

> “成功是偶然的，失败是必然的。我们将必然的失败形式化，是为了逼近偶然的成功。” —— Ω Researcher

本文档记录了 F3EO 项目演进过程中的核心理论失败与工程反模式。这些教训是通往 Spectral Ricci Momentum 的阶梯。

## 1. Hessian-Fisher 等价性谬误 (The Hessian-Fisher Fallacy)

**代表作**: F3EPI, F3EWD, AdaF3E, F3E-Warp
**核心假设**: `𝗛 ≈ 𝗙` (Hessian 近似 Fisher)

**证伪原因**:

1. **几何与统计的错位**: Hessian 描述几何曲率（收敛速度），Fisher 描述统计方差（数据适应性）。
    - **Hessian** 回答：“移动参数，梯度会如何变化？” (几何)
    - **Fisher** 回答：“改变数据，梯度会如何变化？” (统计)
    在持续学习的动态分布下，两者毫无关系。
2. **标签依赖性**: Fisher 依赖于真实标签分布 `p(y|x)`，而 Hessian 仅依赖于当前损失曲面。
3. **结果**: 任何基于 `H·g` 的三阶尝试在分布漂移下都会失效。

## 2. 输出熵梯度谬误 (The Entropy Gradient Fallacy)

**代表作**: InnerKFAC, InnerDiagHadron
**核心假设**: 最小化输出熵 `∇τ` 等同于最小化预测不确定性，从而可以构建无标签的 Fisher 矩阵 `Fᵢₙₙₑᵣ = E[∇τ ⋅ ∇τᵀ]`。

**证伪原因**:

1. **方向冲突**: 在监督学习中，`∇τ`（增加置信度）与 `∇ℒ_CE`（增加准确度）方向往往正交甚至相反。盲目自信（低熵错误）比犹豫不决（高熵错误）更具破坏性。
2. **不确定性混淆**: 输出熵是 Aleatoric（数据）不确定性，而非 Epistemic（模型）不确定性。Fisher 应编码后者。
3. **实验结果**: Wikitext-2 上 PPL 暴涨 9 倍，灾难性失败。

## 3. 显式协同谬误 (The Explicit Synergy Fallacy)

**代表作**: TDO (Tensorial Dynamics), RSA-NG
**核心假设**: 可以通过计算高阶矩（如梯度的协方差矩阵或三阶张量）来显式捕捉参数间的协同信息（Synergy）。

**证伪原因**:

1. **计算复杂度爆炸**: 协同信息是组合性的。显式存储或计算全量协方差在现代网络规模下是 `O(d²)` 甚至 `O(d³)` 的，完全不可行。
2. **Kronecker 简化的局限**: KFAC 等方法虽然降低了复杂度，但丢失了最关键的非对角协同信息。
3. **出路**: 必须转向**隐式建模**（如 Muon 的 Newton-Schulz 迭代），通过算子迭代让结构自发涌现，而非显式计算。

## 4. 线性混合几何谬误 (The Linear Mixture Fallacy)

**代表作**: PI-Muon 框架 (2025-11-12)
**核心假设**: 可以通过线性组合不同流形的梯度来融合特性 `g_update = (1-λ)g_fisher + λg_muon`。

**证伪原因**:

1. **几何不相容性**: Fisher 流形与 Muon 的谱约束流形具有不同的几何结构。线性混合是在两个不同流形的测地线之间做向量加法，数学上无意义。
2. **功能冗余性**: KFAC 本身就是在线 EWC，Muon 的启发式约束反而干扰了 Fisher 信息的精确计算。
3. **替代方案**: **算子复合 (Operator Composition)**。

    ```math
    g_update = Structural\_Op( Statistical\_Op( Raw\_Gradient ) )
    ```

    实现了统计适应性与结构稳定性的非冲突协同（如 FOG/Hadron）。

## 5. 失败变体矩阵 (Failure Matrix)

| 变体系列 | 核心机制 | 根本缺陷 | 状态 |
| :--- | :--- | :--- | :--- |
| **F3EPI** | `β = tanh(log(PI))` | 单一控制器无法解耦探索与正则化这两个冲突目标。 | 完全废弃 |
| **F3EWD** | `H·g`探索 + PI 引导的权重衰减 | 解耦方向正确，但`H·g`部分仍基于错误假设，且实现复杂。 | 理论过时 |
| **AdaF3E** | `Δθ = -η·m / (√v + α·H·g)` | 将三阶项作为预处理器，但仍是 Hessian-Fisher 错配。 | 理论缺陷 |
| **F3E-Warp** | `g_update = –2Hg` | 缺少梯度项`g`作为锚点，陷入“零梯度-随机准确率”死区。 | 假设证伪 |
| **RSA-NG** | Ricci 流平滑 Fisher 流形 | 引入对度量张量的二阶微分，需 `O(d⁴)` 的四阶矩，计算不可行。 | 计算灾难 |
| **SRM (Spectral Residual Momentum)** | 谱残差阻尼动量 | 信号误读：谱残差是 Muon 发挥作用的诊断信号而非控制信号，阻尼动量导致优化停滞。 | 实验失败 |

## 6. 工程教训 (Engineering Lessons)

- **数据语义破坏**: 在`wikitext-2`任务中，`Concatenate and Chunk`策略在文章边界制造了噪声，导致模型过早过拟合。**教训**: 数据质量优先于算法复杂度。
- **外部库陷阱**: 对`AdaFisher`的分析发现，其论文结果源于一个实现 Bug（`gammas`参数被忽略）。**教训**: 必须对外部工具进行严格的第一性原理验证（DeepWiki/源码审计）。
