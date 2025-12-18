# 理论洞察：嵌套 SGD (Nested SGD) 作为 Transformer-RNN 统一范式

**核心论点**: Google 的 **Titans (Nested Learning)** 与 **RWKV v7 (Goose)** 架构，尽管在实现细节上有所不同，但从第一性原理上看，均收敛于一种共同的计算范式——**嵌套 SGD (Nested SGD)**。该范式将模型的权重区分为两部分：一个在推理时保持静态的“外层权重”（Outer Weights），以及一个在推理时通过类似 SGD 的规则进行实时更新的“内层权重”（Inner Weights）。

这种“权重作为隐藏状态”的演化，本质上是将**优化过程本身（SGD）** 嵌入到了网络的前向传播中，从而实现了对上下文的动态适应。

## 1. 范式分析：双层优化视角

该范式可以形式化地描述为一个双层优化问题，其中内层优化在推理（Test-time）过程中实时发生：

### 1.1 外层循环 (训练阶段 / Meta-Learning)

`θ_outer* = argmin_{θ_outer} E_{(x,y) ~ D_train} [ L(y, f(x; θ_outer, θ_inner_final)) ]`

- **目标**: 学习一组能够为内层循环提供良好归纳偏置的“元权重” `θ_outer`。
- **过程**: 标准的反向传播训练，优化的是“学习如何学习”的能力。

### 1.2 内层循环 (推理阶段 / Test-Time Training)

`θ_inner, t = θ_inner, t-1 - η_t ∇_{θ_inner} ℓ(input_t, target_t; θ_inner, t-1)`

- **目标**: 根据当前输入序列的统计特性快速调整 `θ_inner`，以最小化瞬时预测误差或关联损失。
- **过程**: 一个广义的、单步或多步的梯度下降过程，将序列历史编码进权重。

## 2. 架构实例与形式化

### 2.1 Google Titans: 显式的 Test-Time Memorization

**引用**: Behrouz et al., "Titans: Learning to Memorize at Test Time", arXiv:2501.00663 (2025).

Titans 引入了 **Neural Long-Term Memory** 模块，其核心是一个受动量 SGD 驱动的内层模型：

- **内层权重**: 记忆矩阵 $M_t$。
- **关联损失 (Associative Loss)**: $ℓ_t = ‖M_{t-1}(k_t) - v_t‖^2$。
- **更新规则 (Momentum-based SGD)**:
  $S_t = \beta S_{t-1} + \nabla_{M} ℓ_t$
  $M_t = (1 - \alpha) M_{t-1} - \eta S_t$
  其中 $\alpha$ 为遗忘门 (Forgetting gate)，$\beta$ 为动量因子。
- **分析**: Titans 是 Nested SGD 的直接实现。它通过显式的梯度下降将键值对 $(k, v)$ 写入权重，实现了线性复杂度的长程记忆。

### 2.2 RWKV-7 "Goose": 解析式的动态状态演化

**引用**: Peng et al., "RWKV-7 'Goose' with expressive dynamic state evolution", arXiv:2503.14456 (2025).

RWKV-7 通过复杂的递归算子实现了类似 SGD 的权重更新效果：

- **内层权重**: 递归状态矩阵 $S_t$。
- **状态转移方程**:
  $S_t = G_t \odot S_{t-1} + v_t \hat{k}_t^\top$
- **动态几何约束 (Dynamic Recurrence)**:
  $G_t = \text{Diag}(d_t) - \tilde{k}_t i_t^\top$
  其中 $d_t$ 代表通道衰减，$i_t$ 代表替换强度 (Replacement strength)。
- **分析**: RWKV-7 的更新规则可以被视为一种**近似SGD**。它将梯度计算与更新步骤融合在前向算子中，其 $\tilde{k}_t i_t^\top$ 项在数学上类似于对权重矩阵进行秩-1 的 Delta Rule 更新。

## 3. 结论与局限

Titans 和 RWKV-7 共同揭示了，通过将一部分权重动态化并用 SGD 规则进行更新，可以在保持 RNN 线性复杂度的同时，实现甚至超越 Transformer 的上下文学习能力。

- **Titans**: 结构更通用（支持多层 MLP 记忆），更新逻辑更显式（标准 SGD），适合超长上下文。
- **RWKV-7**: 结构更简洁（单层线性递归），更新逻辑更高效（解析递归），在推理效率上具有绝对优势。

**未来方向**: 真正的持续学习需要内外层权重之间更深度的、基于**自由能原理 (FEP)** 的协同演化。当前的内层更新仍主要依赖于启发式的损失函数（如关联损失），而未来的演进方向是通过原生FEP优化，对完整的模型权重θ进行自然稀疏梯度更新，杜绝预定义“memory/fast weights”。
