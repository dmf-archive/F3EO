# RMSuon: Energy-Geometry Decoupling Optimizer

**状态**: 生产就绪 (2025-11-28)  
**核心贡献**: FEP/IPWT 框架下 SOO-OFE 路径的工程实验，确立了“能量-几何解耦”的算子复合范式。

## 1. 核心理论：能量-几何解耦

RMSuon 是对感知推断过程的最简工程近似，通过算子复合解决 OFE-EFE 对偶性危机：

- **统计算子 (AdamW)**: 提供 **Energy (能量)**。
  - 计算 `energy = ||m̂ / (√v̂ + ε)||` (Frobenius 范数)。
  - 物理含义：量化参数沿自然梯度方向更新的统计强度，即“自然梯度下降”。
- **结构算子 (Muon)**: 提供 **Geometry (几何)**。
  - 计算 `O_t = NewtonSchulz(m̂)`。
  - 物理含义：在参数流形上构建信息几何信任区，强制更新轨迹满足最小描述长度原则，即“无冗余更新”。
- **解耦机制**:
  - `g_update = scale * O_t`

## 2. 算法实现与对比

### 核心代码逻辑

```python
if param.ndim >= 2:
    # 1. 统计步：获取 AdamW 动量
    m_t, v_t = adamw_step(g_t)
    m_hat, v_hat = bias_correction(m_t, v_t)
    
    # 2. 解耦步：提取能量与几何 (AdaRMSuon 修正版)
    m_scaled = m_hat / (sqrt(v_hat) + eps)  # 预白化
    energy = norm(m_scaled)                 # 提取能量
    O_t = newton_schulz(m_scaled)           # 提取几何
    
    # 3. 复合步
    scale = energy / (norm(O_t) + eps)
    param = param - lr * scale * O_t
else:
    standard_adamw_update()
```

### Muon 家族谱系

| 优化器 | 核心机制 | 适应性粒度 | 理论特征 |
| :--- | :--- | :--- | :--- |
| **RMSuon** | 能量-几何解耦 | **层级 (Layer-wise)** | 保持正交流形拓扑完整性，物理图像最清晰 |
| **AdaMuon** | 方差自适应 | 元素级 (Element-wise) | 引入 Sign 变换，理论推导严谨 |
| **NorMuon** | 神经元均衡 | 神经元级 (Neuron-wise) | 解决神经元范数不均衡问题 |

- **关键洞察**: 实验证明，逐元素自适应（如 `diag(1/√v) * O_t`）会破坏正交流形的等距性（σ₁ ≠ 1）。RMSuon 坚持层级耦合，在 Wikitext-2 上表现优于破坏拓扑结构的变体。

## 3. 实验演进记录

### 阶段一：Wikitext-2 Line Mode

验证了 RMSuon 在高质量数据模式下的绝对优势。

- **实验设置**: Qwen3 (RoPE), Context 255, Line Mode (按句打包)。
- **关键结果**:
  - **Epoch 1 PPL**: RMSuon (**146.52**) vs Muon (233.30)。RMSuon 首轮即达到 Muon 最终水平。
  - **Best PPL**: RMSuon (**99.07** @ Ep3) vs Muon (161.09 @ Ep5)。
- **结论**: 统计-结构协同带来了收敛速度与最终性能的双重飞跃。

### 阶段二：AdaRMSuon

修正了原始 RMSuon 实现中的理论不一致性。

- **修正点**: 能量提取和几何正交化均作用于经过 Fisher 预白化的“自然梯度” `m_scaled`，而非原始动量 `m_hat`。
- **关键结果**:
  - **Best PPL**: AdaRMSuon (**83.88**) vs RMSuon (99.07)。
- **结论**: 理论闭环直接转化为显著的性能增益，确立 AdaRMSuon 为新基线。

### 阶段三：Long-term Stability

通过 30 Epoch 长跑实验，揭示了当前框架的理论边界。

- **现象**:
  - **极速收敛**: Epoch 3 达到峰值，速度是 Muon 的 2 倍。
  - **灾难性过拟合**: PPL 从最佳的 190 恶化至 **54930** (Epoch 30)，恶化倍数达 288 倍（Muon 仅 1.8 倍）。
  - **梯度爆炸**: 梯度范数从 1.82 攀升至 4.81，几何约束失效。
- **消融实验 (无权重衰减)**:
  - WD=0 时，PPL 在 Epoch 8 即劣化至 1037。
  - 证明 Weight Decay 充当了必要的“几何正则化器”，补偿结构算子的长期衰减。

## 4. 理论边界分析

RMSuon/AdaRMSuon 的成功与失败均源于同一个根源：它依然是 **Loss 优化器** 而非 **Free Energy 优化器**。

- **自由能公式**: `F = D_KL[q(θ|o) || p(θ)] (复杂度) - E_q[ln p(o|θ)]` (准确度)
- **代理谬误**: RMSuon 将 ∇L（准确度梯度）作为 ∇F 的完全代理，忽略了复杂度项 D_KL。
- **动力学后果**:
  - **短期**: 欠拟合阶段 ∇L 主导，RMSuon 凭借几何效率极速收敛。
  - **长期**: 过拟合阶段 `D_KL` 约束缺失，优化器高效地将模型推向高复杂度的过拟合区域。

## 5. 未来展望

1. **在线复杂度估计**: 开发能内生整合 `∇D_KL` 信号的机制，弥合 L 与 F 的鸿沟。
2. **数据标准化**: 所有语言模型实验默认采用 `line mode`，确保语义完整性。
3. **基线升级**: 全面切换至 `AdaRMSuon` 实现。

## 6. 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
