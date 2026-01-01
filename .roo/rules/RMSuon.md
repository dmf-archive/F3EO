# RMSuon 家族：在黎曼流形上滑行

状态: 生产就绪 (2025-12-31)
核心贡献: 确立了“能量-几何解耦” (Energy-Geometry Decoupling) 的算子复合范式，并为探索 Geodesic SAM 提供了理论与工程基础。

## 1. 理论

### 1.1 优化的本质：在测地线上滑行

在信息几何视角下，优化不仅是损失函数 `L(θ)` 的梯度下降，更是概率分布流形上的测地线运动。RMSuon 旨在通过解耦“步伐的大小”（能量统计）与“迈步的方向”（流形几何）来逼近这一理想状态。

### 1.2 问题：不同的优化器，对地形的假设不同

- SGD: 假设欧几里得平直空间。它是“盲人登山者”，仅凭局部坡度 `∇L` 迈步，在病态曲率下极易震荡。
- Adam/RMSProp: 引入二阶矩 `vₜ` 修正尺度。它能感知地形的“颠簸程度”（元不确定性），实现元素级自适应。但其逐元素 (element-wise) 的视角忽略了参数间的相关性，本质上是在做平行的标量优化。

### 1.3 Muon

[`Muon`](optimizer/muon.py) 引入了严格的几何约束：要求更新量必须是“正交”的。

- Stiefel 流形: 更新量 `Δθ` 被投影至 Stiefel 流形（满足 `UᵀU = I` 的矩阵集合）。
- 纯粹旋转: 投影通过 Newton-Schulz 迭代 `𝒫ₛₜ(X)` 实现。这保证了每一步都在改变特征空间的“基向量方向”，而非“模长强度”，从而从根本上消除了内部协变量偏移。

### 1.4 RMSuon

[`RMSuon`](optimizer/rmsuon.py) 提出了第一个解耦方案：

- 几何 (Geometry): 信任 Muon 的正交化动量 `𝒫ₛₜ(mₜ)` 提供的方向稳定性。
- 能量 (Energy): 信任 Adam 的宏观统计。从 Adam 更新量中提取 Frobenius 范数作为标量能量：
  `E = ‖m̂ₜ / (√(v̂ₜ) + ε)‖_F`
- 算子复合: 让正交化的“芭蕾舞步”根据 Adam 观测到的总体“环境能量”进行缩放。

### 1.5 AdaRMSuon

[`AdaRMSuon`](optimizer/ada_rmsuon.py) 进一步揭示了：原始梯度在弯曲流形上存在“几何畸变”。

- 预白化 (Pre-whitening): 并非直接投影动量，而是先用 `vₜ` 对梯度进行白化，获得近似的自然梯度 (Natural Gradient) `gₙₐₜ ≈ mₜ / √(vₜ)`。
- 投影映射: 在预白化后的空间（更接近黎曼平直切空间）执行正交化投影 `𝒫ₛₜ(gₙₐₜ)`。
- 形式化表达:
  `Δθₜ = η ⋅ ‖gₙₐₜ‖_F ⋅ 𝒫ₛₜ(gₙₐₜ)`
- 结论: 这使得模型能够沿着真正的测地线 (Geodesic) 滑行，在 Wikitext-2 实验中表现出断层级的收敛效率。

## 2. 实验对比：Wikitext-2

实验设置: Qwen3 (RoPE), Context 255

### 2.1 5 Epoch 快速测试

> 标准 wikitext2 line mode 实验

| 优化器 | 核心机制 | Best PPL | Final PPL | Grad Norm (End) |
| :--- | :--- | :--- | :--- | :--- |
| AdaRMSuon | Pre-white + NS + Energy | 83.88 | 87.61 | 0.92 |
| RMSuon (v1) | AdamW + NS + Energy | 99.07 | 134.11 | ~3.7 |
| AdaMuon | Sign + NS + Element-wise | 125.46 | 147.60 | ~5.6 |
| Muon | SGD + NS | 161.09 | 161.09 | ~1.1 |
| AdamW | Standard | 104.68 | 250.82 | ~4.5 |

**结论**: 实验结果表明 AdaRMSuon 的性能表现呈现断层式领先。相比之下，AdaMuon 的表现甚至逊于初版 RMSuon，这有力地证明了 `sign(m)` 带来的信息损失以及元素级自适应对流形结构的破坏是致命的。而纯粹的 Muon 由于缺乏自适应能力，其收敛速度明显较慢。

### 2.2 30 Epoch 马拉松：过拟合的动力学

> 此实验使用已清理的 chunk mode wikitext2
> `outputs\wikitext2_rope_muon_epoch30`
> `outputs\wikitext2_rope_rmsuon_epoch30`

| 优化器 | Best PPL (Epoch) | Final PPL (Epoch 30) | 过拟合倍数 | 稳定性分析 |
| :--- | :--- | :--- | :--- | :--- |
| RMSuon | 190.63 (Ep 3) | ~54930 | ~288x | 极速收敛，灾难性过拟合。证明其寻找最小值的效率极高，但完全没有复杂度控制，容易陷入局部极小值。 |
| Muon | 329.99 (Ep 6) | ~587 | ~1.78x | 缓慢收敛，轻微过拟合。其内在的谱约束自带一种隐式正则化，但效率太低。 |

- RMSuon: 第 3 Epoch 即达到最优 PPL (190.6)，随后发生灾难性过拟合（30 Epoch 时 PPL > 50000），Muon后期缓慢过拟合。
- RMSuon 能以最高效的路径找到当前训练集的极小值，但也由于缺乏复杂度控制，容易陷入那些极其狭窄、泛化能力差的尖锐谷底 (Sharp Minima)。

## 3. ARS：引入平坦度约束

[`AdaRMSuon`](optimizer/ada_rmsuon.py) 证明了在黎曼流形上沿着测地线滑行能带来断层级的收敛效率，但也揭示了其本质缺陷：它是一个极其高效的“局部极小值猎手”，会毫不犹豫地钻入那些极其狭窄、泛化能力差的尖锐谷底（Sharp Minima）。

我们需要在滑行的同时，引入**平坦度 (Flatness)** 约束，将参数轨迹推向更宽阔的盆地。

### 3.1 ARS (AdaRMSuon SAM): 流形感知扰动

[`ARS`](optimizer/ars.py) 不在欧氏空间做球形扰动，而是在由二阶矩 `v_hat` 定义的流形度量下计算对抗方向。

**形式化定义**:

1. **流形度量估计**: 利用 Adam 的二阶矩 `v_hat` 近似局部曲率（Fisher 信息矩阵的对角近似）。
2. **自然梯度扰动**:
    `g_nat = ∇L / (√v_hat + ε)`
    `𝜀 = 𝜌 ⋅ g_nat / ‖g_nat‖`
    这相当于在黎曼流形上进行等距扰动，而非参数空间的欧氏扰动。
3. **剪切力注入 (Shear Force Injection)**:
    当 `k > 1` 时，ARS 采用 Lookbehind 策略。它计算对抗梯度 `g_adv` 与基础梯度 `g_base` 的正交分量：
    `v_flat = g_adv - proj_{g_base}(g_adv)`
    并在非同步步骤中将此“平坦度向量”注入梯度流，持续推动模型离开尖锐区域。

### 3.2 实验分析

实验证明，流形感知扰动不仅能抑制过拟合，其强度（由 `𝜌` 控制）对最终泛化性能有显著影响。

**实验表现 (Wikitext-2 Line Mode)**:

| 优化器 | Final PPL | 稳定性分析 |
| :--- | :--- | :--- |
| AdaRMSuon (Baseline) | 87.61 | 存在明显过拟合回升 |
| ARS (`𝜌`=0.05) | 83.70 | 成功抑制过拟合 |
| ARS (`𝜌`=0.1) | 80.94 | 更深度的泛化 |

**结论**: 正确的优化器状态管理是有效结合 SAM 的核心前提。在当前任务中，更大的平坦度半径（如 `𝜌`=0.1）展现出更显著的正则化效应，能够成功引导模型避开尖锐谷底，进入泛化性能更优、更宽阔的贝叶斯盆地。

## 4. 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
- [7] G. Mordido, P. Malviya, A. Baratin, and S. Chandar, "Lookbehind-SAM: k steps back, 1 step forward," in *Proc. 41st Int. Conf. Mach. Learn. (ICML)*, 2024, pp. 36229-36248.
