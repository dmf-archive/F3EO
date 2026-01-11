# ARS 家族：在黎曼流形上滑行

状态: 生产就绪 (2025-12-31)
核心贡献: 发展了 Energy-Geometry Decoupling 的算子复合范式，并为进一步探索 Geodesic Optimizer 提供了实验结果和工程样例。

## 优化的本质：在测地线上滑行

在信息几何视角下，优化不仅是损失函数 `L(θ)` 的梯度下降，更是概率分布流形上的测地线运动。问题在于：不同的优化器，对地形的假设不同

- SGD: 假设欧几里得平直空间。它是“盲人登山者”，仅凭局部坡度 `∇L` 迈步，在病态曲率下极易震荡。
- Adam/RMSProp: 引入二阶矩 `vₜ` 修正尺度。它能感知地形的“颠簸程度”（元不确定性），实现元素级自适应。但其逐元素 (element-wise) 的视角忽略了参数间的相关性，本质上是在做平行的标量优化。

### Muon

[`Muon`](optimizer/muon.py) 引入了严格的几何约束：要求更新量必须是“正交”的。

- Stiefel 流形: 更新量 `Δθ` 被投影至 Stiefel 流形（满足 `UᵀU = I` 的矩阵集合）。
- 纯粹旋转: 投影通过 Newton-Schulz 迭代 `𝒫ₛₜ(X)` 实现。这保证了每一步都在改变特征空间的“基向量方向”，而非“模长强度”，从而从根本上消除了内部协变量偏移。

### RMSuon

[`RMSuon`](optimizer/rmsuon.py) 提出了第一个解耦方案：

- 几何 (Geometry): 信任 Muon 的正交化动量 `𝒫ₛₜ(mₜ)` 提供的方向稳定性。
- 能量 (Energy): 信任 Adam 的宏观统计。从 Adam 更新量中提取 Frobenius 范数作为标量能量：
  `E = ‖m̂ₜ / (√(v̂ₜ) + ε)‖_F`
- 算子复合: 让正交化根据 Adam 观测到的总体“环境能量”进行缩放。

### AdaRMSuon

[`AdaRMSuon`](optimizer/ada_rmsuon.py) 进一步揭示了：原始梯度在弯曲流形上存在“几何畸变”。

- 预白化 (Pre-whitening): 并非直接投影动量，而是先用 `vₜ` 对梯度进行白化，获得近似的自然梯度 (Natural Gradient) `gₙₐₜ ≈ mₜ / √(vₜ)`。
- 投影映射: 在预白化后的空间（更接近黎曼平直切空间）执行正交化投影 `𝒫ₛₜ(gₙₐₜ)`。
- 形式化表达:
  `Δθₜ = η ⋅ ‖gₙₐₜ‖_F ⋅ 𝒫ₛₜ(gₙₐₜ)`
- 结论: 这使得模型能够沿着真正的测地线 (Geodesic) 滑行，在 Wikitext-2 实验中表现出断层级的收敛效率。

## 实验对比：CIFAR-10

实验设置: ResNet-18, 10 Epochs, Batch Size 256.
作为基础视觉任务的基准测试，我们对比了 Muon 及其变体在 CIFAR-10 上的表现。

| 优化器           | Best Eval Acc | Final Eval Acc | Final Eval Loss | 备注                                     |
| :--------------- | :------------ | :------------- | :-------------- | :--------------------------------------- |
| ARS Sync (ρ=0.1) | 89.97%        | 89.43%         | 0.30            | 性能最优。引入平坦度约束后成功逼近 90%。 |
| RMSuon           | 89.09%        | 89.09%         | 0.34            | 收敛极快，基准表现强劲。                 |
| AdaRMSuon        | 89.09%        | 88.63%         | 0.35            | 存在明显波动与过拟合迹象。               |
| Muon             | 87.05%        | 87.05%         | 0.39            | 稳健但收敛速度慢于 RMSuon 家族。         |

核心洞察:

1. 能量-几何解耦的普适性: RMSuon 家族 (89%+) 全面超越了原始 Muon (87.05%)，证明了解耦“迈步方向”与“迈步强度”在视觉任务中同样关键。
2. AdaRMSuon 的不稳定性: 虽然具备极强的收敛动力，但在 CIFAR-10 上表现出明显的波动和过拟合。说明单纯的“测地线滑行”在缺乏平坦度约束时，容易在后期滑出狭窄的极小值区域。
3. 平坦度约束的必要性: ARS (AdaRMSuon + SAM) 通过在黎曼流形上引入平坦度约束，有效抑制了 AdaRMSuon 的过拟合，实现了收敛速度与泛化能力的双重提升。

## 实验对比：Wikitext-2

实验设置: Qwen3 (RoPE), Context 255

### 5 Epoch 快速测试

> 标准 wikitext2 line mode 实验

| 优化器      | 核心机制                 | Best PPL | Final PPL | Grad Norm (End) |
| :---------- | :----------------------- | :------- | :-------- | :-------------- |
| AdaRMSuon   | Pre-white + NS + Energy  | 83.88    | 87.61     | 0.92            |
| RMSuon (v1) | AdamW + NS + Energy      | 99.07    | 134.11    | ~3.7            |
| AdaMuon     | Sign + NS + Element-wise | 125.46   | 147.60    | ~5.6            |
| Muon        | SGD + NS                 | 161.09   | 161.09    | ~1.1            |
| AdamW       | Standard                 | 104.68   | 250.82    | ~4.5            |

结论: 实验结果表明 AdaRMSuon 的性能表现呈现断层式领先。相比之下，AdaMuon 的表现甚至逊于初版 RMSuon，这有力地证明了 `sign(m)` 带来的信息损失以及元素级自适应对流形结构的破坏是致命的。而纯粹的 Muon 由于缺乏自适应能力，其收敛速度明显较慢。

### 30 Epoch 马拉松：过拟合的动力学

> 此实验使用已清理的 chunk mode wikitext2
> `outputs\wikitext2_rope_muon_epoch30` > `outputs\wikitext2_rope_rmsuon_epoch30`

| 优化器 | Best PPL (Epoch) | Final PPL (Epoch 30) | 过拟合倍数 | 稳定性分析                                                                                     |
| :----- | :--------------- | :------------------- | :--------- | :--------------------------------------------------------------------------------------------- |
| RMSuon | 190.63 (Ep 3)    | ~54930               | ~288x      | 极速收敛，灾难性过拟合。证明其寻找最小值的效率极高，但完全没有复杂度控制，容易陷入局部极小值。 |
| Muon   | 329.99 (Ep 6)    | ~587                 | ~1.78x     | 缓慢收敛，轻微过拟合。其内在的谱约束自带一种隐式正则化，但效率太低。                           |

- RMSuon: 第 3 Epoch 即达到最优 PPL (190.6)，随后发生灾难性过拟合（30 Epoch 时 PPL > 50000），Muon 后期缓慢过拟合。
- RMSuon 能以最高效的路径找到当前训练集的极小值，但也由于缺乏复杂度控制，容易陷入那些极其狭窄、泛化能力差的尖锐谷底 (Sharp Minima)。

## ARS：引入平坦度约束

[`AdaRMSuon`](optimizer/ada_rmsuon.py) 证明了在黎曼流形上沿着测地线滑行能带来断层级的收敛效率，但也揭示了其本质缺陷：它是一个极其高效的“局部极小值猎手”，会毫不犹豫地钻入那些极其狭窄、泛化能力差的尖锐谷底（Sharp Minima）。

我们需要在滑行的同时，引入平坦度 (Flatness) 约束，将参数轨迹推向更宽阔的盆地。

### ARS (AdaRMSuon SAM): 流形感知扰动

[`ARS`](optimizer/ars.py) 不在欧氏空间做球形扰动，而是在由二阶矩 `v_hat` 定义的流形度量下计算对抗方向。

形式化定义:

1. 流形度量估计: 利用 Adam 的二阶矩 `v_hat` 近似局部曲率（Fisher 信息矩阵的对角近似）。
2. 自然梯度扰动:
   `g_nat = ∇L / (√v_hat + ε)`
   `𝜀 = 𝜌 ⋅ g_nat / ‖g_nat‖`
   这相当于在黎曼流形上进行等距扰动，而非参数空间的欧氏扰动。
3. 剪切力注入 (Shear Force Injection):
   当 `k > 1` 时，激活 GSAM (Surrogate Gap Guided SAM) 模式。它将对抗梯度 `g_adv` 分解为平行于基础梯度 `g_base` 的分量（用于降低 Loss）和垂直于 `g_base` 的正交分量 `v_flat`（用于降低 Sharpness）：
   `v_flat = g_adv - proj_{g_base}(g_adv)`
   当 `k > 1` 时（Lazy Mode），ARS 会在非同步步骤中复用并注入此“剪切力”向量，从而在不增加计算量的前提下持续推动模型离开尖锐区域。

### 实验分析

实验证明，流形感知扰动不仅能抑制过拟合，其强度（由 `𝜌` 控制）对最终泛化性能有显著影响。

实验表现 (Wikitext-2 Line Mode):

| 优化器               | Final PPL | 稳定性分析         |
| :------------------- | :-------- | :----------------- |
| AdaRMSuon (Baseline) | 87.61     | 存在明显过拟合回升 |
| ARS (`𝜌`=0.05)       | 83.70     | 成功抑制过拟合     |
| ARS (`𝜌`=0.1)        | 80.94     | 更深度的泛化       |

结论: 正确的优化器状态管理是有效结合 SAM 的核心前提。在当前任务中，更大的平坦度半径（如 `𝜌`=0.1）展现出更显著的正则化效应，能够成功引导模型避开尖锐谷底，进入泛化性能更优、更宽阔的贝叶斯盆地。

### GSAM Lazy Mode 效能分析 (2026-01-03)

针对计算开销较大的问题，我们测试了 ARS 的 Lazy Mode（即 `k > 1` 的剪切力复用模式）。

| 模式 | 参数        | Final PPL | 加速比 | 结论                                                     |
| :--- | :---------- | :-------- | :----- | :------------------------------------------------------- |
| Sync | k=1         | 80.94     | 1.0x   | 性能基准，开销最大。                                     |
| Lazy | k=5, α=0.01 | 85.69     | 1.48x  | 注入强度不足，性能损失明显。                             |
| Lazy | k=3, α=0.01 | 84.71     | 1.37x  | 增加修正频率可提升性能，但加速比下降。                   |
| Lazy | k=5, α=0.1  | 82.10     | 1.49x  | 最优平衡点。通过增强注入强度补偿低频修正，逼近同步性能。 |

核心洞察:

- 强度补偿原理: 在 Lazy Mode 下，由于平坦度向量 `v_flat` 的更新频率降低，必须通过增大注入强度 `alpha` 来维持其对轨迹的纠偏能力。
- 二阶平稳性: 实验证明 `v_flat` 在局部具有足够的几何平稳性，支持 `k=5` 级别的复用。

## 实验验证：Grokking 动力学 (Modular Addition)

为了验证优化器在泛化相变（Phase Transition）中的动力学特征，我们在模加法任务 (`task/mod_addition.py`, `p=113`, `train_frac=0.3`) 上对比了各优化器的表现。模型采用 1-Layer Transformer (4 Heads, d_model=128, d_mlp=512)。

| 优化器        | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) | 状态                                                                    |
| :------------ | :----------- | :----------- | :----------- | :---------------------------------------------------------------------- |
| **AdamW**     | ~140         | 228          | 556          | 标准 Grokking 曲线，存在显著延迟。                                      |
| **AdaRMSuon** | **28**       | **54**       | 300          | **极速 Grokking**。泛化延迟几乎消失，证明测地线滑行能高效穿越损失地形。 |
| **ARS**       | 17           | 100          | 290          | 稳健 Grokking。平坦度约束未阻碍泛化，反而引导至更平坦区域。             |
| **Muon**      | >156         | N/A          | N/A          | 在此特定任务配置下未收敛。                                              |

**核心洞察**:

1. **相变加速**: AdaRMSuon 将 Grokking 发生时间提前了 **4 倍** (Epoch 228 -> 54)，有力证明了“能量-几何解耦”能避免模型在过拟合吸引盆中的无效游走。
2. **平坦度兼容性**: ARS 的成功表明，在流形优化中引入平坦度约束 (SAM) 与快速泛化并不冲突，是通往高效且稳健解的正确路径。

## 有趣事实：命名混乱

在开发过程中，我们发现了一个命名上的有趣事实：

- AdaRMSuon 本身就可以缩写为 ARS
- 而 AdaRMSuon + SAM 本应称为 ARS2

这个混乱源于 RMSuon 是 RMS + Muon 的交错造词，AdaRMSuon 类似地延续了这一命名模式。为消除快速迭代中的识别歧义，现明确：

- ARS：*A*da*R*M*S*uon
- ARS2：*A*da*R*M*S*uon + *S*AM

## ARS2-Neo：重构和整合后的参考版本

ARS2-Neo 是 ARS 家族的集大成者，在统一的代码中实现了 AdaRMSuon 的几何优化与 SAM 的平坦度约束，通过参数配置 (`k`, `rho`) 灵活切换模式，旨在取代实验性的独立 `AdaRMSuon` 和 `ARS`。

### 初步验证：CIFAR-10

我们对 ARS2-Neo 进行了初步的功能一致性与性能验证。

| 模式            | 配置         | Best Eval Acc | Final Eval Acc | Final Eval Loss | 结论                                                               |
| :-------------- | :----------- | :------------ | :------------- | :-------------- | :----------------------------------------------------------------- |
| ARS2-Neo (Sync) | k=1, ρ=0.1   | 90.70%        | 90.70%         | 0.28            | SOTA。超越了旧版 ARS Sync (89.97%)，证明了新实现的正确性与优越性。 |
| ARS2-Neo (Base) | k=0 (No SAM) | 90.13%        | 90.13%         | 0.32            | 表现优于旧版 AdaRMSuon (89.09%)，且极其稳健，无过拟合迹象。        |

下一步计划:
我们将继续在 Wikitext-2 等任务上进行功能完整性和性能测试。待验证完成后，我们将移除旧的 `AdaRMSuon` 和 `ARS` 优化器代码，以简化实验空间并降低研究员的心智压力。

## 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
- [7] J. Zhuang et al., "GSAM: Surrogate Gap Guided Sharpness-Aware Minimization," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*, 2022. [Official PyTorch Implementation](https://github.com/juntang-zhuang/GSAM)
