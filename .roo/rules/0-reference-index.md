# Reference Index

本索引旨在为 `ref/` 目录下的外部研究资料提供导航，明确其在 F3EO 项目中的理论定位与工程价值。

## 1. 架构与机制 (Architecture & Mechanism)

### [`ref/gated-attention-qwen/`](ref/gated-attention-qwen/)

- **核心理论**: 后注意力门控 (Post-SDPA Gating)。
- **关键洞察**: 通过在 SDPA 后引入头特定的乘法 Sigmoid 门控，打破低秩线性瓶颈，引入稀疏性并消除注意力沉溺 (Attention Sink)。

## 2. 优化理论与算法 (Optimization Theory & Algorithms)

### [`ref/LwGN/`](ref/LwGN/)

- **核心理论**: 全高斯-牛顿 (Full Gauss-Newton) 预处理的性能上限。
- **关键洞察**: 层级 Hessian 结构足以捕获大部分二阶增益；高阶损失项对收敛速度非关键。

### [`ref/muon/`](ref/muon/)

- **核心理论**: 隐藏层正交化优化。
- **关键洞察**: 对隐藏层权重执行 Newton-Schulz 正交化更新，能显著提升大批次训练的样本效率。
- **工程价值**: F3EO 结构算子的直接来源，提供了 `Muon` 优化器的标准实现参考。

### [`ref/shampoo/`](ref/shampoo/)

- **核心理论**: 分布式二阶预处理。
- **关键洞察**: 通过 Kronecker 积分解近似 Hessian，平衡计算开销与二阶信息获取。
- **工程价值**: 作为 F3EO 对标的高性能二阶基线，提供了分布式预处理器的工程实现范式。

### [`ref/Hadron.md`](ref/Hadron.md)

- **核心理论**: 算子复合的二阶优化 (KFAC + Muon)。
- **关键洞察**: 拒绝 Hessian-Fisher 谬误，通过算子复合在流形间建立连续映射。
- **工程价值**: 验证了算子复合范式的有效性，后被更鲁棒的 `RMSuon` 取代。

## 3. 学习动力学与现象学 (Learning Dynamics & Phenomenology)

### [`ref/continual-learning/`](ref/continual-learning/)

- **核心理论**: 增量学习的三种类型与 Fisher 信息计算。
- **关键洞察**: 探讨了在任务切换时如何通过 Fisher 信息保护关键权重。
- **工程价值**: 为 F3EO 的持续学习任务（如 `fashion_cl.py`）提供了实验框架与基线算法（EWC, SI, FROMP）。

### [`ref/Omnigrok/`](ref/Omnigrok/)

- **核心理论**: 损失地形视角下的 Grokking 现象。
- **关键洞察**: 泛化（Grokking）与权重范数、损失地形的平坦度密切相关。
- **工程价值**: 提供了研究模型从过拟合向泛化转变的实验工具，指导 F3EO 在长周期训练中的稳定性设计。

## 4. 杂项 (Miscellaneous)

- [`ref/grokking.md`](ref/grokking.md): 关于泛化突变的早期笔记。
- [`ref/Adafisher/`](ref/Adafisher/): 外部库源码，用于验证 Fisher 近似的实现细节。
