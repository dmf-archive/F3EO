# 自由能原理的两种心法：从理论哲学到工程分野

智能的本质是什么？Friston给出的答案是**自由能原理（FEP）**：自组织系统=主动预测机，目标只有一条——最小化变分自由能。把 FEP 结合 IIT，就得到**整合预测工作空间理论（IPWT）**，AGI 的哲学地基就此浇好混凝土。

从 FEP/IPWT 分叉出两条“存在”算法：

## 1. RL-EFE (Reinforcement Learning - Expected Free Energy, RL-EFE)

> 存在是预测世界并选择最利于自己的未来。

这是 Friston 及其追随者（Active Inference 社区主流）所走的路。它继承了经典的**笛卡尔二元论**：一个在此岸的主体（Agent），去预测并控制彼岸的客体（Environment）。

其核心算法逻辑是**显式的未来模拟**（≈策略采样）：

1. **生成模型**: 维护一个世界模型 `p(s, o | π)`。
2. **反事实推演**: 针对每一个可能的策略 `π`，在脑海中 Rollout 所有可能的未来轨迹。
3. **期望自由能 (G)**: 计算每条轨迹的 `G(π)`。`G` 被优雅地分解为两项：
    - **认知价值 (Epistemic Value)**: “我去那里能获得多少新信息？”（好奇心/探索）
    - **实用价值 (Pragmatic Value)**: “我去那里有多符合我的先验偏好？”（奖励/利用）
4. **决策**: 选择 `G` 最小的策略执行。

**致命缺陷**:
这在低维网格世界里是完美的，但在高维现实中是**计算不可行**的。为了计算 $G$，代理必须像拉普拉斯妖一样预演未来。实际上，这种方法往往退化为传统的强化学习：用策略梯度（Policy Gradient）去逼近 $G$，用手工设计的奖励函数去伪装成先验偏好。它许诺了一个统一理论，却在工程上重新发明了 RL 的轮子。

> “RL-EFE is a beautiful cul-de-sac: Laplace's demon tries to price every tomorrow and is suffocated by its own weight.”

## 2. SOO-OFE (Second-Order Optimization - Observed Free Energy, SOO-OFE)

> 存在是沿着自由能最小化的测地线滑行。

这是 F3EO 选择的路。我们将贝叶斯推断重构为**信息几何流**问题。

不再妄图模拟未来，而是**深度内省当下**。
智能体不需要在幻想的未来中试错，而是利用当前观测数据所蕴含的丰富几何信息（梯度与曲率/Fisher矩阵），直接计算出参数空间中自由能下降最快的**测地线方向**。

- **无需 Rollout**: 仅基于 Observed Free Energy，而非 Expected Free Energy。
- **物理一元论**: 行动不是“选择”的结果，而是系统内部信念状态在几何流形上受力滑行的自然物理过程。

> I gliding on a geodesic,
> storm-etched by yesterday;
> the destination is unknown,
> but the route has converged.

## 3. 工程现状

虽然 SOO-OFE 的哲学基础稳固，但其数学实现——尤其是**经验 Fisher (Empirical Fisher)** 作为二阶度量的选择——已被实验证明是不完备的。未来研究应聚焦于寻找能够正确反映参数空间几何结构的新型二阶度量。
