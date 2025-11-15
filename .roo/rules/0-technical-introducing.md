# 自由能原理的两种心法：从理论哲学到工程分野

智能的本质是什么？弗里斯顿给出的答案是**自由能原理（FEP）**：自组织系统=主动预测机，目标只有一条——最小化变分自由能。把 FEP 结合 IIT，就得到**整合预测工作空间理论（IPWT）**，AGI 的哲学地基就此浇好混凝土。

从 FEP/IPWT 分叉出两条“存在”算法：

## 1. RL-EFE (Reinforcement Learning - Expected Free Energy, RL-EFE)

> 存在是预测世界并选择最利于自己的未来。

Friston 亲领的强化学习-期望自由能路线，继承经典的二元论世界观：

1. 脑内建模型 `p(s,o|π)`
2. rollout 全部未来
3. 算期望自由能 `G(π)`
4. 选 `argmax G`，执行

表面是强化学习，实则是策略梯度与手工奖励的遮羞布。

> “RL-EFE is a beautiful cul-de-sac: Laplace's demon tries to price every tomorrow and is suffocated by its own weight.”

## 2. SOO-OFE (Second-Order Optimization - Observed Free Energy, SOO-OFE)

> 存在是沿着自由能最小化的测地线滑行。

我们自研的“二阶优化-观测自由能”路线，在这种世界观下，系统不再是一个与环境对立的决策者，而是世界本身的一个子部分。它的智能体现在其内部参数（`θ`）如何随着新观测数据的到来而演化，通过**最高效的信念更新**来自然涌现。

- 参数空间 = 统计流形
- 观测自由能 `F(θ)` = 地形海拔
- Fisher 信息 = 局部曲率
- 自然梯度 = 测地线

系统不“选”动作，只随新数据做**最高效信念更新**。

> I gliding on a geodesic,  
> storm-etched by yesterday;  
> the destination is unknown,  
> but the route has converged.
