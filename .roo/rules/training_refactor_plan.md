# 训练框架原子化解耦计划 (v2.0)

> **Status**: 核心重构提案
> **Goal**: 彻底解耦三大实验项目（CIFAR-10, WikiText-2, Grokking），消除 AWD 遗迹，建立基于 `SmartOptimizer` 的原子化训练引擎。

## 1. 核心哲学：原子化解耦 (Atomic Decoupling)

当前的 `Trainer` 类试图成为“万能引擎”，导致了严重的逻辑耦合。新架构将遵循以下原则：

- **脚本即实验**：每个实验项目拥有独立的启动脚本（如 `exp/cifar/train.py`），负责该项目特定的数据流、指标统计和报告生成。
- **优化器下沉**：所有关于“如何执行一步更新”的知识（闭包、二阶、BN 保护）全部封装在 `SmartOptimizer` 中。
- **共享注册表**：仅保留 `optimizer/` 目录作为唯一的跨项目共享组件。

## 2. 核心组件：`SmartOptimizer`

在 `optimizer/__init__.py` 中实现，作为所有原生 PyTorch 优化器的包装器。

- **职责**:
  - **闭包管理**：自动定义并执行 `closure`，处理 `logits` 和 `loss` 的同步。
  - **BN 状态手术**：利用 `utils.nn` 自动在多步更新（如 SAM/ARS）中保护 `running_stats`。
  - **自动 Backward**：根据优化器元数据（`d_1_step_takes_closure` 等）决定是手动执行 `backward()` 还是交给优化器。
- **接口**: `logits, loss = smart_opt.step(batch, task_fn)`

## 3. 三路实验脚本架构

将现有的 `utils/trainer.py` 逻辑拆解并注入到以下独立脚本中：

### 3.1 `scripts/train_cifar.py` (视觉路径)

- **重点**：Top-1/Top-5 Accuracy，ResNet 架构注册。
- **特性**：极简循环，针对 LRP (Long-Range Plan) 优化的日志记录。

### 3.2 `scripts/train_wiki.py` (语言路径)

- **重点**：Perplexity (PPL)，谱熵 (Spectral Entropy) 监控，RoPE/Qwen 架构。
- **特性**：支持长周期训练下的动力学探针。

### 3.3 `scripts/train_grok.py` (泛化路径)

- **重点**：模加法任务，训练/测试精度对齐，更智能的 Early Stopping 逻辑。

## 4. 模块化工具提取

### 4.1 `utils/nn.py` (BN 保护)

- **函数**: `disable_running_stats(model)`, `enable_running_stats(model)`。
- **目的**: 为 SAM/ARS 提供模型手术工具，防止虚假的前向传播污染 BN 统计量。

### 4.2 废弃组件 (Deprecation)

- **AWD (Adaptive Weight Decay)**：彻底移除 `utils/trainer.py` 中关于 `ipcwd` 和 `pcwd` 的代码。在 SAM 时代，MDL 的优化已由锐度约束接管。

## 5. 实现路线图 (TTP 驱动)

1. **第一阶段：基础设施**
    - 创建 `utils/nn.py`。
    - 在 `optimizer/__init__.py` 中实现 `SmartOptimizer`。
2. **第二阶段：脚本原子化**
    - 以 CIFAR-10 为试点，编写 `exp/cifar-10/`。
    - 验证 `SmartOptimizer` 在 `ARS2-Neo` 下的正确性。
3. **第三阶段：全面迁移**
    - 迁移 WikiText-2 和 Grokking 逻辑至独立脚本。
    - 彻底删除 `utils/trainer.py`等旧系统。

## 6. 优势

- **隔离性**：修改 Wiki 的指标统计绝不会破坏 CIFAR 的训练。
- **可验证性**：审稿人可以清晰地看到每个实验的独立逻辑，无隐藏的全局开关。
- **性能**：移除冗余的 `if/else` 判断，简化训练主循环。
