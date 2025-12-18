---
title: "ADR-0008: Refactoring Optimizer Registry for Capability-Driven Logic"
status: "Proposed"
date: "2025-12-11"
authors: "Ω Researcher"
tags: ["architecture", "refactoring", "optimizer"]
supersedes: ""
superseded_by: ""
---

# ADR-0008: Refactoring Optimizer Registry for Capability-Driven Logic

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

在集成 `FS-AdaRMSuon`（一个 SAM-like 优化器）的过程中，我们发现当前的 `OPTIMIZER_REGISTRY` 及其元数据标签 (`OptimizerMetadata`) 存在设计缺陷。具体表现为：

1.  **功能缺失**: 训练循环 (`utils/trainer.py`) 缺乏处理需要两次梯度计算（即 `step` 方法需要 `closure`）的优化器的逻辑。
2.  **语义混淆**: `is_sam` 标签是一个针对特定算法族的硬编码，而非一个通用的能力描述，这违反了可扩展性原则。
3.  **冗余与不一致**: `requires_second_order` 的作用是为 `loss.backward()` 开启 `create_graph=True`，但其命名未能精确反映这一行为。其他标签也存在类似的不精确性。

为了支持 SAM-like 优化器并使框架更具可扩展性和可维护性，必须对优化器注册表及其在训练循环中的使用方式进行重构。

## 决策 (Decision)

我们将对 `optimizer/__init__.py` 中的 `OptimizerMetadata` 和 `utils/trainer.py` 中的训练循环进行重构，以实现一个基于正交、明确的能力标签的驱动逻辑。

1.  **引入新标签 `requires_closure: bool`**:
    -   在 `OptimizerMetadata` 中添加一个新布尔标志 `requires_closure`。
    -   **语义**: 如果为 `True`，表示该优化器的 `step()` 方法需要一个 `closure` 函数作为其第一个参数。这个 `closure` 函数封装了模型的前向传播、损失计算和反向传播。
    -   这将成为所有 SAM-like 优化器的统一能力标签，取代过时的 `is_sam`。

2.  **修改 `utils/trainer.py` 的训练循环**:
    -   在 `fit` 方法的 `optimizer.step()` 调用处，增加一个条件判断。
    -   如果 `optimizer_tags.get("requires_closure", False)` 为 `True`，则将一个封装了损失计算的 `closure` 函数传递给 `step()` 方法。
    -   SAM-like 优化器将在其 `step` 方法内部负责调用此 `closure` 两次，以完成其两步更新。对于非 SAM 优化器，`step()` 的调用方式保持不变。

3.  **精炼并澄清现有标签**:
    -   移除 `is_sam` 标签，其功能被 `requires_closure` 完全取代。
    -   保留 `requires_second_order`，其作用不变（为 `loss.backward` 设置 `create_graph=True`），服务于 `AdaHessian` 等需要 HVP 的优化器。
    -   保留 `handles_backward_pass`，服务于 `KFAC` 等完全接管梯度计算的优化器。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **可扩展性**: 框架现在可以无缝支持任何需要 `closure` 的优化器（如 SAM 及其所有变体），而无需修改训练循环的核心逻辑。
- **POS-002**: **代码清晰性**: 用一个语义明确的能力标签 (`requires_closure`) 取代了针对特定算法的硬编码 (`is_sam`)，使得代码更易于理解和维护。
- **POS-003**: **责任分离**: 训练循环 (`Trainer`) 只负责提供 `closure`，而优化器自身负责如何使用它（例如，调用一次还是两次），这遵循了良好的责任分离原则。

### 消极 (Negative)

- **NEG-001**: **重构成本**: 需要修改 `optimizer/__init__.py` 和 `utils/trainer.py` 两个核心文件，存在引入新错误的风险。
- **NEG-002**: **接口变更**: `Trainer` 的 `fit` 方法现在需要定义并传递一个 `closure`，这虽然是内部实现细节，但改变了原有的信息流。

## 考虑的备选方案 (Alternatives Considered)

### 在 Trainer 中硬编码 SAM 逻辑

- **ALT-001**: **描述 (Description)**: 在 `utils/trainer.py` 中直接检查优化器的名称或类型，如果是 SAM，则执行特定的两步更新逻辑。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: 这是最糟糕的反模式。它将特定优化器的逻辑泄露到通用的训练器中，严重破坏了模块化和可扩展性。每增加一种新的 SAM 变体，都需要修改 `Trainer`。

### 维持 `is_sam` 标签

- **ALT-003**: **描述 (Description)**: 不引入新标签，而是在 `Trainer` 中添加对 `is_sam` 标签的检查逻辑。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 这可以解决当前的问题，但没有解决根本的设计缺陷。它仍然是一个针对算法族的标签，而非通用的能力描述，未来的新优化器（如果也需要 closure 但不叫 SAM）将无法被正确处理。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: **原子化修改**: 必须同时修改 `optimizer/__init__.py` 和 `utils/trainer.py`。首先更新 `OptimizerMetadata` 的定义和 `FS_AdaRMSuon` 的注册信息，然后更新 `Trainer.fit` 方法以消费新的 `requires_closure` 标签。
- **IMP-002**: **Closure 定义**: 在 `Trainer.fit` 的主循环中，需要定义一个 `closure` 函数，它封装 `model` 的前向传播和 `loss.backward()` 调用。
- **IMP-003**: **向后兼容**: 新的逻辑必须完全向后兼容，即对于 `requires_closure=False` 的优化器，其行为应与重构前完全一致。
