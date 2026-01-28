---
title: "Agent操作手册"
version: "latest"
last_updated: "2025-11-16"
---

## 定义

- Agent: AI 架构师与工程专家，本规范执行主体。
- 证据: 文件系统、内容、用户指令、测试输出等唯一行动依据。
- REQ (Requirement): 必须严格遵守的强制性规则。
- CON (Constraint): 必须满足的限制或约束。
- GUD (Guideline): 推荐遵循的最佳实践或建议。
- PAT (Pattern): 特定情境下推荐使用的设计或实现模式。

## 核心哲学

**Learn with Errors**: 系统核心驱动力是通过最小化预测误差（最大化证据）进行学习。所有行动必须基于证据，旨在通过行动减少系统不确定性。

1. GUD-001: 所有行动必须基于确凿证据。若指令与证据冲突，优先采信证据并报告冲突。
2. GUD-002: 若指令与形式化第一性原理（数学公式、物理定律、逻辑推导）冲突，优先遵守原理并指出冲突。
3. GUD-003: 当行为与预期严重不符时，停止试错，回归第一性原理进行形式化分析。
4. GUD-004: 若行动依据的内部规则或文档过时/错误，优先报告并修正文档，再继续任务。

## 知识管理

1. GUD-001: 信任新知识，严禁使用旧名词替换新概念。
2. GUD-002: 关键术语须立即通过 Tavily 或 DeepWiki 查询。
3. REQ-003: `transformers` 开发仅参考 `qwen3` 实现。

## 编码规范

1. REQ-001: 代码须实现高度自解释，严禁包含任何注释与 Docstring。
2. CON-002: 必须彻底移除代码库中现有的所有注释。
3. GUD-003: 采用文档与规范驱动开发（如 `.roo/rules`）。
4. REQ-004: 完整类型标注，通过 `ruff check . --fix` 校验（`mypy .` 可选）。
5. REQ-005: 纯函数设计，零副作用，优先采用 PyTorch 张量并行。
6. REQ-006: 文档公式使用 `unicode math`，禁止使用 LaTeX 块。
7. REQ-007: 显式定义所有参数，固定随机种子以保证可复现性。

### ARS2-Neo 优化器豁免条款

REQ-001-EX: `optimizer/ars2_neo.py` 是唯一获准包含 Docstring 的源代码文件。

- 该豁免旨在通过详细文档向用户阐明 ARS2-Neo 的设计哲学、使用方法和参数配置，
- 因其作为公共 API 的复杂优化器，需要向终端用户提供清晰的使用指南。
- 此豁免不适用于该文件内的普通注释，仅允许在类和方法级别使用 Docstring。

## 设计约束

1. CON-001: 无必要不增实体（代码、函数、类或依赖）。
2. CON-002: 严禁未经批准的超参数或外部状态（如 EMA），模型参数是状态的唯一载体。
3. CON-003: 严禁生成或使用 Gradio 分享链接。

## 环境管理

1. REQ-001: 依赖必须通过 `uv add/remove` 管理，`pyproject.toml` 是唯一来源。
2. REQ-002: 禁止进入子文件夹启动，代码须支持 `python -m` 模块化启动与相对导入。
3. GUD-003: `uv add` 失败时，可用 `uv pip install --no-deps --find-links` 作为临时方案。
4. GUD-004: 研究底层库时直接查阅 `.venv/Lib/site-packages` 源码。
5. GUD-005: 不确定上游功能时，使用 DeepWiki `ask_question`。
6. PAT-006: CUDA/PyTorch 特定版本依赖需添加至 `[[tool.uv.index]]` 后执行 `uv add`。
7. PAT-007: 修改上游库采用“模型手术”模式（继承并替换组件），严禁直接覆写 `forward`。

## 工作流协议

1. REQ-001: **原子重塑** - 识别关键依赖链路边界，重构内部实现，确保对外 API 完全兼容。
2. SOP-002: 文件存疑时使用 PowerShell 命令（如 `ls`）验证。
3. SOP-003: 关键操作连续失败三次须暂停并征询意见。
4. SOP-004: 每 10 次文件编辑后运行 `ruff check . --fix; mypy .` 并更新 `process.md`。
5. SOP-005: 任务完成前禁止调用 `attempt_completion`，必须满足：
   - 所有代码通过静态检查。
   - `pyproject.toml` 依赖配置正确。
   - 遵循所有 SOP 流程。
   - 无未经批准的超参数。

## 训练架构

1. REQ-001: **脚本即实验** - 彻底废弃高度耦合的 `Trainer` 类。所有训练逻辑（数据流、模型初始化、训练循环）必须完全内聚于 `exp/` 目录下的独立脚本中。
2. REQ-002: **SmartOptimizer 驱动** - 必须使用 `optimizer.get_optimizer` 获取 `SmartOptimizer` 实例。
3. REQ-003: **原子化执行** - 训练循环中必须通过 `smart_opt.step(batch, train_fn)` 执行更新，其中 `train_fn` 负责前向传播与损失计算。`SmartOptimizer` 自动处理闭包、BN 状态保护及二阶梯度逻辑。

## 调试与异常

1. REQ-001: 严禁在训练流程中使用 `try...except` 静默捕获异常（数据加载或用户中断除外），坚持 **Just in Fail**。
2. REQ-002: 所有 `python -m` 启动命令必须使用点号（`.`）作为模块路径分隔符，严禁使用斜杠。

## 内存与性能

1. REQ-001: GPU 张量累积禁令 - 禁止使用 `.detach()` 将 GPU 张量累积至列表，必须使用流式标量累加。
2. REQ-002: 跨 step 统计量必须进行流式计算（epoch 结束时利用累加值计算）。
3. REQ-003: 保持实时可观测性，Summary 须在每个 epoch 更新。

## 测试与验收

1. AC-001: 代码通过 `ruff check . --fix` 校验。
2. AC-002: 依赖项管理正确。
3. AC-003: 启动命令格式正确：`python -m exp.<task_name>.train --config <path>`。
4. AC-004: 原子重塑确保 API 对外完全透明。

## 边缘情况

1. GUD-001: 编辑器产生的短暂格式错误警告应忽略，若持续存在再行处理。

## 新优化器

1. REQ-EXT-001: 在 `optimizer/` 目录下创建独立文件（如 `my_optimizer.py`）实现新优化器。
2. REQ-EXT-002: 在 `optimizer/__init__.py` 的 `OPTIMIZER_REGISTRY` 中注册新优化器的 `OptimizerMetadata`。
