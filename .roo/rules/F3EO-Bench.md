# F3EO-Bench: 轻量级三阶优化器评测框架

> 目标：用别人已经跑通的数据，只换优化器，就能给出可信速度/遗忘曲线

---

## 1. 边界与目标（Scope）

| 维度| 本次保留| 本次剔除  |
| --- | -- | --- |
| **任务**   | CIFAR-10 单任务速度<br>WikiText-2 单任务速度<br>MNIST → FashionMNIST 持续学习 | ARC-AGI（算力/代码复杂度超限） |
| **优化器** | F3EO（主）<br>AdamW（必须）<br>AdaHessian（二阶对照）    | Shampoo / KFAC（依赖过重）     |
| **硬件**   | RTX 2070 8 GB / Colab P100 16 GB    | 多卡 & 大 batch  |
| **指标**   | 收敛步数、最终精度、遗忘率、终端实时曲线   | 分布式、ImageNet、大模型|
| **自动化** | 一键跑完三类实验<br>markdownlized实验日志 + 终端 Rich 面板    | 人工调参、手动绘图      |

## 2. 目录结构（参考 adafisher 极简风）

```
f3eo-bench/
├── README.md      # 30 秒上手命令
├── pyproject.toml # 只留 torch rich tqdm
├── optimizer/
│   ├── __init__.py
│   ├── f3eo.py    # 三阶核心，<150 行
│   └── adahessian.py  # 二阶对照，直接抄官方
├── model/
│   ├── __init__.py
│   ├── resnet.py # ResNet-18
│   ├── vit.py   # Transformer
│   └── nano-gpt.py   # MNIST/Fashion 头
├── data/ # 自动创建的数据集缓存
├── config/
│   ├── cifar10.toml
│   ├── wikitext2.toml
│   └── cl_stream.toml
├── task/ # 不同任务具体的训练调度器
│   ├── cifar10.py
│   ├── wikitext2.py
│   └── cl_stream.py   # 持续学习 MNIST→Fashion
├── outputs/# gitignored，自动生成
│   ├── report/      # markdown report
│   └── checkpoints/   # 只存 best.pt
└──  scripts/
   ├── train.py   # 统一入口、rich log print
   └── notebook/
      └── loss_landscape.ipynb  # 损失地形可视化（参考 adafisher）
```

## 3. 技术契约

1. **零手工调参**  
   所有实验使用同级配置文件，学习率、batch、epoch 固定；只换优化器类名。
2. **单卡 8 GB 上限**  
   默认 bf16 native，batch 自动回退到不 OOM。
3. **可复现**  
   全局 `seed=42`，PyTorch 确定性卷积；日志输出完整命令与环境哈希。
4. **终端可视化**  
   每 10 步刷新 Rich 表格：loss、acc、lr、GPU-Mem、F3EO-grad-norm。
5. **一键报告**  
   跑完自动生成 `outputs/summary.md`：
   - step级别print log markdownlized转录
   - 最终验证指标
   - 遗忘率（CL 任务）

## 4. 实验流水线（3 个脚本，一条命令）

```bash
python -m scripts/train.py --config config/cifar10.toml
```

1. **CIFAR-10**  
   ResNet-20，200 epoch，SGD→AdamW→F3EO→AdaHessian，记录 wall-time & 最佳 top-1。
2. **WikiText-2**  
   2-layer Transformer，epoch=10，batch=16，记录 perplexity vs. step。
3. **CL Stream**  
   顺序 MNIST→FashionMNIST，每任务 5 epoch，记录最终平均遗忘率。

## 5. 配置示例（`config/cifar10.toml`）

```toml
[experiment]
task = "cifar10"
seed = 42
device = "cuda"

[model]
arch = "resnet20"
num_classes = 10

[data]
batch_size = 128   # 自动减半直到不 OOM
num_workers = 4

[optimizer]
# 只改这一行即可切换
name = "F3EO"  # 可选 AdamW / AdaHessian / F3EO
lr = 0.1
weight_decay = 5e-4

[scheduler]
milestones = [60, 120, 160]
gamma = 0.2

[train]
epochs = 200
log_every = 10
ckpt_every = 50
```

## 6. 输出样例（终端 Rich 面板）

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Optimizer     ┃ Loss   ┃ Top-1  ┃ GradNorm ┃ GPU-Mem ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ F3EO   │ 0.142  │ 94.7 % │ 2.31e-2  │ 7.3 GB  │
│ AdamW  │ 0.156  │ 94.1 % │ 1.85e-2  │ 7.3 GB  │
└───────────────┴────────┴────────┴──────────┴─────────┘
```

## 7. 交付 checklist

- [ ] `uv run scripts/run_all.sh` 在 RTX 2070 8 GB 上一小时内跑完三类实验
- [ ] 自动生成 `outputs/summary.md` 含加速比 & 遗忘率表格
- [ ] 终端 Rich 实时面板无额外依赖（已内置 rich）
- [ ] 所有代码 ≤ 500 行，单文件即可读
- [ ] 开源许可证 AGPL-3.0（与 Tiny-ONN 保持一致）

---

> 一句话打动用户：  
> “把 `optimizer = AdamW` 换成 `optimizer = F3EO()`，**CIFAR-10 收敛快 1.5×，遗忘降 30 %**，8 GB 单卡一键复现。”
