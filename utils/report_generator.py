import json
import time
from pathlib import Path
from typing import Any


class ReportGenerator:
    def __init__(self, config: dict[str, Any], output_dir: Path, preload_history: dict[str, list[Any]] | None = None):
        self.config = config
        self.output_dir = output_dir
        self.metrics_history = {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "train_metric": [],
            "valid_metric": [],
            "learning_rate": [],
            "epoch_time": [],
            "log_pi": []
        }
        if preload_history is not None:
            for key in self.metrics_history:
                if key in preload_history:
                    self.metrics_history[key].extend(preload_history[key])
        self.start_time = time.time()
        self.task_type = config["experiment"]["task"]

    def log_epoch(self, epoch_num: int, train_results: dict[str, float],
                  valid_results: dict[str, float], lr: float, epoch_time: float, log_pi: float | None = None):
        self.metrics_history["epoch"].append(epoch_num) # Change to epoch_num
        self.metrics_history["train_loss"].append(train_results["loss"])
        self.metrics_history["valid_loss"].append(valid_results["loss"])
        self.metrics_history["learning_rate"].append(lr)
        self.metrics_history["epoch_time"].append(epoch_time)
        self.metrics_history["log_pi"].append(log_pi) 

        if self.task_type == "wikitext2":
            self.metrics_history["train_metric"].append(train_results["perplexity"])
            self.metrics_history["valid_metric"].append(valid_results["perplexity"])
        else:
            self.metrics_history["train_metric"].append(train_results["accuracy"])
            self.metrics_history["valid_metric"].append(valid_results["accuracy"])

    def generate_summary(self) -> str:
        total_time = time.time() - self.start_time
        epochs = len(self.metrics_history["epoch"])

        if self.task_type == "wikitext2":
            best_metric = min(self.metrics_history["valid_metric"])
            final_metric = self.metrics_history["valid_metric"][-1]
            metric_name = "Perplexity"
        else:
            best_metric = max(self.metrics_history["valid_metric"])
            final_metric = self.metrics_history["valid_metric"][-1]
            metric_name = "Accuracy (%)"

        # 生成 markdown 表格
        table_rows = []
        for i in range(epochs):
            # 使用 history 中的 epoch 编号，而不是循环索引
            row = f"| {self.metrics_history['epoch'][i]} | "
            row += f"{self.metrics_history['train_loss'][i]:.4f} | "
            row += f"{self.metrics_history['valid_loss'][i]:.4f} | "
            row += f"{self.metrics_history['train_metric'][i]:.2f} | "
            row += f"{self.metrics_history['valid_metric'][i]:.2f} | "
            row += f"{self.metrics_history['learning_rate'][i]:.6f} | "
            log_pi_val = self.metrics_history['log_pi'][i]
            if log_pi_val is not None:
                row += f"{log_pi_val:.3f} | "
            else:
                row += f"N/A | "
            row += f"{self.metrics_history['epoch_time'][i]:.2f}s |"
            table_rows.append(row)

        table_content = "\n".join(table_rows)

        report = f"""# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | {self.config['experiment']['task']} |
| Model | {self.config['model']['arch']} |
| Optimizer | {self.config['optimizer']['name']} |
| Learning Rate | {self.config['optimizer']['lr']} |
| Weight Decay | {self.config['optimizer']['weight_decay']} |
| Epochs | {self.config['train']['epochs']} |
| Batch Size | {self.config['data']['batch_size']} |
| Device | {self.config['experiment']['device']} |
| Seed | {self.config['experiment']['seed']} |

## Training Results
| Epoch | Train Loss | Valid Loss | Train {metric_name} | Valid {metric_name} | Learning Rate | Log(PI) | Time |
|-----|--|-----|-----|-----|-----|--------|------|
{table_content}

## Performance Summary
- **Best Validation {metric_name}**: {best_metric:.2f}
- **Final Validation {metric_name}**: {final_metric:.2f}
- **Total Training Time**: {total_time:.2f}s
- **Average Epoch Time**: {sum(self.metrics_history['epoch_time'])/epochs:.2f}s

## Configuration Details
```toml
{json.dumps(self.config, indent=2)}
```
"""

        # 保存报告
        report_path = self.output_dir / "summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)

        return report
