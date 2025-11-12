import json
from pathlib import Path
from typing import Any

from utils.data import EpochMetric, MetricStore


class MDLogger:
    def __init__(self, config: dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def on_train_end(self, store: MetricStore):
        epoch_history = store.get_flat_epoch_history()
        if not epoch_history:
            return

        report = self._generate_report(epoch_history)
        report_path = self.output_dir / "summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)

    def _generate_report(self, epoch_data: list[EpochMetric]) -> str:

        task_names = sorted(list(set(e.task_name for e in epoch_data)))

        headers = ["Epoch", "Task", "Train Loss", "LR", "PI", "Eff. Gamma", "Entropy", "Grad Norm", "Epoch Time (s)", "Peak GPU Mem (MB)"]

        metric_keys = set()
        for epoch in epoch_data:
            metric_keys.update(epoch.task_metrics.metrics.keys())
        sorted_metric_keys = sorted(list(metric_keys))
        headers.extend([f"Eval {key.capitalize()}" for key in sorted_metric_keys])

        table_header = "| " + " | ".join(headers) + " |"
        table_separator = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"

        table_rows = []
        for data in epoch_data:
            row = f"| {data.global_epoch + 1} "
            row += f"| {data.task_name} "
            row += f"| {data.avg_train_loss:.4f} "
            row += f"| {data.learning_rate:.6f} "
            pi_val = getattr(data, 'avg_pi_obj', None)
            if pi_val is not None:
                row += f"| {pi_val.raw_pi:.3f} "
            else:
                row += f"| {getattr(data, 'avg_pi', 'N/A')} " if getattr(data, 'avg_pi', None) is not None else "| N/A "
            row += f"| {getattr(data, 'avg_effective_gamma', 'N/A')} " if getattr(data, 'avg_effective_gamma', None) is not None else "| N/A "
            row += f"| {data.avg_entropy:.3f} " if data.avg_entropy is not None else "| N/A "
            row += f"| {data.grad_norm:.4f} " if data.grad_norm is not None else "| N/A "
            row += f"| {data.epoch_time_s:.2f} " if data.epoch_time_s is not None else "| N/A "
            row += f"| {data.peak_gpu_mem_mb:.1f} " if data.peak_gpu_mem_mb is not None else "| N/A "

            for key in sorted_metric_keys:
                metric_val = data.task_metrics.metrics.get(key)
                row += f"| {metric_val:.2f} " if isinstance(metric_val, float) else "| N/A "
            row += "|"
            table_rows.append(row)
        table_content = "\n".join(table_rows)

        final_metrics_summary = self._get_final_metrics_summary(epoch_data, task_names)
        best_metric_summary = self._get_best_metric_summary(epoch_data, task_names, sorted_metric_keys)

        report = f"""# F3EO-Bench Experiment Report

## Configuration Summary
```json
{json.dumps(self.config, indent=2)}
```

## Training Results
{table_header}
{table_separator}
{table_content}

## Performance Summary
- **Best Validation Metrics**: {best_metric_summary}
- **Final Validation Metrics**: {final_metrics_summary}
"""
        return report

    def _get_best_metric_summary(self, epoch_data: list[EpochMetric], task_names: list[str], metric_keys: list[str]) -> str:
        summary = []
        for name in task_names:
            for key in metric_keys:
                is_ppl = 'perplexity' in key
                metrics = [e.task_metrics.metrics.get(key) for e in epoch_data if e.task_name == name and e.task_metrics.metrics.get(key) is not None]
                if not metrics: continue
                best_val = min(metrics) if is_ppl else max(metrics)
                summary.append(f"{name} {key.capitalize()}: {best_val:.2f}")
        return ", ".join(summary)

    def _get_final_metrics_summary(self, epoch_data: list[EpochMetric], task_names: list[str]) -> str:
        summary = []
        for name in task_names:
            last_epoch_for_task = max([e for e in epoch_data if e.task_name == name], key=lambda x: x.global_epoch)
            summary.append(f"{name}: {json.dumps(last_epoch_for_task.task_metrics.metrics)}")
        return ", ".join(summary)
