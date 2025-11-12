from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from utils.data import MetricStore, StepMetric


class ConsoleLogger:
    def __init__(self, config: dict[str, Any]):
        self.console = Console()
        self.config = config
        self.progress = None
        self.step_task_id = None

    def on_train_begin(self, output_dir: str):
        self.console.print(Panel.fit(
            f"[bold cyan]Tasks:[/bold cyan] {', '.join(self.config['experiment']['tasks'])}\n"
            f"[bold cyan]Model:[/bold cyan] {self.config['model']['arch']}\n"
            f"[bold cyan]Optimizer:[/bold cyan] {self.config['optimizer']['name']}\n"
            f"[bold cyan]Epochs:[/bold cyan] {self.config['train']['epochs']}\n"
            f"[bold cyan]Device:[/bold cyan] {self.config['experiment']['device']}\n"
            f"[bold cyan]Output:[/bold cyan] {output_dir}",
            title="[bold]F3EO-Bench Training[/bold]",
            border_style="cyan"
        ))
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        self.progress.start()

    def on_epoch_begin(self, epoch: int, total_steps: int):
        self.step_task_id = self.progress.add_task(f"Epoch {epoch+1}", total=total_steps)

    def on_step_end(self, metric: StepMetric, total_steps: int):
        self.progress.update(self.step_task_id, advance=1)

    def on_epoch_end(self, store: MetricStore):
        if self.step_task_id is not None:
            self.progress.remove_task(self.step_task_id)
            self.step_task_id = None

        flat_history = store.get_flat_epoch_history()
        if not flat_history:
            return

        last_metric = flat_history[-1]

        table = Table(title=f"Epoch {last_metric.global_epoch+1} Results")
        table.add_column("Task", style="cyan")
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column("Eval Metric", justify="right", style="green")

        for epoch_metric in store.get_history_for_task(last_metric.task_name):
            if epoch_metric.global_epoch == last_metric.global_epoch:
                task_name = epoch_metric.task_name
                metrics_dict = epoch_metric.task_metrics.metrics

                metric_key = "perplexity" if "wikitext2" in task_name else "accuracy"
                metric_val = metrics_dict.get(metric_key, "N/A")
                metric_str = f"{metric_val:.2f}" if isinstance(metric_val, float) else "N/A"
                if metric_key == "accuracy":
                    metric_str += "%"

                table.add_row(task_name, f"{epoch_metric.avg_train_loss:.4f}", metric_str)

        self.console.print(table)
        pi_val = getattr(last_metric, 'avg_pi_obj', None)
        if pi_val is not None:
            pi_str = f"{pi_val.raw_pi:.3f}"
        else:
            pi_str = f"{getattr(last_metric, 'avg_pi', 'N/A')}"
        self.console.print(f"[dim]LR: {last_metric.learning_rate:.6f} | PI: {pi_str} | Grad: {last_metric.grad_norm:.4f}[/dim]")

    def on_train_end(self, store: MetricStore):
        self.progress.stop()
        self.console.print("\n[bold green]Training completed![/bold green]")
