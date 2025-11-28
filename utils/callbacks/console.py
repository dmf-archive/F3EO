from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .base import Callback

if TYPE_CHECKING:
    from .context import TrainerContext


class ConsoleLogger(Callback):
    def __init__(self):
        self.console = Console()
        self.progress: Progress | None = None
        self.step_task_id: TaskID | None = None

    def on_train_end(self, context: "TrainerContext"):
        if self.progress:
            self.progress.stop()
        self.console.print("\n[bold green]Training completed![/bold green]")

    def on_train_begin(self, context: "TrainerContext"):
        model_name = context.config['model'].get('type', context.config['model'].get('arch', 'unknown'))
        self.console.print(Panel.fit(
            f"[bold cyan]Tasks:[/bold cyan] {', '.join(context.config['experiment']['tasks'])}\n"
            f"[bold cyan]Model:[/bold cyan] {model_name}\n"
            f"[bold cyan]Optimizer:[/bold cyan] {context.config['optimizer']['name']}\n"
            f"[bold cyan]Epochs:[/bold cyan] {context.total_epochs}\n"
            f"[bold cyan]Device:[/bold cyan] {context.device}\n"
            f"[bold cyan]Output:[/bold cyan] {context.output_dir}",
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

    def on_epoch_begin(self, context: "TrainerContext"):
        if not self.progress:
            raise RuntimeError("Progress bar not initialized. `on_train_begin` must be called first.")
        self.step_task_id = self.progress.add_task(f"Epoch {context.current_epoch + 1}", total=context.total_steps_in_epoch)

    def on_step_end(self, context: "TrainerContext"):
        if not self.progress or self.step_task_id is None:
            raise RuntimeError("Progress bar not initialized for step. `on_epoch_begin` must be called first.")
        self.progress.update(self.step_task_id, advance=1)

    def on_epoch_end(self, context: "TrainerContext"):
        if self.progress and self.step_task_id is not None:
            self.progress.remove_task(self.step_task_id)
            self.step_task_id = None

        last_metric = context.store.get_latest_epoch_for_task(context.current_task_name)
        if not last_metric:
            return

        table = Table(title=f"Epoch {last_metric.global_epoch + 1} Results")
        table.add_column("Task", style="cyan")
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column("Eval Metric", justify="right", style="green")

        task_name = last_metric.task_name
        metrics_dict = last_metric.task_metrics.metrics
        metric_key = "perplexity" if "wikitext2" in task_name else "accuracy"
        metric_val = metrics_dict.get(metric_key, "N/A")
        metric_str = f"{metric_val:.2f}" if isinstance(metric_val, float) else "N/A"
        if metric_key == "accuracy":
            metric_str += "%"

        table.add_row(task_name, f"{last_metric.avg_train_loss:.4f}", metric_str)
        self.console.print(table)

        pi_val = getattr(last_metric, 'avg_pi_obj', None)
        pi_str = f"{pi_val.raw_pi:.3f}" if pi_val else "N/A"
        grad_norm_str = f"{last_metric.grad_norm:.4f}" if last_metric.grad_norm is not None else "N/A"
        self.console.print(f"[dim]LR: {last_metric.learning_rate:.6f} | PI: {pi_str} | Grad: {grad_norm_str}[/dim]")

        if hasattr(last_metric, 'diagnostics') and last_metric.diagnostics:
            diag_str = " | ".join([f"{k}: {v:.4f}" for k, v in last_metric.diagnostics.items() if isinstance(v, float)])
            if diag_str:
                self.console.print(f"[dim cyan]Diagnostics: {diag_str}[/dim cyan]")

    def on_step_begin(self, context: "TrainerContext"):
        pass

    def save(self, context: "TrainerContext"):
        pass

    def load(self, context: "TrainerContext") -> bool:
        return False
