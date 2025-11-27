import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Task


def get_task(name: str, config: dict) -> "Task":
    try:
        module = importlib.import_module(f"task.{name}")
        task_class_name = "".join(part.capitalize() for part in name.split('_')) + "Task"
        task_class = getattr(module, task_class_name)
        return task_class(config)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load task '{name}'. Error: {e}") from e
