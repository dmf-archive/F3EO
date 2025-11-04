

class EarlyStop:
    def __init__(self, patience: int = 10, threshold: float = -1.0, mode: str = "min"):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_metric: float | None = None
        self.counter = 0
        self.early_stop = False

        if self.threshold == -1.0:
            self.enabled = False
        else:
            self.enabled = True

    def __call__(self, metric: float) -> bool:
        if not self.enabled:
            return False

        if self.best_metric is None:
            self.best_metric = metric
            return False

        if self.mode == "min":
            improved = metric < (self.best_metric - self.threshold)
        else:
            improved = metric > (self.best_metric + self.threshold)

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

    def reset(self):
        self.best_metric = None
        self.counter = 0
        self.early_stop = False
