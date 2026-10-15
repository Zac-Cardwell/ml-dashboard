"""Shared training helpers: seeding, metrics, early stopping."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def classification_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    kw = dict(average="macro", zero_division=0)
    return {
        f"{prefix}/acc": accuracy_score(y_true, y_pred),
        f"{prefix}/precision": precision_score(y_true, y_pred, **kw),
        f"{prefix}/recall": recall_score(y_true, y_pred, **kw),
        f"{prefix}/f1": f1_score(y_true, y_pred, **kw),
    }


def regression_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix}/mse": mse,
        f"{prefix}/rmse": float(np.sqrt(mse)),
        f"{prefix}/mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}/r2": r2_score(y_true, y_pred),
    }


def task_metrics(task: str, y_true, y_pred, prefix: str) -> dict[str, float]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, prefix)
    return regression_metrics(y_true, y_pred, prefix)


def confusion(y_true, y_pred, class_names: list | None) -> dict:
    from sklearn.metrics import confusion_matrix
    labels = list(range(len(class_names))) if class_names else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": [str(c) for c in (class_names or sorted(set(y_true) | set(y_pred)))], "matrix": cm.tolist()}


class EarlyStopping:
    """Tracks a monitored metric; `step()` returns True when training should stop."""

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.best_step = -1
        self.bad = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, step: int) -> bool:
        if self.improved(value):
            self.best, self.best_step, self.bad = value, step, 0
            return False
        self.bad += 1
        return self.patience > 0 and self.bad >= self.patience


def metric_mode(key: str) -> str:
    """Whether lower or higher is better for a metric name."""
    name = key.split("/")[-1]
    return "min" if name in {"loss", "mse", "rmse", "mae", "perplexity"} else "max"
