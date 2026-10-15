"""Example: a model plugin with its OWN training loop (full control).

Use this pattern for anything the built-in loops don't cover (GANs, custom losses,
other libraries such as XGBoost / LightGBM / JAX). Inside train():
  * ctx.params          the values from the form (already typed and defaulted)
  * ctx.seed, ctx.device, ctx.run_dir
  * ctx.log(step, {...}) record curves, keys like 'train/loss', 'val/acc'
  * ctx.log_artifact(name, kind, data)  confusion_matrix | table | text | json
  * ctx.set_progress(fraction, text)
  * ctx.should_stop()   True when the user pressed Stop
Return a dict of final metrics (shown in the summary table and aggregated across seeds).
"""
import numpy as np

from mltrain.plugins import ModelPlugin, Param
from mltrain.training.utils import EarlyStopping, task_metrics


class NumpyLogReg(ModelPlugin):
    id = "example.numpy_logreg"
    name = "NumPy logistic regression (example)"
    description = "Hand-written softmax regression with mini-batch SGD"
    framework = "custom"
    tasks = ("classification",)
    modalities = ("tabular",)

    def params(self):
        return [Param("epochs", "int", 100, min=1, group="Training"),
                Param("lr", "float", 0.1, min=1e-6, log=True, group="Training"),
                Param("batch_size", "int", 32, min=1, group="Training"),
                Param("l2", "float", 1e-4, min=0.0, group="Training"),
                Param("patience", "int", 10, "Early stopping patience on val/loss (0 = off)", 0, group="Training")]

    def train(self, data, ctx):
        p = ctx.params
        (X, y), (Xv, yv), (Xt, yt) = data.train, data.val, data.test
        k = data.info["n_classes"]
        rng = np.random.default_rng(ctx.seed)
        W, b = rng.normal(0, 0.01, (X.shape[1], k)), np.zeros(k)

        def forward(X):
            z = X @ W + b
            z -= z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)

        def loss(X, y):
            return float(-np.log(forward(X)[np.arange(len(y)), y] + 1e-12).mean())

        stopper = EarlyStopping(p["patience"], "min")
        for epoch in range(1, p["epochs"] + 1):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), p["batch_size"]):
                bi = idx[i:i + p["batch_size"]]
                P = forward(X[bi])
                P[np.arange(len(bi)), y[bi]] -= 1
                W -= p["lr"] * (X[bi].T @ P / len(bi) + p["l2"] * W)
                b -= p["lr"] * P.mean(0)
            row = {"train/loss": loss(X, y), "val/loss": loss(Xv, yv),
                   **task_metrics("classification", y, forward(X).argmax(1), "train"),
                   **task_metrics("classification", yv, forward(Xv).argmax(1), "val")}
            ctx.log(epoch, row)
            ctx.set_progress(epoch / p["epochs"], f"epoch {epoch}")
            if stopper.step(row["val/loss"], epoch) or ctx.should_stop():
                break
        return {"epochs_trained": epoch, **task_metrics("classification", yt, forward(Xt).argmax(1), "test")}
