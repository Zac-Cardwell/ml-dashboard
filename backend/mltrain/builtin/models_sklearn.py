"""scikit-learn models. Subclass `SklearnModel` and implement `build()` for your own."""
from __future__ import annotations

import inspect
import time

import numpy as np

from ..plugins.base import ModelPlugin, Param
from ..plugins.registry import register
from ..training.utils import confusion, task_metrics


class SklearnModel(ModelPlugin):
    """Base class: fits an estimator, logs train/val/test metrics.

    Iterative estimators that expose `loss_curve_` (MLP) or `staged_predict` (boosting)
    also get per-iteration curves (and per-stage decision-boundary snapshots).
    """
    framework = "sklearn"
    modalities = ("tabular",)
    tasks = ("classification",)
    supports_init = True
    supports_eval = True

    def build(self, params: dict, task: str, seed: int):
        raise NotImplementedError

    def _continue_from(self, est_new, ctx):
        """Fine-tune: load the saved estimator and keep training it when it supports warm_start."""
        import joblib
        est = joblib.load(ctx.init_path / "model.joblib")
        if "warm_start" in est.get_params():
            grow = {k: v for k, v in est_new.get_params().items() if k in ("n_estimators", "max_iter", "learning_rate",
                                                                             "learning_rate_init", "alpha", "subsample")}
            est.set_params(warm_start=True, **grow)
            ctx.print(f"continuing training of the saved estimator (warm_start) with {grow}")
            return est
        ctx.print(f"{type(est).__name__} has no warm_start; fitting a fresh estimator instead")
        return est_new

    def train(self, data, ctx):
        (Xtr, ytr), (Xva, yva), (Xte, yte) = data.train, data.val, data.test
        est = self.build(ctx.params, data.task, ctx.seed)
        if ctx.init_path:
            est = self._continue_from(est, ctx)
        ctx.set_progress(0.05, "fitting")
        t0 = time.time()
        w = (data.weights or {}).get("train")
        if w is not None and "sample_weight" in inspect.signature(est.fit).parameters:
            est.fit(Xtr, ytr, sample_weight=w)
        else:
            if w is not None:
                ctx.print(f"{type(est).__name__} does not support sample weights; recency weighting ignored")
            est.fit(Xtr, ytr)
        fit_time = time.time() - t0

        # iteration curves, when the estimator exposes them
        n_iter = 0
        if hasattr(est, "loss_curve_"):
            for i, v in enumerate(est.loss_curve_):
                ctx.log(i + 1, {"train/loss": v})
            for i, v in enumerate(getattr(est, "validation_scores_", None) or []):
                ctx.log(i + 1, {"val/score": v})
        elif hasattr(est, "staged_predict"):
            splits = [("train", Xtr, ytr), ("val", Xva, yva)]
            gens = {n: est.staged_predict(X) for n, X, _ in splits if len(X)}
            n_iter = len(getattr(est, "estimators_", [])) or 0
            stride = max(1, n_iter // 200)
            for i in range(n_iter):
                row = {}
                for n, X, y in splits:
                    if n in gens:
                        pred = next(gens[n])
                        if i % stride == 0 or i == n_iter - 1:
                            key = "acc" if data.task == "classification" else "mse"
                            row[f"{n}/{key}"] = (pred == y).mean() if key == "acc" else float(((pred - y) ** 2).mean())
                if row and (i % stride == 0 or i == n_iter - 1):
                    ctx.log(i + 1, row)

        self._snapshots(est, data, ctx, n_iter)
        summary: dict = {"fit_time_s": fit_time, **self._metrics(est, data, ctx, ("train", "val", "test"))}
        if hasattr(est, "feature_importances_"):
            names = data.info.get("feature_names") or [f"x{i}" for i in range(len(est.feature_importances_))]
            order = np.argsort(est.feature_importances_)[::-1][:30]
            ctx.log_artifact("feature_importance", "table",
                             {"columns": ["feature", "importance"],
                              "rows": [[str(names[i]), float(est.feature_importances_[i])] for i in order]})
        if ctx.save_model:
            try:
                import joblib
                joblib.dump(est, ctx.model_path("model.joblib"))
            except Exception as e:  # unpicklable custom estimators are fine to skip
                ctx.print(f"could not save model: {e}")
        return summary

    def _metrics(self, est, data, ctx, splits) -> dict:
        out = {}
        for split in splits:
            X, y = getattr(data, split)
            if len(X):
                pred = est.predict(X)
                out.update(task_metrics(data.task, y, pred, split))
                if data.task == "classification" and hasattr(est, "predict_proba") and len(set(y.tolist())) == 2:
                    from sklearn.metrics import roc_auc_score
                    out[f"{split}/auc"] = float(roc_auc_score(y, est.predict_proba(X)[:, 1]))
                if split == "test" and data.task == "classification":
                    ctx.log_artifact("confusion_matrix_test", "confusion_matrix", confusion(y, pred, data.info.get("class_names")))
        return out

    def _snapshots(self, est, data, ctx, n_iter: int) -> None:
        from ..training.snapshots import tabular_snapshot
        if not ctx.snapshots:
            return
        X, y = data.val if len(data.val[0]) else data.train
        if n_iter and hasattr(est, "staged_predict"):   # boosting: watch the boundary form stage by stage
            due = ctx.snapshot_schedule(n_iter)
            for k in range(1, n_iter + 1):
                if due(k):
                    res = tabular_snapshot(data.task, lambda A, k=k: _nth(est.staged_predict(A), k - 1), X, y, data.info)
                    if not res:
                        break
                    ctx.log_media(k, res[0], res[0], res[1])
            return
        res = tabular_snapshot(data.task, est.predict, X, y, data.info)
        if res:
            ctx.log_media(1, res[0], res[0], res[1])

    def evaluate(self, data, ctx):
        import joblib
        est = joblib.load(ctx.init_path / "model.joblib")
        splits = ctx.opt("eval_splits", "val+test").split("+")
        summary = self._metrics(est, data, ctx, splits)
        self._snapshots(est, data, ctx, 0)
        return summary


def _nth(gen, k):
    for i, v in enumerate(gen):
        if i == k:
            return v
    return v


@register
class LogisticRegressionModel(SklearnModel):
    id = "sklearn.logistic_regression"
    name = "Logistic regression"
    description = "Linear classifier (sklearn LogisticRegression)"

    def params(self):
        return [Param("C", "float", 1.0, "Inverse regularization strength", 1e-6, log=True),
                Param("max_iter", "int", 1000, min=1)]

    def build(self, p, task, seed):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=seed)


@register
class RidgeModel(SklearnModel):
    id = "sklearn.ridge"
    name = "Ridge regression"
    description = "L2-regularized linear regression"
    tasks = ("regression",)

    def params(self):
        return [Param("alpha", "float", 1.0, "Regularization strength", 0.0, log=True)]

    def build(self, p, task, seed):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=p["alpha"], random_state=seed)


@register
class RandomForestModel(SklearnModel):
    id = "sklearn.random_forest"
    name = "Random forest"
    description = "Bagged decision trees (classifier or regressor)"
    tasks = ("classification", "regression")

    def params(self):
        return [Param("n_estimators", "int", 200, min=1),
                Param("max_depth", "int", 0, "0 = unlimited", 0),
                Param("min_samples_leaf", "int", 1, min=1),
                Param("max_features", "choice", "sqrt", choices=["sqrt", "log2", "all"]),
                Param("n_jobs", "int", -1, "CPU cores (-1 = all)")]

    def build(self, p, task, seed):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=p["n_estimators"], max_depth=p["max_depth"] or None, min_samples_leaf=p["min_samples_leaf"],
                   max_features=None if p["max_features"] == "all" else p["max_features"], n_jobs=p["n_jobs"], random_state=seed)


@register
class GradientBoostingModel(SklearnModel):
    id = "sklearn.gradient_boosting"
    name = "Gradient boosting"
    description = "sklearn GradientBoosting; logs per-stage train/val curves"
    tasks = ("classification", "regression")

    def params(self):
        return [Param("n_estimators", "int", 200, min=1),
                Param("learning_rate", "float", 0.1, min=1e-5, log=True),
                Param("max_depth", "int", 3, min=1),
                Param("subsample", "float", 1.0, min=0.05, max=1.0)]

    def build(self, p, task, seed):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        return cls(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                   subsample=p["subsample"], random_state=seed)


@register
class SVMModel(SklearnModel):
    id = "sklearn.svm"
    name = "Support vector machine"
    description = "SVC / SVR with selectable kernel"
    tasks = ("classification", "regression")

    def params(self):
        return [Param("C", "float", 1.0, min=1e-6, log=True),
                Param("kernel", "choice", "rbf", choices=["rbf", "linear", "poly", "sigmoid"]),
                Param("gamma", "choice", "scale", choices=["scale", "auto"])]

    def build(self, p, task, seed):
        from sklearn.svm import SVC, SVR
        if task == "classification":
            return SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], random_state=seed)
        return SVR(C=p["C"], kernel=p["kernel"], gamma=p["gamma"])


@register
class KNNModel(SklearnModel):
    id = "sklearn.knn"
    name = "k-nearest neighbors"
    tasks = ("classification", "regression")

    def params(self):
        return [Param("n_neighbors", "int", 5, min=1), Param("weights", "choice", "uniform", choices=["uniform", "distance"])]

    def build(self, p, task, seed):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        cls = KNeighborsClassifier if task == "classification" else KNeighborsRegressor
        return cls(n_neighbors=p["n_neighbors"], weights=p["weights"])


@register
class SklearnMLPModel(SklearnModel):
    id = "sklearn.mlp"
    name = "MLP (sklearn)"
    description = "sklearn MLPClassifier/Regressor with loss curve and optional early stopping"
    tasks = ("classification", "regression")

    def params(self):
        return [Param("hidden_layer_sizes", "int_list", [100], "Units per hidden layer"),
                Param("activation", "choice", "relu", choices=["relu", "tanh", "logistic"]),
                Param("learning_rate_init", "float", 1e-3, min=1e-6, log=True, group="Training"),
                Param("alpha", "float", 1e-4, "L2 penalty", 0.0, log=True, group="Training"),
                Param("max_iter", "int", 200, "Epochs", 1, group="Training"),
                Param("early_stopping", "bool", False, "Hold out 10% of train for early stopping", group="Training"),
                Param("n_iter_no_change", "int", 10, "Early stopping patience", 1, group="Training")]

    def build(self, p, task, seed):
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPClassifier if task == "classification" else MLPRegressor
        return cls(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), activation=p["activation"], alpha=p["alpha"],
                   learning_rate_init=p["learning_rate_init"], max_iter=p["max_iter"], early_stopping=p["early_stopping"],
                   n_iter_no_change=p["n_iter_no_change"], random_state=seed)


@register
class AnyEstimator(SklearnModel):
    id = "sklearn.any"
    name = "Any sklearn estimator"
    description = "Import path + JSON kwargs, e.g. sklearn.tree.DecisionTreeClassifier"
    tasks = ("classification", "regression")

    def params(self):
        return [Param("estimator", "str", "sklearn.tree.DecisionTreeClassifier", "Fully-qualified class path",
                      suggestions=["sklearn.tree.DecisionTreeClassifier", "sklearn.tree.DecisionTreeRegressor",
                                   "sklearn.ensemble.ExtraTreesClassifier", "sklearn.ensemble.HistGradientBoostingClassifier",
                                   "sklearn.ensemble.HistGradientBoostingRegressor", "sklearn.naive_bayes.GaussianNB",
                                   "sklearn.linear_model.Lasso", "sklearn.linear_model.ElasticNet"]),
                Param("kwargs", "json", {}, "Constructor keyword arguments as JSON"),
                Param("pass_random_state", "bool", True, "Pass the run seed as random_state if supported")]

    def build(self, p, task, seed):
        import importlib
        import inspect
        mod, _, cls_name = p["estimator"].rpartition(".")
        cls = getattr(importlib.import_module(mod), cls_name)
        kw = dict(p.get("kwargs") or {})
        if p["pass_random_state"] and "random_state" in inspect.signature(cls).parameters:
            kw.setdefault("random_state", seed)
        return cls(**kw)
