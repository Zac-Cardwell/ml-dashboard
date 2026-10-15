"""XGBoost and LightGBM gradient-boosted trees with live per-iteration metrics."""
from __future__ import annotations

import time

import numpy as np

from ..plugins.base import Param
from ..plugins.registry import register
from .models_sklearn import SklearnModel

GBM_COMMON = [
    Param("n_estimators", "int", 500, "Boosting rounds (max)", 1, group="Training"),
    Param("learning_rate", "float", 0.05, min=1e-5, log=True, group="Training"),
    Param("subsample", "float", 0.8, "Row subsample per tree", 0.05, 1.0, group="Training"),
    Param("colsample_bytree", "float", 0.8, "Feature subsample per tree", 0.05, 1.0, group="Training"),
    Param("reg_lambda", "float", 1.0, "L2 regularization", 0.0, group="Training"),
    Param("early_stopping_rounds", "int", 50, "Stop when val metric hasn't improved for N rounds (0 = off)", 0, group="Early stopping"),
]


class _GBM(SklearnModel):
    tasks = ("classification", "regression")
    uses_gpu = False

    def _metric_names(self, task):
        raise NotImplementedError

    def _fit(self, est, data, ctx, log_iter, init):
        raise NotImplementedError

    def _predict_upto(self, est, X, k):
        raise NotImplementedError

    def train(self, data, ctx):
        import joblib
        (Xtr, ytr), (Xva, yva) = data.train, data.val
        p = ctx.params
        est = self.build(p, data.task, ctx.seed)
        init = joblib.load(ctx.init_path / "model.joblib") if ctx.init_path else None
        total = p["n_estimators"]
        state = {"t": time.time()}

        def log_iter(i: int, row: dict) -> bool:
            """Called after every boosting round; returns True to stop."""
            out = {}
            for k, v in row.items():
                out[k] = v
                if k.endswith("/error"):            # error rate -> accuracy for consistency with other models
                    out[k.replace("/error", "/acc")] = 1 - v
            ctx.log(i, out)
            if time.time() - state["t"] > 0.5:
                ctx.set_progress(i / total, f"round {i}/{total}")
                state["t"] = time.time()
            return ctx.should_stop()

        ctx.set_progress(0.01, "fitting")
        t0 = time.time()
        self._fit(est, data, ctx, log_iter, init)
        fit_time = time.time() - t0
        n_iter = self._n_iter(est)

        if ctx.snapshots:
            from ..training.snapshots import tabular_snapshot
            X, y = (Xva, yva) if len(Xva) else (Xtr, ytr)
            due = ctx.snapshot_schedule(n_iter)
            for k in range(1, n_iter + 1):
                if due(k):
                    res = tabular_snapshot(data.task, lambda A, k=k: self._predict_upto(est, A, k), X, y, data.info)
                    if not res:
                        break
                    ctx.log_media(k, res[0], res[0], res[1])

        summary = {"fit_time_s": fit_time, "n_trees": n_iter, **self._metrics(est, data, ctx, ("train", "val", "test"))}
        if hasattr(est, "feature_importances_"):
            names = data.info.get("feature_names") or [f"x{i}" for i in range(len(est.feature_importances_))]
            order = np.argsort(est.feature_importances_)[::-1][:30]
            ctx.log_artifact("feature_importance", "table", {"columns": ["feature", "importance"],
                                                            "rows": [[str(names[i]), float(est.feature_importances_[i])] for i in order]})
        if ctx.save_model:
            joblib.dump(est, ctx.model_path("model.joblib"))
        return summary


@register
class XGBoost(_GBM):
    id = "xgboost"
    name = "XGBoost"
    description = "Gradient-boosted trees (XGBoost); live per-round curves, early stopping, optional GPU"
    framework = "xgboost"

    def params(self):
        return [Param("max_depth", "int", 6, min=1, group="Model"),
                Param("min_child_weight", "float", 1.0, min=0.0, group="Model"),
                Param("use_gpu", "bool", False, "Train on the GPU (device='cuda')", group="Model")] + GBM_COMMON

    def build(self, p, task, seed):
        import xgboost as xgb
        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        return cls(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                   min_child_weight=p["min_child_weight"], subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                   reg_lambda=p["reg_lambda"], random_state=seed, device="cuda" if p["use_gpu"] else "cpu",
                   early_stopping_rounds=p["early_stopping_rounds"] or None,
                   eval_metric=(["logloss", "error"] if task == "classification" else ["rmse", "mae"]))

    def _fit(self, est, data, ctx, log_iter, init):
        import xgboost as xgb
        (Xtr, ytr), (Xva, yva) = data.train, data.val
        if data.task == "classification" and data.info.get("n_classes", 2) > 2:
            est.set_params(eval_metric=["mlogloss", "merror"])
        names = ["train"] + (["val"] if len(Xva) else [])

        class Live(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                row = {}
                for i, n in enumerate(names):
                    for m, vals in evals_log.get(f"validation_{i}", {}).items():
                        row[f"{n}/{ {'mlogloss': 'logloss', 'merror': 'error'}.get(m, m)}"] = vals[-1]
                return log_iter(epoch + 1, row)

        est.set_params(callbacks=[Live()])
        eval_set = [(Xtr, ytr)] + ([(Xva, yva)] if len(Xva) else [])
        if not len(Xva):
            est.set_params(early_stopping_rounds=None)
        est.fit(Xtr, ytr, eval_set=eval_set, verbose=False, xgb_model=init.get_booster() if init is not None else None,
                sample_weight=(data.weights or {}).get("train"))
        est.callbacks = None   # keep the saved model picklable (set_params would re-configure the booster)

    def _n_iter(self, est):
        return int(est.get_booster().num_boosted_rounds())

    def _predict_upto(self, est, X, k):
        return est.predict(X, iteration_range=(0, k))


@register
class LightGBM(_GBM):
    id = "lightgbm"
    name = "LightGBM"
    description = "Fast histogram gradient boosting (LightGBM); live per-round curves and early stopping"
    framework = "lightgbm"

    def params(self):
        return [Param("num_leaves", "int", 31, min=2, group="Model"),
                Param("max_depth", "int", -1, "-1 = unlimited", group="Model"),
                Param("min_child_samples", "int", 20, min=1, group="Model")] + GBM_COMMON

    def build(self, p, task, seed):
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        return cls(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                   max_depth=p["max_depth"], min_child_samples=p["min_child_samples"], subsample=p["subsample"],
                   subsample_freq=1 if p["subsample"] < 1 else 0, colsample_bytree=p["colsample_bytree"],
                   reg_lambda=p["reg_lambda"], random_state=seed, verbose=-1)

    def _fit(self, est, data, ctx, log_iter, init):
        import lightgbm as lgb
        (Xtr, ytr), (Xva, yva) = data.train, data.val
        p = ctx.params
        clf = data.task == "classification"
        metrics = (["multi_logloss", "multi_error"] if data.info.get("n_classes", 2) > 2 else ["binary_logloss", "binary_error"]) \
            if clf else ["rmse", "l1"]
        rename = {"multi_logloss": "logloss", "binary_logloss": "logloss", "multi_error": "error", "binary_error": "error",
                  "l1": "mae", "rmse": "rmse"}

        def live(env):
            row = {f"{'train' if d == 'training' else 'val'}/{rename.get(m, m)}": v for d, m, v, _ in env.evaluation_result_list}
            if log_iter(env.iteration + 1, row):
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

        callbacks = [live]
        eval_set = [(Xtr, ytr)] + ([(Xva, yva)] if len(Xva) else [])
        if p["early_stopping_rounds"] and len(Xva):
            callbacks.append(lgb.early_stopping(p["early_stopping_rounds"], first_metric_only=True, verbose=False))
        est.fit(Xtr, ytr, sample_weight=(data.weights or {}).get("train"), eval_set=eval_set,
                eval_names=["training", "valid"][:len(eval_set)], eval_metric=metrics,
                callbacks=callbacks, init_model=init.booster_ if init is not None else None)

    def _n_iter(self, est):
        return int(est.booster_.current_iteration())

    def _predict_upto(self, est, X, k):
        return est.predict(X, num_iteration=k)
