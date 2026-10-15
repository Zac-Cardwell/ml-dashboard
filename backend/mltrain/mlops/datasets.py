"""Datasets built from production: labeled prediction logs become (re)training / evaluation data."""
from __future__ import annotations

import json
import time

import numpy as np

from .. import db
from ..plugins.base import DataBundle, DatasetPlugin, Param
from ..plugins.registry import register


def _labeled_rows(dep_id: str, days: float, limit: int, until: float = 0.0):
    with db.get_conn() as c:
        dep = db.one(c, "SELECT * FROM deployments WHERE id=?", (dep_id,))
        if not dep:
            raise ValueError(f"Deployment '{dep_id}' not found")
        ver = db.one(c, "SELECT * FROM model_versions WHERE id=?", (dep["champion_version_id"],))
        rows = db.rows(c, "SELECT payload, label, ts FROM predictions WHERE deployment_id=? AND shadow=0 AND label IS NOT NULL "
                          "AND payload IS NOT NULL AND ts>=? AND ts<? AND (label_ts IS NULL OR label_ts<?) ORDER BY ts DESC LIMIT ?",
                       (dep_id, (until or time.time()) - days * 86400, until or time.time(), until or time.time(), limit))
    meta = json.loads(open(f"{ver['path']}/meta.json").read()) if ver else {}
    rows = sorted((r for r in rows if r["payload"] is not None), key=lambda r: r["ts"])   # oldest -> newest
    return rows, meta


def time_split(n: int, params: dict):
    """Out-of-time split, the standard for production data: train on older rows, validate / test on the newest,
    so a model is never evaluated on traffic it (or an earlier model trained on this feed) learned from."""
    nt = int(n * params.get("test_fraction", 0.15))
    nv = int(n * params.get("val_fraction", 0.15))
    idx = np.arange(n)
    return idx[: n - nt - nv], idx[n - nt - nv: n - nt], idx[n - nt:]


@register
class DeploymentFeedback(DatasetPlugin):
    """Production inputs + the ground-truth labels that came back for them."""
    id = "deployment.feedback"
    name = "Production feedback data"
    description = "Labeled prediction logs of a deployment (tabular or text). Used to retrain on fresh data."
    task = "classification"
    modality = "tabular"

    def params(self):
        from ..builtin.datasets import TABULAR_PARAMS
        return [Param("deployment_id", "str", "", "Deployment whose labeled traffic to use", group="Source"),
                Param("days", "float", 30.0, "Use labeled predictions from the last N days", 0.01, group="Source"),
                Param("max_rows", "int", 200_000, min=10, group="Source"),
                Param("until", "float", 0.0, "Freeze the snapshot at this unix time (0 = now); automation sets it so challenger "
                      "and champion see identical data", 0.0, group="Source")] + TABULAR_PARAMS

    def load(self, params, seed):
        import pandas as pd

        from ..builtin.datasets import tabular_bundle
        rows, meta = _labeled_rows(params["deployment_id"], params["days"], params["max_rows"], params.get("until") or 0.0)
        if len(rows) < 20:
            raise ValueError(f"Only {len(rows)} labeled production examples available; need at least 20")
        classes = (meta.get("info") or {}).get("class_names")
        df = pd.DataFrame([r["payload"] for r in rows])
        raw_cols = (meta.get("info") or {}).get("raw_columns")
        if raw_cols:
            df = df.reindex(columns=raw_cols)
        labels = [r["label"] for r in rows]
        split = time_split(len(df), params)
        if meta.get("task") == "regression":
            return tabular_bundle(df, np.asarray(labels, float), params, seed, "regression", indices=split)
        names = list(classes) if classes else sorted({str(v) for v in labels})
        idx = {n: i for i, n in enumerate(names)}
        y = np.asarray([idx[str(v)] if str(v) in idx else int(v) for v in labels])
        return tabular_bundle(df, y, params, seed, "classification", class_names=names, indices=split)


@register
class DeploymentFeedbackRegression(DeploymentFeedback):
    id = "deployment.feedback_regression"
    name = "Production feedback data (regression)"
    task = "regression"


@register
class DeploymentFeedbackText(DatasetPlugin):
    id = "deployment.feedback_text"
    name = "Production feedback data (text classification)"
    description = "Labeled text requests of a deployment"
    task = "classification"
    modality = "text"

    def params(self):
        return [Param("deployment_id", "str", "", group="Source"), Param("days", "float", 30.0, min=0.01, group="Source"),
                Param("max_rows", "int", 50_000, min=10, group="Source"), Param("until", "float", 0.0, min=0.0, group="Source"),
                Param("val_fraction", "float", 0.15, min=0.0, max=0.5, group="Data split"),
                Param("test_fraction", "float", 0.15, min=0.0, max=0.5, group="Data split"),
                Param("split_seed", "int", 0, min=-1, group="Data split")]

    def load(self, params, seed):
        rows, meta = _labeled_rows(params["deployment_id"], params["days"], params["max_rows"], params.get("until") or 0.0)
        names = (meta.get("info") or {}).get("class_names") or sorted({str(r["label"]) for r in rows})
        idx = {n: i for i, n in enumerate(names)}
        data = [{"text": str(r["payload"]), "label": idx.get(str(r["label"]), 0)} for r in rows]
        tr, va, te = time_split(len(data), params)
        info = {"n_classes": len(names), "class_names": names, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        return DataBundle("classification", "text", [data[i] for i in tr], [data[i] for i in va], [data[i] for i in te], info)
