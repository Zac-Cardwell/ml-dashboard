"""Studio datasets as training datasets ("studio.<slug>" in New training).

* With a target (set in the Studio): a classification / regression dataset.
* Without a target: task "unsupervised" — only clustering / anomaly / dimensionality-reduction models match.
* If the dataset has a __split column (a Split step), training uses exactly that split, so statistics fitted in the
  recipe (train rows only) and the model's train/val/test agree — no leakage between them. Otherwise the usual
  random split applies.
* Each run records the dataset version it used (dataset_info artifact), for reproducibility.
"""
from __future__ import annotations

import numpy as np

from ..plugins.base import DataBundle, DatasetPlugin, Param
from .steps import SPLIT


def studio_datasets() -> list[DatasetPlugin]:
    from ..builtin.datasets import TABULAR_PARAMS
    from . import catalog
    try:
        rows = [d for d in catalog.all_datasets() if d["status"] == "ready" or (d.get("version") or 0) > 0]
    except Exception:            # table not created yet / standalone export
        return []
    out = []
    for d in rows:
        task = d.get("task") if d.get("target") else "unsupervised"
        has_split = any(s["name"] == SPLIT for s in (d.get("build") or {}).get("schema") or [])

        class _S(DatasetPlugin):
            id = f"studio.{d['slug']}"
            name = f"{d['name']} (Studio)"
            description = (d.get("description") or f"Data Studio {d['kind']} dataset") + \
                          (f" · target {d['target']}" if d.get("target") else " · no target (unsupervised)") + \
                          (" · uses its own train/val/test split" if has_split else "")
            modality = "tabular"
            _ds_id = d["id"]
            _split = has_split

            def params(self):
                ps = [Param("drop_columns", "str", "", "Comma-separated columns to ignore (ids etc.)", group="Columns")]
                if self._split:
                    return ps + [p for p in TABULAR_PARAMS if p.name == "standardize"]
                return ps + TABULAR_PARAMS

            def load(self, params, seed):
                return load_bundle(self._ds_id, params, seed)

        _S.task = task or "classification"
        _S.__name__ = f"Studio_{d['slug']}"
        out.append(_S())
    return out


def load_bundle(ds_id: str, params: dict, seed: int) -> DataBundle:
    import pandas as pd

    from ..builtin.datasets import tabular_bundle
    from . import catalog, engine
    d = catalog.get(ds_id)
    df = pd.read_parquet(engine.table_path(d["id"], d["version"]))
    drop = [c.strip() for c in (params.get("drop_columns") or "").split(",") if c.strip()]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    target, task = d.get("target"), d.get("task") if d.get("target") else "unsupervised"
    indices = None
    if SPLIT in df.columns:
        s = df.pop(SPLIT).astype(str).to_numpy()
        indices = tuple(np.flatnonzero(s == k) for k in ("train", "val", "test"))
    if target:
        if target not in df.columns:
            raise ValueError(f"target column '{target}' is no longer in the dataset")
        df = df.dropna(subset=[target]).reset_index(drop=True) if indices is None else df
        y_raw = df.pop(target)
        class_names = None
        if task == "classification":
            cats = pd.Categorical(y_raw.astype(str))
            y, class_names = cats.codes, list(cats.categories)
        else:
            y = pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        b = tabular_bundle(df, y, params, seed, task, class_names=class_names, indices=indices)
    else:
        b = tabular_bundle(df, np.zeros(len(df), dtype=np.float32), params, seed, "regression", indices=indices)
        b.task = "unsupervised"
    b.info.update(studio_dataset=d["id"], studio_version=d["version"], studio_name=d["name"], target_name=target,
                  split_source="dataset __split column" if indices is not None else "random")
    return b
