"""Frozen dataset snapshots (data/snapshots/index.json).

Portfolio exports freeze datasets that can't be re-downloaded (live database sources, uploaded/feedback data) into
files, with the exact train/val/test assignment (and recency weights) the original run used, so a fresh clone
reproduces the same splits without access to the original database.
"""
from __future__ import annotations

import json

import numpy as np

from .. import config
from ..plugins.base import DataBundle, DatasetPlugin, Param
from ..plugins.registry import register

PARAMS = [Param("standardize", "bool", True, "Standardize numeric features (fit on train)", group="Preprocessing")]


def _load_tabular(entry: dict, params: dict, seed: int) -> DataBundle:
    import pandas as pd

    from .datasets import tabular_bundle
    path = config.DATA_DIR / "snapshots" / _file_for(entry, seed)
    df = pd.read_csv(path, low_memory=False)
    split = df.pop("__split").to_numpy()
    w = df.pop("__weight").to_numpy(dtype=np.float32) if "__weight" in df else None
    y = df.pop(entry["target"]).to_numpy()
    idx = tuple(np.flatnonzero(split == s) for s in ("train", "val", "test"))
    b = tabular_bundle(df, y, params, seed, entry["task"], class_names=entry.get("class_names"), indices=idx)
    if w is not None:
        b.weights = {"train": w[idx[0]]}
    return b


def _load_text(entry: dict, params: dict, seed: int) -> DataBundle:
    rows = {"train": [], "val": [], "test": []}
    with open(config.DATA_DIR / "snapshots" / _file_for(entry, seed)) as f:
        for line in f:
            r = json.loads(line)
            rows[r.pop("__split")].append(r)
    return DataBundle(entry["task"], "text", rows["train"], rows["val"], rows["test"], dict(entry.get("info") or {}))


def _file_for(entry: dict, seed: int) -> str:
    files = entry["files"]                       # {"<seed>" | "*": filename}
    return files.get(str(seed)) or files.get("*") or next(iter(files.values()))


def _make(sid: str, entry: dict):
    class _S(DatasetPlugin):
        id = sid
        name = entry.get("name", sid)
        description = entry.get("description", "Frozen dataset snapshot")
        task = entry["task"]
        modality = entry["modality"]

        def params(self):
            return PARAMS if entry["modality"] == "tabular" else []

        def load(self, params, seed):
            return (_load_tabular if entry["modality"] == "tabular" else _load_text)(entry, params, seed)

    _S.__name__ = "Snapshot_" + sid.replace(".", "_").replace("-", "_")
    return register(_S)


_index = config.DATA_DIR / "snapshots" / "index.json"
if _index.exists():
    for _sid, _entry in json.loads(_index.read_text()).items():
        _make(_sid, _entry)
