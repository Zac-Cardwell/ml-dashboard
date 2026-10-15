"""Fitted, serializable preprocessing so training and serving use exactly the same transform.

Raw records (dicts / DataFrames with original column names and string categories) go in,
the model's float matrix comes out. Saved with every model (meta.json) and every registry version.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

MAX_CATEGORIES = 100


class TabularPreprocessor:
    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.columns: list[str] = []
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.medians: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.mean: list[float] | None = None
        self.std: list[float] | None = None

    # ---------------------------------------------------------------- fit / transform
    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.columns = list(df.columns)
        self.numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        self.categorical = [c for c in df.columns if c not in self.numeric]
        self.medians = {c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in self.numeric}
        self.categories = {c: sorted(df[c].dropna().astype(str).unique().tolist())[:MAX_CATEGORIES] for c in self.categorical}
        self.mean = self.std = None
        if self.standardize:
            raw = self._encode(df)
            self.mean = raw.mean(0).tolist()
            self.std = (raw.std(0) + 1e-8).tolist()
        return self

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        parts = []
        for c in self.numeric:
            col = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
            parts.append(col.fillna(self.medians[c]).to_numpy(np.float32)[:, None])
        for c in self.categorical:
            vals = df[c].astype(str).to_numpy() if c in df else np.array([""] * n)
            cats = self.categories[c]
            parts.append((vals[:, None] == np.array(cats)[None, :]).astype(np.float32))
        return np.concatenate(parts, 1) if parts else np.zeros((n, 0), np.float32)

    def transform(self, df: pd.DataFrame | list[dict]) -> np.ndarray:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(list(df))
        X = self._encode(df)
        if self.mean is not None:
            X = (X - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        return X.astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + [f"{c}={v}" for c in self.categorical for v in self.categories[c]]

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"type": "tabular", "standardize": self.standardize, "columns": self.columns, "numeric": self.numeric,
                "categorical": self.categorical, "medians": self.medians, "categories": self.categories,
                "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "TabularPreprocessor":
        p = cls(d.get("standardize", True))
        for k in ("columns", "numeric", "categorical", "medians", "categories", "mean", "std"):
            setattr(p, k, d.get(k))
        return p

    def schema(self) -> dict:
        """Input schema for serving docs / validation."""
        return {"numeric": self.numeric, "categorical": {c: self.categories[c] for c in self.categorical}}


def apply_saved_preprocessor(problem, init_path: Path):
    """Evaluate / fine-tune a saved tabular model with ITS fitted preprocessor (the serving contract),
    not one re-fitted on the new data — otherwise train/serve skew sneaks in."""
    meta = json.loads((init_path / "meta.json").read_text())
    pre_d = (meta.get("info") or {}).get("preprocessor")
    if problem.modality != "tabular" or not pre_d or not problem.raw:
        return problem
    pre = TabularPreprocessor.from_dict(pre_d)
    missing = [c for c in pre.columns if c not in problem.raw["train"][0].columns]
    if missing:
        print(f"warning: columns expected by the saved model are missing (filled with training medians): {missing}", flush=True)
    for split in ("train", "val", "test"):
        df, y = problem.raw[split]
        setattr(problem, split, (pre.transform(df), y))
    problem.info.update(preprocessor=pre_d, feature_names=pre.feature_names, n_features=len(pre.feature_names))
    return problem
