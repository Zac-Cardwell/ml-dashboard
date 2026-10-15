"""Data quality report for static datasets (downloaded / uploaded; they rarely change, so reports are cached)."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from .. import config
from ..plugins.base import coerce_params
from ..plugins.registry import get_registry
from .profiling import clean, column_profile, quality_issues, target_summary, window_drift

REPORT_DIR = config.DATA_DIR / "reports"


def _grade(issues: list[dict]) -> str:
    crit = sum(i["severity"] == "critical" for i in issues)
    warn = sum(i["severity"] == "warning" for i in issues)
    return "A" if crit == 0 and warn == 0 else "B" if crit == 0 and warn <= 2 else "C" if crit == 0 else "D"


def report(dataset_id: str, params: dict | None = None, refresh: bool = False) -> dict:
    plugin = get_registry().get("dataset", dataset_id)
    p = coerce_params(plugin.params(), params or {})
    key = hashlib.md5(json.dumps([dataset_id, p], sort_keys=True, default=str).encode()).hexdigest()[:12]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{dataset_id}-{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    t0 = time.time()
    data = plugin.load(p, 0)
    out: dict = {"dataset": dataset_id, "name": plugin.name, "task": data.task, "modality": data.modality, "params": p,
                 "sizes": {k: data.info.get(f"n_{k}") for k in ("train", "val", "test")}, "created_at": time.time()}
    if data.modality == "tabular" and data.raw:
        out.update(_tabular(data))
    elif data.modality == "image":
        out.update(_image(data))
    elif data.modality == "text":
        out.update(_text(data))
    out["grade"] = _grade(out.get("issues", []))
    out["seconds"] = round(time.time() - t0, 2)
    out = clean(out)
    path.write_text(json.dumps(out))
    return out


def cached_summary(dataset_id: str) -> dict | None:
    """Latest cached report summary for a dataset (any params)."""
    files = sorted(REPORT_DIR.glob(f"{dataset_id}-*.json"), key=lambda f: f.stat().st_mtime) if REPORT_DIR.exists() else []
    if not files:
        return None
    r = json.loads(files[-1].read_text())
    return {"grade": r.get("grade"), "n_issues": len(r.get("issues", [])), "created_at": r.get("created_at")}


def _tabular(data) -> dict:
    classes = data.info.get("class_names")
    frames = []
    for split in ("train", "val", "test"):
        df, y = data.raw[split]
        if len(df):
            f = df.copy()
            f["__target__"] = [classes[int(v)] for v in y] if classes and data.task == "classification" else y
            f["__split__"] = split
            frames.append(f)
    full = pd.concat(frames, ignore_index=True)
    feats = [c for c in full.columns if c not in ("__target__", "__split__")]
    profiles = {c: column_profile(full[c]) for c in feats}
    issues = quality_issues(full[feats + ["__target__"]], profiles, "__target__", data.task)
    for i in issues:
        if i.get("column") == "__target__":
            i["column"] = "target"
            i["message"] = i["message"].replace("'__target__'", "the target")
    tr, te = full[full["__split__"] == "train"], full[full["__split__"] == "test"]
    drift = window_drift(tr, te, feats) if len(te) else {}
    for c, d in drift.items():
        if d.get("status") in ("warn", "alert"):
            issues.append({"severity": "warning", "kind": "split_drift", "column": c,
                           "message": f"'{c}' is distributed differently in test vs train (PSI {d['psi']:.2f}) — the split may not be representative"})
    overlap = _overlap(tr[feats], te[feats]) if len(te) else 0.0
    if overlap > 0.01:
        issues.append({"severity": "critical", "kind": "split_leak", "column": None,
                       "message": f"{overlap:.1%} of test rows also appear in train — test scores will be optimistic"})
    return {"n_rows": len(full), "n_columns": len(feats), "columns": profiles, "issues": issues,
            "target": target_summary(full["__target__"], data.task), "split_drift": drift,
            "target_by_split": {s: target_summary(full.loc[full["__split__"] == s, "__target__"], data.task) for s in ("train", "val", "test")}}


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ha = set(pd.util.hash_pandas_object(a.astype(str), index=False))
    hb = pd.util.hash_pandas_object(b.astype(str), index=False)
    return float(hb.isin(ha).mean()) if len(hb) else 0.0


def _image(data) -> dict:
    from ..mlops.features import image_stats
    names = data.info.get("class_names") or []
    out: dict = {"input_shape": data.info.get("input_shape"), "issues": []}
    rng = np.random.default_rng(0)
    stats, labels = [], []
    src = data.val if data.val is not None and len(data.val) else data.train
    norm = data.info.get("image_norm") or {"mean": [0.5], "std": [0.5]}
    for i in rng.choice(len(src), min(600, len(src)), replace=False):
        x, y = src[int(i)]
        a = x.numpy() * np.asarray(norm["std"])[:, None, None] + np.asarray(norm["mean"])[:, None, None]
        stats.append(image_stats(np.clip(a, 0, 1).transpose(1, 2, 0).squeeze()))
        labels.append(names[int(y)] if names else str(int(y)))
    df = pd.DataFrame(stats)
    out["columns"] = {c: column_profile(df[c]) for c in df.columns}
    out["target"] = target_summary(pd.Series(labels), "classification")
    shares = pd.Series(labels).value_counts(normalize=True)
    if len(shares) and shares.min() < 0.5 / len(shares):
        out["issues"].append({"severity": "warning", "kind": "imbalance", "column": "label",
                              "message": f"Class '{shares.idxmin()}' is under-represented ({shares.min():.1%} of samples)"})
    dark = float((df["brightness"] < 0.05).mean())
    if dark > 0.01:
        out["issues"].append({"severity": "warning", "kind": "blank_images", "column": None, "message": f"{dark:.1%} of sampled images are almost black"})
    return out


def _text(data) -> dict:
    key = "text" if data.task == "classification" else "source"
    names = data.info.get("class_names") or []
    rows = {s: getattr(data, s) or [] for s in ("train", "val", "test")}
    allrows = [r for v in rows.values() for r in v]
    texts = pd.Series([r[key] for r in allrows])
    lengths = texts.str.split().str.len()
    out: dict = {"n_rows": len(allrows), "columns": {"words per example": column_profile(lengths)}, "issues": []}
    if data.task == "classification":
        out["target"] = target_summary(pd.Series([names[r["label"]] if names else r["label"] for r in allrows]), "classification")
    else:
        out["columns"]["target words"] = column_profile(pd.Series([len(str(r["target"]).split()) for r in allrows]))
    dup = float(texts.duplicated().mean())
    if dup > 0.01:
        out["issues"].append({"severity": "warning", "kind": "duplicates", "column": key, "message": f"{dup:.1%} of examples are duplicate texts"})
    empty = float((texts.str.strip() == "").mean())
    if empty:
        out["issues"].append({"severity": "warning", "kind": "empty", "column": key, "message": f"{empty:.1%} of examples are empty"})
    train_set = set(r[key] for r in rows["train"])
    leak = float(np.mean([r[key] in train_set for r in rows["test"]])) if rows["test"] else 0.0
    if leak > 0.01:
        out["issues"].append({"severity": "critical", "kind": "split_leak", "column": key,
                              "message": f"{leak:.1%} of test examples also appear in train — test scores will be optimistic"})
    trunc = float((lengths > 400).mean())
    if trunc > 0.05:
        out["issues"].append({"severity": "info", "kind": "long", "column": key,
                              "message": f"{trunc:.1%} of examples exceed 400 words and may be truncated by the tokenizer"})
    return out
