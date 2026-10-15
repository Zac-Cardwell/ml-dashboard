"""Column profiling and data-quality checks shared by static dataset reports and live-source health."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _jsonable(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def column_profile(s: pd.Series) -> dict:
    n = len(s)
    miss = float(s.isna().mean()) if n else 0.0
    v = s.dropna()
    out: dict = {"missing": miss, "n_unique": int(v.nunique()) if len(v) else 0}
    if pd.api.types.is_bool_dtype(v):
        v = v.astype(str)
    if pd.api.types.is_numeric_dtype(v):
        a = v.to_numpy(dtype=float)
        q1, q3 = (np.percentile(a, [25, 75]) if len(a) else (0, 0))
        iqr = q3 - q1
        out.update(kind="numeric", mean=float(a.mean()) if len(a) else None, std=float(a.std()) if len(a) else None,
                   min=float(a.min()) if len(a) else None, max=float(a.max()) if len(a) else None,
                   p50=float(np.median(a)) if len(a) else None,
                   zeros=float((a == 0).mean()) if len(a) else 0.0,
                   outliers=float(((a < q1 - 3 * iqr) | (a > q3 + 3 * iqr)).mean()) if len(a) and iqr > 0 else 0.0,
                   hist=_hist(a))
    else:
        vs = v.astype(str)
        top = vs.value_counts().head(12)
        out.update(kind="text" if len(vs) and vs.str.len().mean() > 40 else "categorical",
                   top=[{"value": k, "share": float(c / max(len(vs), 1))} for k, c in top.items()],
                   mean_length=float(vs.str.len().mean()) if len(vs) else None)
    return out


def _hist(a: np.ndarray, bins: int = 20) -> dict | None:
    if not len(a) or a.min() == a.max():
        return None
    lo, hi = np.percentile(a, [0.5, 99.5])
    if lo == hi:
        lo, hi = a.min(), a.max()
    counts, edges = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": counts.astype(int).tolist()}


def quality_issues(df: pd.DataFrame, profiles: dict, target: str | None, task: str | None, id_col: str | None = None) -> list[dict]:
    """Common data problems, most severe first."""
    issues: list[dict] = []
    n = len(df)
    add = lambda sev, kind, col, msg: issues.append({"severity": sev, "kind": kind, "column": col, "message": msg})
    if n == 0:
        add("critical", "empty", None, "The dataset has no rows")
        return issues
    feats = [c for c in df.columns if c not in (target, id_col, "__ts")]
    dup = float(df[feats].astype(str).duplicated().mean()) if feats else 0.0
    if dup > 0.01:
        add("warning" if dup < 0.2 else "critical", "duplicates", None, f"{dup:.1%} of rows are exact duplicates (features)")
    if id_col and id_col in df:
        d = float(df[id_col].duplicated().mean())
        if d > 0:
            add("critical", "duplicate_ids", id_col, f"{d:.2%} of rows reuse an existing id")
    for c, p in profiles.items():
        if c == "__ts":
            continue
        if p["missing"] >= 0.5:
            add("critical", "missing", c, f"'{c}' is {p['missing']:.0%} missing")
        elif p["missing"] >= 0.05:
            add("warning", "missing", c, f"'{c}' is {p['missing']:.0%} missing")
        if p["n_unique"] <= 1 and c != target:
            add("warning", "constant", c, f"'{c}' has a single value — it carries no information")
        if p.get("kind") in ("categorical", "text") and c not in (target, id_col) and p["n_unique"] > 0.9 * n and n > 50:
            add("warning", "id_like", c, f"'{c}' is unique per row (an ID or free text?) — exclude it from features")
        if p.get("kind") == "numeric" and p.get("outliers", 0) > 0.01:
            add("info", "outliers", c, f"'{c}' has {p['outliers']:.1%} extreme outliers (beyond 3×IQR)")
    if target and target in df:
        y = df[target].dropna()
        if task == "classification" and len(y):
            shares = y.astype(str).value_counts(normalize=True)
            if shares.min() < 0.05:
                add("warning", "imbalance", target, f"Class imbalance: '{shares.idxmin()}' is only {shares.min():.1%} of rows "
                                                     "— prefer AUC / F1 over accuracy")
        miss = df[target].isna().mean()
        if miss > 0:
            add("info" if miss < 0.2 else "warning", "unlabeled", target, f"{miss:.1%} of rows have no target (excluded from training)")
        issues += leakage(df, target, task, feats)
    issues.sort(key=lambda i: SEVERITY_ORDER[i["severity"]])
    return issues


def leakage(df: pd.DataFrame, target: str, task: str | None, feats: list[str]) -> list[dict]:
    """Features that predict the target almost perfectly on their own are usually leaks (e.g. an outcome column)."""
    out = []
    d = df.dropna(subset=[target])
    if len(d) < 30:
        return out
    if task == "classification":
        y = d[target].astype(str)
        base = y.value_counts(normalize=True).max()
        for c in feats:
            x = d[c]
            if x.nunique() < 2 or x.nunique() > 0.5 * len(d):
                continue
            key = pd.qcut(x, 10, duplicates="drop").astype(str) if pd.api.types.is_numeric_dtype(x) and x.nunique() > 20 else x.astype(str)
            acc = y.groupby(key).agg(lambda s: s.value_counts(normalize=True).max())
            purity = float((acc * key.value_counts(normalize=True).reindex(acc.index)).sum())
            if purity > 0.99 and base < 0.95:
                out.append({"severity": "critical", "kind": "leakage", "column": c,
                            "message": f"'{c}' alone predicts the target {purity:.1%} of the time — possible target leakage"})
    else:
        y = pd.to_numeric(d[target], errors="coerce")
        for c in feats:
            if pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 2:
                r = y.corr(d[c])
                if r is not None and abs(r) > 0.98:
                    out.append({"severity": "critical", "kind": "leakage", "column": c,
                                "message": f"'{c}' correlates {r:.3f} with the target — possible target leakage"})
    return out


def target_summary(y: pd.Series, task: str | None) -> dict:
    y = y.dropna()
    if task == "classification" or not pd.api.types.is_numeric_dtype(y):
        vc = y.astype(str).value_counts(normalize=True)
        return {"kind": "classes", "shares": {str(k): float(v) for k, v in vc.items()}}
    a = y.to_numpy(dtype=float)
    return {"kind": "numeric", "mean": float(a.mean()), "std": float(a.std()), "hist": _hist(a)}


def window_drift(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str]) -> dict:
    """PSI of each column: current window vs reference window (uses the monitoring implementation)."""
    from ..mlops.monitoring import DEFAULT_CONFIG, feature_drift
    from ..mlops.profile import categorical_profile, numeric_profile
    rng = np.random.default_rng(0)
    out = {}
    for c in cols:
        if c not in ref or c not in cur or ref[c].dropna().empty:
            continue
        numeric = pd.api.types.is_numeric_dtype(ref[c]) and not pd.api.types.is_bool_dtype(ref[c])
        prof = numeric_profile(ref[c].to_numpy(dtype=float), rng) if numeric else categorical_profile(ref[c].tolist())
        d = feature_drift(c, prof, cur[c].tolist(), DEFAULT_CONFIG)
        out[c] = {"psi": d.get("psi"), "status": d.get("status"), "ks": d.get("ks"), "mean_shift_sd": d.get("mean_shift_sd"),
                  "ref_mean": prof.get("mean"), "cur_mean": d.get("mean"), "missing": d.get("missing_rate"),
                  "ref_missing": prof.get("missing"), "out_of_range": d.get("out_of_range_rate"), "unseen": d.get("unseen_rate"),
                  "unseen_examples": d.get("unseen_examples"), "psi_noise": d.get("psi_noise")}
    return out


def clean(obj):
    """Make nested profiling output JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return _jsonable(obj)
