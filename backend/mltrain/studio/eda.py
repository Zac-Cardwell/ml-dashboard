"""Exploratory data analysis for Studio datasets: column profiles, data-quality alerts, correlations, missingness,
target relationships and time coverage. Aggregations run in DuckDB over the full table; pairwise statistics
(correlations, mutual information) use a bounded random sample and say so."""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from . import engine
from .engine import quote_ident as q
from .steps import SPLIT

SAMPLE = 200_000
NUMERIC = re.compile(r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|DOUBLE|REAL|DECIMAL)")
TEMPORAL = re.compile(r"^(DATE|TIMESTAMP|TIME)")


def _f(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _cache(ds: dict):
    v = (ds.get("build") or {}).get("version") or ds["version"]
    return engine.root() / "profiles" / f"{ds['id']}_v{v}.json"


def profile(ds: dict, refresh: bool = False) -> dict:
    path = _cache(ds)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    out = _profile(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, default=str))
    return out


def _profile(ds: dict) -> dict:
    con = engine.connect([ds])
    t = q(ds["slug"])
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    schema = con.execute(f"DESCRIBE SELECT * FROM {t}").fetchall()
    target = ds.get("target")
    cols: dict = {}
    for name, typ, *_ in schema:
        if name == SPLIT:
            continue
        cols[name] = _column(con, t, name, typ, n)
    distinct_rows = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * EXCLUDE ({q(SPLIT)}) FROM {t})" if any(s[0] == SPLIT for s in schema)
                                else f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {t})").fetchone()[0]
    dup = n - distinct_rows
    sample = con.execute(f"SELECT * FROM {t} USING SAMPLE reservoir({SAMPLE} ROWS) REPEATABLE (0)").fetchdf() if n > SAMPLE else con.execute(f"SELECT * FROM {t}").fetchdf()
    split_counts = dict(con.execute(f"SELECT {q(SPLIT)}, COUNT(*) FROM {t} GROUP BY 1").fetchall()) if any(s[0] == SPLIT for s in schema) else None
    out = {"rows": n, "columns": cols, "duplicates": int(dup), "split": split_counts, "sample_rows": int(len(sample)), "sampled": n > SAMPLE,
           "missing_cells": _f(sum(c["missing"] * n for c in cols.values()) / max(n * max(len(cols), 1), 1)),
           "target": target, "bytes": (ds.get("build") or {}).get("bytes")}
    out["correlations"] = _correlations(sample, cols)
    out["missingness"] = _missingness(sample, cols)
    if target and target in sample:
        out["target_analysis"] = _target(sample, cols, target, ds.get("task"))
    out["time"] = _time(con, t, cols)
    out["alerts"] = _alerts(out, cols, target)
    con.close()
    return out


def kind_of(typ: str, unique: int, n: int, avg_len: float | None) -> str:
    if typ == "BOOLEAN":
        return "boolean"
    if TEMPORAL.match(typ):
        return "datetime"
    if NUMERIC.match(typ):
        if unique <= 2:
            return "boolean"
        if unique <= 20 and "INT" in typ:
            return "discrete"
        return "numeric"
    if avg_len and avg_len > 40 and unique > 0.5 * n:
        return "text"
    return "categorical"


def _column(con, t: str, name: str, typ: str, n: int) -> dict:
    c = q(name)
    nulls, uniq = con.execute(f"SELECT COUNT(*) - COUNT({c}), approx_count_distinct({c}) FROM {t}").fetchone()
    avg_len = None
    if not NUMERIC.match(typ) and not TEMPORAL.match(typ) and typ != "BOOLEAN":
        avg_len = _f(con.execute(f"SELECT avg(length(CAST({c} AS VARCHAR))) FROM {t}").fetchone()[0])
    kind = kind_of(typ, uniq or 0, n, avg_len)
    out: dict = {"type": typ, "kind": kind, "missing": (nulls / n) if n else 0.0, "unique": int(uniq or 0)}
    if NUMERIC.match(typ):
        r = con.execute(f"SELECT min({c}), max({c}), avg({c}), stddev_samp({c}), quantile_cont({c}, [0.01, 0.25, 0.5, 0.75, 0.99]), "
                        f"skewness({c}), kurtosis({c}), COUNT(*) FILTER (WHERE {c} = 0), COUNT(*) FILTER (WHERE {c} < 0) FROM {t}").fetchone()
        lo, hi, mean, std, qs, skew, kurt, zeros, neg = r
        qs = qs or [None] * 5
        out.update(min=_f(lo), max=_f(hi), mean=_f(mean), std=_f(std), p01=_f(qs[0]), q1=_f(qs[1]), median=_f(qs[2]), q3=_f(qs[3]),
                   p99=_f(qs[4]), skew=_f(skew), kurtosis=_f(kurt), zeros=(zeros or 0) / max(n, 1), negative=(neg or 0) / max(n, 1))
        if qs[1] is not None and qs[3] is not None:
            iqr = qs[3] - qs[1]
            lo_f, hi_f = qs[1] - 1.5 * iqr, qs[3] + 1.5 * iqr
            out["outliers"] = con.execute(f"SELECT COUNT(*) FILTER (WHERE {c} < {lo_f} OR {c} > {hi_f}) FROM {t}").fetchone()[0] / max(n, 1)
        out["hist"] = histogram(con, t, name, 30, lo, hi) if lo is not None and hi is not None else None
    elif TEMPORAL.match(typ):
        lo, hi = con.execute(f"SELECT min({c}), max({c}) FROM {t}").fetchone()
        out.update(min=str(lo) if lo is not None else None, max=str(hi) if hi is not None else None)
    if kind in ("categorical", "boolean", "discrete", "text") or not NUMERIC.match(typ):
        top = con.execute(f"SELECT CAST({c} AS VARCHAR) AS v, COUNT(*) AS n FROM {t} WHERE {c} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall()
        out["top"] = [{"value": v, "count": k, "share": k / max(n, 1)} for v, k in top]
        if not NUMERIC.match(typ) and not TEMPORAL.match(typ) and typ != "BOOLEAN":
            out["numeric_like"] = _f(con.execute(f"SELECT AVG(CASE WHEN TRY_CAST({c} AS DOUBLE) IS NOT NULL THEN 1 ELSE 0 END) FROM {t} "
                                                  f"WHERE {c} IS NOT NULL").fetchone()[0])
            out["avg_length"] = avg_len
    return out


def histogram(con, t: str, name: str, bins: int, lo=None, hi=None, where: str = "") -> dict:
    c = q(name)
    if lo is None or hi is None:
        lo, hi = con.execute(f"SELECT min({c}), max({c}) FROM {t} {where}").fetchone()
    if lo is None:
        return {"edges": [], "counts": []}
    lo, hi = float(lo), float(hi)
    if hi == lo:
        n = con.execute(f"SELECT COUNT({c}) FROM {t} {where}").fetchone()[0]
        return {"edges": [lo, lo + 1], "counts": [n]}
    w = (hi - lo) / bins
    rows = con.execute(f"SELECT least(floor((CAST({c} AS DOUBLE) - {lo}) / {w}), {bins - 1})::INT AS b, COUNT(*) FROM {t} "
                       f"{where + ' AND' if where else 'WHERE'} {c} IS NOT NULL GROUP BY 1").fetchall()
    counts = [0] * bins
    for b, k in rows:
        if b is not None and 0 <= b < bins:
            counts[b] = k
    return {"edges": [lo + w * i for i in range(bins + 1)], "counts": counts}


def _numeric_cols(cols: dict) -> list[str]:
    return [k for k, v in cols.items() if NUMERIC.match(v["type"]) and v["kind"] != "boolean" or v["kind"] == "boolean" and NUMERIC.match(v["type"])]


def _correlations(sample: pd.DataFrame, cols: dict) -> dict | None:
    num = [c for c in _numeric_cols(cols) if c in sample and sample[c].nunique() > 1][:40]
    if len(num) < 2:
        return None
    X = sample[num].astype(float)
    pear = X.corr(method="pearson").round(4)
    spear = X.rank().corr(method="pearson").round(4)
    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            r = pear.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.7:
                pairs.append({"a": a, "b": b, "r": float(r)})
    clean = lambda m: [[None if pd.isna(v) else float(v) for v in row] for row in m.values]
    return {"columns": num, "pearson": clean(pear), "spearman": clean(spear), "strong_pairs": sorted(pairs, key=lambda p: -abs(p["r"]))[:20]}


def _missingness(sample: pd.DataFrame, cols: dict) -> dict:
    miss = [c for c, v in cols.items() if v["missing"] > 0 and c in sample]
    out = {"columns": {c: cols[c]["missing"] for c in miss}, "rows_with_missing": float(sample.isna().any(axis=1).mean()) if len(sample) else 0.0}
    if len(miss) >= 2:
        m = sample[miss].isna().astype(float)
        cm = m.corr().round(3)
        out["co_missing"] = {"columns": miss[:30], "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in cm.iloc[:30, :30].values]}
    return out


def _target(sample: pd.DataFrame, cols: dict, target: str, task: str | None) -> dict:
    y = sample[target]
    classif = (task == "classification") or (task is None and (y.dtype == object or y.nunique() <= 20))
    out: dict = {"task": "classification" if classif else "regression"}
    feats = [c for c in cols if c != target and c in sample]
    if classif:
        out["balance"] = {str(k): float(v) for k, v in y.astype(str).value_counts(normalize=True).head(20).items()}
    assoc = []
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    yy = y.astype(str) if classif else pd.to_numeric(y, errors="coerce")
    ok = yy.notna()
    for c in feats[:60]:
        x = sample.loc[ok, c]
        kind = cols[c]["kind"]
        try:
            if NUMERIC.match(cols[c]["type"]):
                xv = pd.to_numeric(x, errors="coerce").fillna(x.median() if x.notna().any() else 0).to_numpy().reshape(-1, 1)
                mi = (mutual_info_classif if classif else mutual_info_regression)(xv, yy[ok], random_state=0)[0]
                r = None if classif else _f(np.corrcoef(xv[:, 0], yy[ok])[0, 1])
            else:
                codes = x.astype(str).astype("category").cat.codes.to_numpy().reshape(-1, 1)
                mi = (mutual_info_classif if classif else mutual_info_regression)(codes, yy[ok], discrete_features=True, random_state=0)[0]
                r = None
            row = {"column": c, "kind": kind, "mutual_info": _f(mi), "corr": r}
            if classif and not NUMERIC.match(cols[c]["type"]):
                top = x.astype(str).value_counts().head(8).index
                pos = yy[ok].value_counts().index[-1] if yy[ok].nunique() == 2 else None
                if pos is not None:
                    row["rate_by_value"] = {v: float((yy[ok][x.astype(str) == v] == pos).mean()) for v in top}
                    row["positive_class"] = pos
            elif classif:
                row["mean_by_class"] = {str(k): _f(v) for k, v in pd.to_numeric(x, errors="coerce").groupby(yy[ok]).mean().head(10).items()}
            assoc.append(row)
        except Exception:
            continue
    out["associations"] = sorted(assoc, key=lambda r: -(r["mutual_info"] or 0))
    return out


def _time(con, t: str, cols: dict) -> dict | None:
    tcols = [c for c, v in cols.items() if v["kind"] == "datetime"]
    if not tcols:
        return None
    c = tcols[0]
    lo, hi = con.execute(f"SELECT min({q(c)}), max({q(c)}) FROM {t}").fetchone()
    if lo is None:
        return None
    span = (pd.Timestamp(hi) - pd.Timestamp(lo)).total_seconds()
    unit = "hour" if span <= 3 * 86400 else "day" if span <= 180 * 86400 else "week" if span <= 3 * 365 * 86400 else "month"
    rows = con.execute(f"SELECT date_trunc('{unit}', CAST({q(c)} AS TIMESTAMP)) AS p, COUNT(*) FROM {t} WHERE {q(c)} IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall()
    return {"column": c, "unit": unit, "periods": [str(p) for p, _ in rows], "counts": [k for _, k in rows]}


def _alerts(prof: dict, cols: dict, target: str | None) -> list[dict]:
    a = []
    n = prof["rows"]

    def add(sev, col, kind, msg):
        a.append({"severity": sev, "column": col, "kind": kind, "message": msg})
    if prof["duplicates"]:
        add("warning", None, "duplicates", f"{prof['duplicates']:,} duplicate rows ({prof['duplicates'] / max(n, 1):.1%})")
    for c, v in cols.items():
        if v["unique"] <= 1:
            add("warning", c, "constant", f"'{c}' has a single value — it carries no information")
            continue
        if v["missing"] >= 0.5:
            add("critical", c, "missing", f"'{c}' is {v['missing']:.0%} missing")
        elif v["missing"] >= 0.05:
            add("warning", c, "missing", f"'{c}' is {v['missing']:.1%} missing")
        if v["kind"] == "categorical" and v["unique"] > 50 and v["unique"] > 0.05 * n:
            add("info", c, "cardinality", f"'{c}' has {v['unique']:,} categories — one-hot encoding would explode; consider grouping rare values")
        if v["unique"] >= 0.98 * n and n > 100 and v["kind"] in ("categorical", "numeric", "discrete") and c != target:
            add("warning", c, "id_like", f"'{c}' is unique for almost every row — likely an ID; drop it before modeling")
        if v.get("skew") is not None and abs(v["skew"]) > 2 and v["kind"] == "numeric":
            add("info", c, "skew", f"'{c}' is heavily skewed (skew {v['skew']:.1f}) — consider a log transform")
        if (v.get("outliers") or 0) > 0.05:
            add("info", c, "outliers", f"'{c}' has {v['outliers']:.1%} values outside the IQR fences")
        if v.get("numeric_like") and v["numeric_like"] > 0.95:
            add("warning", c, "type", f"'{c}' is stored as text but {v['numeric_like']:.0%} of values are numbers — change its type")
    ta = prof.get("target_analysis")
    if ta and ta["task"] == "classification" and ta.get("balance"):
        mn = min(ta["balance"].values())
        if mn < 0.1:
            add("warning", target, "imbalance", f"The rarest class of '{target}' is {mn:.1%} of rows — use stratified splits and AUC / F1, not accuracy")
    if ta:
        for r in ta["associations"][:5]:
            if (r.get("corr") is not None and abs(r["corr"]) > 0.95) or (r.get("mutual_info") or 0) > 0.9:
                add("critical", r["column"], "leakage", f"'{r['column']}' predicts '{target}' almost perfectly — check it isn't derived from the target (leakage)")
    corr = prof.get("correlations")
    if corr:
        for p in corr["strong_pairs"][:5]:
            if abs(p["r"]) >= 0.95:
                add("info", p["a"], "collinear", f"'{p['a']}' and '{p['b']}' are almost collinear (r = {p['r']:.2f}) — one may be redundant")
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(a, key=lambda x: order[x["severity"]])


# ============================================================================= rows / column detail
def rows(ds: dict, offset: int = 0, limit: int = 100, sort: str | None = None, desc: bool = False, where: str = "") -> dict:
    con = engine.connect([ds])
    t = q(ds["slug"])
    from .steps import check_expr
    w = f"WHERE {check_expr(where)}" if where else ""
    total = con.execute(f"SELECT COUNT(*) FROM {t} {w}").fetchone()[0]
    order = f"ORDER BY {q(sort)} {'DESC' if desc else 'ASC'} NULLS LAST" if sort else ""
    df = con.execute(f"SELECT * FROM {t} {w} {order} LIMIT {int(limit)} OFFSET {int(offset)}").fetchdf()
    con.close()
    return {"total": total, "offset": offset, **json.loads(df.to_json(orient="split", date_format="iso", default_handler=str, index=False))}


def column_detail(ds: dict, col: str, by: str | None = None, bins: int = 40) -> dict:
    con = engine.connect([ds])
    t = q(ds["slug"])
    typ = dict((r[0], r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {t}").fetchall())[col]
    out: dict = {"column": col, "type": typ}
    if NUMERIC.match(typ):
        lo, hi = con.execute(f"SELECT min({q(col)}), max({q(col)}) FROM {t}").fetchone()
        out["hist"] = histogram(con, t, col, bins, lo, hi)
        if by:
            groups = [r[0] for r in con.execute(f"SELECT {q(by)} FROM {t} GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 8").fetchall()]
            out["by"] = {"column": by, "groups": {str(g): histogram(con, t, col, bins, lo, hi, f"WHERE {q(by)} {'IS NULL' if g is None else '= ' + _lit(g)}")
                                                  for g in groups}}
    top = con.execute(f"SELECT CAST({q(col)} AS VARCHAR), COUNT(*) FROM {t} GROUP BY 1 ORDER BY 2 DESC LIMIT 50").fetchall()
    out["top"] = [{"value": v, "count": k} for v, k in top]
    con.close()
    return out


def _lit(v):
    from .steps import lit
    return lit(v)
