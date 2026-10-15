"""Chart data and subset comparison. Aggregation happens in DuckDB (so 10M-row tables chart instantly); scatter plots
use a reproducible sample. Several datasets can be charted together: histograms share bin edges and are normalized,
and comparisons report sample sizes, confidence intervals and two-sample tests so noise isn't mistaken for a difference."""
from __future__ import annotations

import math

import numpy as np

from . import engine
from .eda import NUMERIC, TEMPORAL, _f, histogram
from .engine import quote_ident as q
from .steps import check_expr, lit

SCATTER_POINTS = 4000
AGGS = {"count": "COUNT(*)", "sum": "sum({})", "avg": "avg({})", "median": "median({})", "min": "min({})", "max": "max({})",
        "stddev": "stddev_samp({})", "count_distinct": "count(DISTINCT {})"}


def _types(con, t):
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM {t}").fetchall()}


def _where(spec_filter: str | None, extra: str = "") -> str:
    parts = [p for p in [check_expr(spec_filter) if spec_filter else "", extra] if p]
    return ("WHERE " + " AND ".join(f"({p})" for p in parts)) if parts else ""


def chart(datasets: list[dict], spec: dict) -> dict:
    """spec: type (histogram | ecdf | box | bar | scatter | line | heatmap), x, y, color, agg, bins, filter, normalize, top."""
    con = engine.connect(datasets)
    kind = spec.get("type", "histogram")
    x, y, color = spec.get("x"), spec.get("y"), spec.get("color") or None
    agg = spec.get("agg", "count")
    out = {"type": kind, "series": [], "x": x, "y": y, "color": color, "agg": agg}
    flt = spec.get("filter")
    multi = len(datasets) > 1
    if multi and color:
        color = None                                    # one dimension of grouping at a time: the subsets themselves
        out["note"] = "Color grouping is replaced by the datasets when comparing several."
    try:
        if kind in ("histogram", "ecdf", "box"):
            groups = _groups(con, datasets, color, flt)
            if kind == "histogram":
                lo = hi = None
                for t, w, _ in groups:
                    a, b = con.execute(f"SELECT min({q(x)}), max({q(x)}) FROM {t} {w}").fetchone()
                    if a is not None:
                        lo, hi = (a if lo is None else min(lo, a)), (b if hi is None else max(hi, b))
                if NUMERIC.match(_types(con, groups[0][0])[x]):
                    for t, w, label in groups:
                        h = histogram(con, t, x, int(spec.get("bins") or 40), lo, hi, w)
                        tot = sum(h["counts"]) or 1
                        out["series"].append({"name": label, "edges": h["edges"], "counts": h["counts"], "n": sum(h["counts"]),
                                              "values": [c / tot for c in h["counts"]] if spec.get("normalize", multi) else h["counts"]})
                    out["normalized"] = bool(spec.get("normalize", multi))
                else:
                    kind = out["type"] = "bar"
                    return _bar(con, datasets, {**spec, "type": "bar"}, out, groups)
            elif kind == "ecdf":
                for t, w, label in groups:
                    qs = con.execute(f"SELECT quantile_cont({q(x)}, {[i / 100 for i in range(101)]}), COUNT({q(x)}) FROM {t} {w}").fetchone()
                    out["series"].append({"name": label, "x": [_f(v) for v in (qs[0] or [])], "y": [i / 100 for i in range(101)], "n": qs[1]})
            else:
                for t, w, label in groups:
                    r = con.execute(f"SELECT quantile_cont({q(y or x)}, [0.05, 0.25, 0.5, 0.75, 0.95]), avg({q(y or x)}), COUNT({q(y or x)}) "
                                    f"FROM {t} {w}").fetchone()
                    out["series"].append({"name": label, "q": [_f(v) for v in (r[0] or [])], "mean": _f(r[1]), "n": r[2]})
        elif kind == "bar":
            return _bar(con, datasets, spec, out, _groups(con, datasets, color, flt))
        elif kind == "scatter":
            for t, w, label in _groups(con, datasets, color, flt):
                df = con.execute(f"SELECT {q(x)} AS x, {q(y)} AS y FROM {t} {w} USING SAMPLE reservoir({SCATTER_POINTS} ROWS) REPEATABLE (0)").fetchdf()
                out["series"].append({"name": label, "x": [_jsonable(v) for v in df["x"]], "y": [_jsonable(v) for v in df["y"]], "n": len(df)})
            out["sampled_per_series"] = SCATTER_POINTS
        elif kind == "line":
            typ = _types(con, q(datasets[0]["slug"]))[x]
            bucket = spec.get("bucket") or ("day" if TEMPORAL.match(typ) else None)
            xe = f"date_trunc('{bucket}', CAST({q(x)} AS TIMESTAMP))" if bucket else q(x)
            ye = AGGS[agg].format(q(y)) if agg != "count" else "COUNT(*)"
            for t, w, label in _groups(con, datasets, color, flt):
                rows = con.execute(f"SELECT {xe} AS x, {ye} AS v FROM {t} {w} GROUP BY 1 ORDER BY 1 LIMIT 5000").fetchall()
                out["series"].append({"name": label, "x": [_jsonable(a) for a, _ in rows], "y": [_f(b) for _, b in rows]})
            out["bucket"] = bucket
        elif kind == "heatmap":
            t = q(datasets[0]["slug"])
            w = _where(flt)
            ye = AGGS[agg].format(q(spec.get("value"))) if agg != "count" else "COUNT(*)"
            xs = [r[0] for r in con.execute(f"SELECT CAST({q(x)} AS VARCHAR) FROM {t} {w} GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 30").fetchall()]
            ys = [r[0] for r in con.execute(f"SELECT CAST({q(y)} AS VARCHAR) FROM {t} {w} GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 30").fetchall()]
            cells = {(a, b): v for a, b, v in con.execute(f"SELECT CAST({q(x)} AS VARCHAR), CAST({q(y)} AS VARCHAR), {ye} FROM {t} {w} GROUP BY 1, 2").fetchall()}
            out["series"] = [{"name": datasets[0]["name"], "xs": xs, "ys": ys, "z": [[_f(cells.get((a, b))) for a in xs] for b in ys]}]
        else:
            raise ValueError(f"unknown chart type {kind}")
    finally:
        con.close()
    return out


def _groups(con, datasets, color, flt):
    """[(table, where, label)] — one per dataset, or per color value (top 8) for a single dataset."""
    if len(datasets) > 1 or not color:
        return [(q(d["slug"]), _where(flt), d["name"]) for d in datasets]
    t = q(datasets[0]["slug"])
    vals = [r[0] for r in con.execute(f"SELECT {q(color)} FROM {t} {_where(flt)} GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 8").fetchall()]
    return [(t, _where(flt, f"{q(color)} IS NULL" if v is None else f"{q(color)} = {lit(v if not isinstance(v, (int, float)) else v)}"),
             "(missing)" if v is None else str(v)) for v in vals]


def _bar(con, datasets, spec, out, groups):
    x, y, agg = spec["x"], spec.get("y"), spec.get("agg", "count")
    top = int(spec.get("top") or 20)
    t0, w0, _ = groups[0]
    cats = [r[0] for r in con.execute(f"SELECT CAST({q(x)} AS VARCHAR) FROM {t0} {w0} GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {top}").fetchall()]
    ye = AGGS[agg].format(q(y)) if agg != "count" else "COUNT(*)"
    for t, w, label in groups:
        rows = dict(con.execute(f"SELECT CAST({q(x)} AS VARCHAR), {ye} FROM {t} {w} GROUP BY 1").fetchall())
        vals = [_f(rows.get(c)) for c in cats]
        n = sum(v or 0 for v in vals)
        norm = agg == "count" and spec.get("normalize", len(groups) > 1)
        out["series"].append({"name": label, "categories": cats, "values": [(v or 0) / (n or 1) for v in vals] if norm else vals})
    out["normalized"] = bool(agg == "count" and spec.get("normalize", len(groups) > 1))
    con.close()
    return out


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v if not (isinstance(v, float) and not math.isfinite(v)) else None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(v)


# ============================================================================= compare subsets
def compare(datasets: list[dict], column: str, flt: str | None = None) -> dict:
    """Side-by-side statistics for one column across datasets, with tests against the first (the reference)."""
    from scipy import stats
    con = engine.connect(datasets)
    w = _where(flt)
    typ = _types(con, q(datasets[0]["slug"])).get(column)
    if typ is None:
        raise ValueError(f"'{column}' isn't a column of '{datasets[0]['name']}'")
    numeric = bool(NUMERIC.match(typ))
    rows, samples, cats = [], [], None
    for d in datasets:
        t = q(d["slug"])
        if column not in _types(con, t):
            rows.append({"dataset": d["name"], "missing_column": True})
            samples.append(None)
            continue
        if numeric:
            n, miss, mean, sd, med, q1, q3 = con.execute(
                f"SELECT COUNT({q(column)}), COUNT(*) - COUNT({q(column)}), avg({q(column)}), stddev_samp({q(column)}), median({q(column)}), "
                f"quantile_cont({q(column)}, 0.25), quantile_cont({q(column)}, 0.75) FROM {t} {w}").fetchone()
            ci = 1.96 * (sd or 0) / math.sqrt(n) if n else None
            rows.append({"dataset": d["name"], "n": n, "missing": miss, "mean": _f(mean), "ci95": _f(ci), "std": _f(sd), "median": _f(med),
                         "q1": _f(q1), "q3": _f(q3)})
            samples.append(con.execute(f"SELECT CAST({q(column)} AS DOUBLE) FROM {t} {w} {'AND' if w else 'WHERE'} {q(column)} IS NOT NULL "
                                       f"USING SAMPLE reservoir(50000 ROWS) REPEATABLE (0)").fetchdf().iloc[:, 0].to_numpy())
        else:
            counts = dict(con.execute(f"SELECT CAST({q(column)} AS VARCHAR), COUNT(*) FROM {t} {w} GROUP BY 1").fetchall())
            n = sum(counts.values())
            if cats is None:
                cats = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:15]
            rows.append({"dataset": d["name"], "n": n, "shares": {c: counts.get(c, 0) / max(n, 1) for c in cats},
                         "other": sum(v for k, v in counts.items() if k not in cats) / max(n, 1)})
            samples.append(counts)
    tests = []
    ref = samples[0]
    for d, s in list(zip(datasets, samples))[1:]:
        if s is None or ref is None:
            continue
        if numeric:
            if len(ref) > 1 and len(s) > 1:
                ks = stats.ks_2samp(ref, s)
                tt = stats.ttest_ind(ref, s, equal_var=False)
                tests.append({"dataset": d["name"], "vs": datasets[0]["name"], "ks_stat": _f(ks.statistic), "ks_p": _f(ks.pvalue),
                              "welch_t_p": _f(tt.pvalue), "mean_diff": _f(float(np.mean(s) - np.mean(ref))),
                              "effect_size_d": _f((np.mean(s) - np.mean(ref)) / (np.sqrt((np.var(s) + np.var(ref)) / 2) or 1))})
        else:
            keys = sorted(set(ref) | set(s))
            table = np.array([[ref.get(k, 0) for k in keys], [s.get(k, 0) for k in keys]])
            table = table[:, table.sum(0) > 0]
            if table.shape[1] > 1:
                chi2, p, _, _ = stats.chi2_contingency(table)
                v = math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
                tests.append({"dataset": d["name"], "vs": datasets[0]["name"], "chi2_p": _f(p), "cramers_v": _f(v)})
    con.close()
    from .catalog import value_transforms
    tf = {d["name"]: value_transforms(d, column) for d in datasets}
    warnings = []
    if len({tuple(v) for v in tf.values()}) > 1:
        warnings.append(f"'{column}' is on different scales across these datasets: " +
                        "; ".join(f"{n}: {', '.join(v) if v else 'raw values'}" for n, v in tf.items()) +
                        ". Differences reflect the transformation, not the data — compare datasets with matching preprocessing "
                        "(e.g. filter the raw dataset instead of a scaled one).")
    return {"column": column, "numeric": numeric, "rows": rows, "tests": tests, "categories": cats, "warnings": warnings, "transforms": tf,
            "note": "Tests compare each dataset with the first one. Overlapping subsets (one contains the other) violate the tests' "
                    "independence assumption — read p-values as indicative."}
