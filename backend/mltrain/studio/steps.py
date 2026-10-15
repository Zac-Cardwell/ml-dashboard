"""Recipe steps: the reproducible transformations of a derived dataset.

A step is {"id", "type", "params", "enabled"}. SQL steps compile to DuckDB SQL over the previous step; "python" steps
(PCA, feature selection, clustering, anomaly scores) run on a DataFrame. Steps that *learn statistics* (imputation,
scaling, quantile clipping / binning, one-hot categories, PCA, selection, clustering) are "fitted":

  * if an earlier Split step created __split, they fit on __split = 'train' rows only and apply to every row —
    the same discipline as a fit-on-train model pipeline, so validation/test rows never leak into the statistics;
  * otherwise they fit on all rows and the step carries a leakage warning (fine for exploration, not for modeling).

Fitted statistics are stored with each build, so a version is fully reproducible and inspectable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .engine import quote_ident as q

SPLIT = "__split"


@dataclass
class StepSpec:
    type: str
    label: str
    category: str
    help: str
    params: list[dict]
    fitted: bool = False          # learns statistics from the data
    python: bool = False          # runs on a DataFrame, not as SQL
    fitted_if: str | None = None  # fitted only for some parameter values (e.g. impute strategy != constant)


def P(name, type, default=None, label=None, help="", **kw):
    return {"name": name, "type": type, "default": default, "label": label or name.replace("_", " "), "help": help, **kw}


AGG_FNS = ["count", "sum", "avg", "min", "max", "median", "stddev", "count_distinct", "first", "last"]
CAST_TYPES = ["DOUBLE", "BIGINT", "INTEGER", "VARCHAR", "BOOLEAN", "DATE", "TIMESTAMP"]
OPS = ["=", "!=", ">", ">=", "<", "<=", "between", "in", "not in", "contains", "starts with", "is null", "is not null"]

SPECS = [
    # ---------------------------------------------------------------- rows
    StepSpec("filter", "Filter rows", "Rows", "Keep rows matching all (or any) conditions, or a SQL expression",
             [P("conditions", "conditions", [], help="column · operator · value"), P("combine", "choice", "and", choices=["and", "or"]),
              P("sql", "sql_expr", "", "SQL condition", "Optional: a WHERE expression, e.g. age > 30 AND region IN ('west')")]),
    StepSpec("dedupe", "Remove duplicates", "Rows", "Drop duplicate rows (all columns, or a key)",
             [P("columns", "columns", [], help="Empty = compare all columns")]),
    StepSpec("drop_missing", "Drop missing", "Rows", "Drop rows with missing values",
             [P("columns", "columns", [], help="Empty = any column"), P("how", "choice", "any", choices=["any", "all"])]),
    StepSpec("sample", "Sample rows", "Rows", "Random sample (reproducible with the seed)",
             [P("n", "int", 10000, "rows"), P("fraction", "float", 0.0, help="Use a fraction instead of n (0 = use n)"),
              P("seed", "int", 0)]),
    StepSpec("sort", "Sort", "Rows", "Order rows", [P("columns", "columns", []), P("descending", "bool", False)]),
    StepSpec("limit", "Limit", "Rows", "Keep the first N rows", [P("n", "int", 1000)]),
    # ---------------------------------------------------------------- columns
    StepSpec("select", "Keep columns", "Columns", "Keep only these columns", [P("columns", "columns", [])]),
    StepSpec("drop", "Drop columns", "Columns", "Remove columns", [P("columns", "columns", [])]),
    StepSpec("rename", "Rename", "Columns", "Rename a column", [P("column", "column", ""), P("to", "str", "")]),
    StepSpec("cast", "Change type", "Columns", "Convert columns to another type (invalid values become NULL)",
             [P("columns", "columns", []), P("to", "choice", "DOUBLE", choices=CAST_TYPES)]),
    StepSpec("derive", "New column (formula)", "Columns", "A new column from a SQL expression",
             [P("name", "str", "new_feature"), P("expression", "sql_expr", "", help="e.g. income / NULLIF(household_size, 0)")]),
    StepSpec("datetime_parts", "Date/time parts", "Columns", "Extract year, month, day of week, hour, ... from a date/time",
             [P("column", "column", ""), P("parts", "multi", ["year", "month", "dayofweek", "hour"],
                                            choices=["year", "quarter", "month", "week", "day", "dayofweek", "dayofyear", "hour", "minute", "epoch"])]),
    StepSpec("text", "Text transform", "Columns", "Lowercase, trim, length, regex replace / extract",
             [P("column", "column", ""), P("op", "choice", "lower", choices=["lower", "upper", "trim", "length", "regex_replace", "regex_extract"]),
              P("pattern", "str", ""), P("replacement", "str", ""), P("new_name", "str", "", help="Empty = overwrite")]),
    StepSpec("math", "Math transform", "Columns", "log1p, sqrt, abs, square of numeric columns",
             [P("columns", "columns", []), P("fn", "choice", "log1p", choices=["log1p", "sqrt", "abs", "square", "sign"])]),
    StepSpec("bin", "Bin (discretize)", "Columns", "Turn a numeric column into bins", [
        P("column", "column", ""), P("method", "choice", "quantile", choices=["quantile", "equal_width", "custom"]),
        P("bins", "int", 5), P("edges", "str", "", help="custom: comma-separated edges"), P("new_name", "str", "")],
        fitted=True),
    # ---------------------------------------------------------------- cleaning (fitted)
    StepSpec("impute", "Fill missing", "Cleaning", "Fill missing values (statistics from train rows when split)", [
        P("columns", "columns", [], help="Empty = all columns with missing values"),
        P("strategy", "choice", "median", choices=["mean", "median", "mode", "constant"]), P("value", "str", "0"),
        P("add_indicator", "bool", False, help="Add <col>_was_missing flags")], fitted=True, fitted_if="strategy!=constant"),
    StepSpec("clip", "Clip outliers", "Cleaning", "Cap values at quantiles, IQR fences or fixed bounds", [
        P("columns", "columns", []), P("method", "choice", "quantile", choices=["quantile", "iqr", "fixed"]),
        P("lower", "float", 0.01), P("upper", "float", 0.99), P("iqr_k", "float", 1.5)], fitted=True, fitted_if="method!=fixed"),
    StepSpec("scale", "Scale", "Cleaning", "Standardize / min-max / robust scale numeric columns", [
        P("columns", "columns", []), P("method", "choice", "standard", choices=["standard", "minmax", "robust"])], fitted=True),
    StepSpec("one_hot", "One-hot encode", "Cleaning", "Indicator columns for the most frequent categories (+ other)", [
        P("columns", "columns", []), P("max_categories", "int", 20), P("drop_original", "bool", True)], fitted=True),
    # ---------------------------------------------------------------- combine
    StepSpec("join", "Join", "Combine", "Join another dataset on key columns", [
        P("other", "dataset", ""), P("how", "choice", "left", choices=["inner", "left", "right", "full"]),
        P("left_on", "columns", []), P("right_on", "columns_other", []), P("suffix", "str", "_r")]),
    StepSpec("union", "Append rows", "Combine", "Stack rows of other datasets (matched by column name)", [P("others", "datasets", [])]),
    StepSpec("aggregate", "Group & aggregate", "Combine", "One row per group with aggregates", [
        P("group_by", "columns", []), P("aggs", "aggs", [{"column": "*", "fn": "count", "as": "n"}])]),
    StepSpec("window", "Window feature", "Combine", "Lag / lead / rolling / cumulative features (per group, ordered)", [
        P("column", "column", ""), P("fn", "choice", "lag", choices=["lag", "lead", "rolling_mean", "rolling_sum", "rolling_std", "cumsum", "rank", "pct_change"]),
        P("n", "int", 1, help="lag/lead offset or rolling window (rows)"), P("partition_by", "columns", []), P("order_by", "column", ""),
        P("new_name", "str", "")]),
    # ---------------------------------------------------------------- modeling
    StepSpec("split", "Train / val / test split", "Modeling", "Add a __split column (random, stratified or by time). "
             "Fitted steps after it learn from train rows only; models trained on this dataset use this split.", [
        P("method", "choice", "random", choices=["random", "stratified", "time"]), P("column", "column", "", help="stratify-by or time column"),
        P("val_fraction", "float", 0.15), P("test_fraction", "float", 0.15), P("seed", "int", 0)]),
    StepSpec("sql", "SQL", "Advanced", "Any SELECT over the previous step, referenced as `input`",
             [P("sql", "sql", "SELECT *\nFROM input")]),
    # ---------------------------------------------------------------- python (fitted on samples, applied to all rows)
    StepSpec("pca", "PCA (feature reduction)", "Features", "Replace / add principal components of numeric columns", [
        P("columns", "columns", [], help="Empty = all numeric columns except the target"), P("n_components", "int", 0, help="0 = use variance"),
        P("variance", "float", 0.95, help="Keep enough components to explain this share"), P("keep_original", "bool", False),
        P("prefix", "str", "pc")], fitted=True, python=True),
    StepSpec("feature_select", "Feature selection", "Features", "Drop low-variance, highly correlated or low-information columns", [
        P("method", "choice", "correlation", choices=["variance", "correlation", "mutual_info", "f_test"]),
        P("threshold", "float", 0.95, help="variance: min variance · correlation: max |r|"), P("k", "int", 20, help="mutual_info / f_test: keep top k"),
        P("target", "column", "", help="mutual_info / f_test: target column")], fitted=True, python=True),
    StepSpec("cluster", "Cluster labels", "Features", "Add a cluster id column (KMeans, Gaussian mixture, HDBSCAN, ...)", [
        P("columns", "columns", []), P("algorithm", "choice", "kmeans", choices=["kmeans", "gmm", "hdbscan", "dbscan"]),
        P("k", "int", 5), P("eps", "float", 0.5, help="dbscan"), P("min_cluster_size", "int", 25, help="hdbscan"),
        P("name", "str", "cluster"), P("seed", "int", 0)], fitted=True, python=True),
    StepSpec("anomaly", "Anomaly score", "Features", "Add an anomaly score and flag (Isolation Forest / LOF)", [
        P("columns", "columns", []), P("algorithm", "choice", "isolation_forest", choices=["isolation_forest", "lof"]),
        P("contamination", "float", 0.01), P("name", "str", "anomaly"), P("seed", "int", 0)], fitted=True, python=True),
]
SPEC = {s.type: s for s in SPECS}
FIT_SAMPLE = 200_000            # python steps fit on at most this many (train) rows, then transform every row


def catalog() -> list[dict]:
    return [{"type": s.type, "label": s.label, "category": s.category, "help": s.help, "params": s.params,
             "fitted": s.fitted, "python": s.python} for s in SPECS]


def is_fitted(step: dict) -> bool:
    s = SPEC[step["type"]]
    if not s.fitted:
        return False
    if s.fitted_if:
        k, v = s.fitted_if.split("!=")
        return (step.get("params") or {}).get(k, _default(s, k)) != v
    return True


def _default(s: StepSpec, name: str):
    return next((p["default"] for p in s.params if p["name"] == name), None)


def params_of(step: dict) -> dict:
    s = SPEC[step["type"]]
    return {**{p["name"]: p["default"] for p in s.params}, **(step.get("params") or {})}


# ============================================================================= SQL helpers
def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", s):
        return s
    return "'" + s.replace("'", "''") + "'"


def check_expr(expr: str) -> str:
    """Basic guard for user SQL fragments inside a SELECT (the connection is sandboxed anyway)."""
    e = (expr or "").strip()
    if ";" in e:
        raise ValueError("SQL fragments can't contain ';'")
    return e


def condition_sql(c: dict) -> str:
    col, op, v = q(c["column"]), c.get("op", "="), c.get("value")
    if op == "is null":
        return f"{col} IS NULL"
    if op == "is not null":
        return f"{col} IS NOT NULL"
    if op == "between":
        lo, hi = (v or [None, None]) if isinstance(v, list) else str(v).split(",")
        return f"{col} BETWEEN {lit(lo)} AND {lit(hi)}"
    if op in ("in", "not in"):
        vals = v if isinstance(v, list) else [x.strip() for x in str(v).split(",") if x.strip()]
        return f"{col} {'NOT ' if op == 'not in' else ''}IN ({', '.join(lit(x) for x in vals)})"
    if op == "contains":
        return f"CAST({col} AS VARCHAR) ILIKE {lit('%' + str(v) + '%')}"
    if op == "starts with":
        return f"CAST({col} AS VARCHAR) ILIKE {lit(str(v) + '%')}"
    return f"{col} {op} {lit(v)}"


# ============================================================================= compiler
class Context:
    """State while executing a recipe: the DuckDB connection, the current relation name, its columns,
    whether a split exists, the target column, and the collected fitted statistics / warnings."""

    def __init__(self, con, rel: str, target: str | None, datasets: dict, sample_rows: int | None = None):
        self.con, self.rel, self.target, self.datasets = con, rel, target, datasets
        self.n = 0
        self.fitted: dict = {}
        self.warnings: list[dict] = []
        self.sample_rows = sample_rows
        self.trace: list[dict] = []

    def columns(self) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.con.execute(f"DESCRIBE SELECT * FROM {self.rel}").fetchall()]

    def colnames(self) -> list[str]:
        return [c for c, _ in self.columns()]

    def numeric(self, cols=None, exclude_target=True) -> list[str]:
        num = [c for c, t in self.columns() if re.match(r"(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|FLOAT|DOUBLE|DECIMAL|REAL|U?(TINY|SMALL|BIG)?INT)", t)]
        num = [c for c in num if c != SPLIT and not (exclude_target and c == self.target)]
        if cols:
            missing = [c for c in cols if c not in self.colnames()]
            if missing:
                raise ValueError(f"Unknown columns: {missing}")
            return [c for c in cols]
        return num

    def has_split(self) -> bool:
        return SPLIT in self.colnames()

    def fit_rel(self) -> str:
        """Relation the statistics are learned from: train rows when split."""
        return f"(SELECT * FROM {self.rel} WHERE {q(SPLIT)} = 'train')" if self.has_split() else self.rel

    def new(self, sql: str) -> None:
        self.n += 1
        name = f"__s{self.n}"
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")
        self.rel = name


def run(ctx: Context, steps: list[dict]) -> Context:
    for i, step in enumerate(steps):
        if not step.get("enabled", True):
            continue
        t = step["type"]
        if t not in SPEC:
            raise ValueError(f"Unknown step type '{t}'")
        p = params_of(step)
        before = ctx.rel
        try:
            if is_fitted(step) and not ctx.has_split():
                ctx.warnings.append({"step": step.get("id"), "index": i, "kind": "leakage",
                                     "message": f"'{SPEC[t].label}' learns statistics from all rows. Add a Train/val/test split "
                                                f"step before it so it fits on training rows only (otherwise test rows leak into training)."})
            stats = (_python if SPEC[t].python else _sql)(ctx, t, p, step)
            if stats is not None:
                ctx.fitted[step.get("id") or str(i)] = stats
        except Exception as e:
            raise ValueError(f"Step {i + 1} ({SPEC[t].label}): {e}") from None
        rows = ctx.con.execute(f"SELECT COUNT(*) FROM {ctx.rel}").fetchone()[0]
        ctx.trace.append({"step": step.get("id"), "index": i, "rows": rows, "columns": len(ctx.colnames())})
        if rows == 0 and before != ctx.rel:
            ctx.warnings.append({"step": step.get("id"), "index": i, "kind": "empty", "message": f"'{SPEC[t].label}' left no rows"})
    return ctx


# ----------------------------------------------------------------------------- SQL steps
def _sql(ctx: Context, t: str, p: dict, step: dict):
    R, cols = ctx.rel, ctx.colnames()
    ex = lambda keep: ", ".join(q(c) for c in cols if c not in keep)
    stats = None
    if t == "filter":
        parts = [condition_sql(c) for c in p["conditions"] if c.get("column")]
        if (p.get("sql") or "").strip():
            parts.append(f"({check_expr(p['sql'])})")
        if not parts:
            return None
        ctx.new(f"SELECT * FROM {R} WHERE {(' ' + p['combine'].upper() + ' ').join(parts)}")
    elif t == "dedupe":
        key = ", ".join(q(c) for c in p["columns"]) if p["columns"] else ", ".join(q(c) for c in cols)
        ctx.new(f"SELECT * EXCLUDE (__rn) FROM (SELECT *, row_number() OVER (PARTITION BY {key}) AS __rn FROM {R}) WHERE __rn = 1")
    elif t == "drop_missing":
        cs = p["columns"] or cols
        joiner = " OR " if p["how"] == "any" else " AND "
        ctx.new(f"SELECT * FROM {R} WHERE NOT ({joiner.join(f'{q(c)} IS NULL' for c in cs)})")
    elif t == "sample":
        seed = int(p["seed"])
        if p.get("fraction"):
            ctx.new(f"SELECT * FROM {R} USING SAMPLE {float(p['fraction']) * 100}% (bernoulli, {seed})")
        else:
            ctx.new(f"SELECT * FROM {R} USING SAMPLE reservoir({int(p['n'])} ROWS) REPEATABLE ({seed})")
    elif t == "sort":
        ctx.new(f"SELECT * FROM {R} ORDER BY {', '.join(q(c) + (' DESC' if p['descending'] else '') for c in p['columns'])}")
    elif t == "limit":
        ctx.new(f"SELECT * FROM {R} LIMIT {int(p['n'])}")
    elif t == "select":
        keep = list(p["columns"]) + ([SPLIT] if ctx.has_split() and SPLIT not in p["columns"] else [])
        ctx.new(f"SELECT {', '.join(q(c) for c in keep)} FROM {R}")
    elif t == "drop":
        ctx.new(f"SELECT {ex(set(p['columns']))} FROM {R}")
    elif t == "rename":
        ctx.new(f"SELECT * RENAME ({q(p['column'])} AS {q(p['to'])}) FROM {R}")
    elif t == "cast":
        repl = ", ".join(f"TRY_CAST({q(c)} AS {p['to']}) AS {q(c)}" for c in p["columns"])
        ctx.new(f"SELECT * REPLACE ({repl}) FROM {R}")
    elif t == "derive":
        ctx.new(f"SELECT *, ({check_expr(p['expression'])}) AS {q(p['name'])} FROM {R}")
    elif t == "datetime_parts":
        c = q(p["column"])
        ts = f"TRY_CAST({c} AS TIMESTAMP)"
        exprs = [f"{'epoch' if part == 'epoch' else 'date_part'}({lit(part) + ', ' if part != 'epoch' else ''}{ts}) AS {q(p['column'] + '_' + part)}"
                 for part in p["parts"]]
        ctx.new(f"SELECT *, {', '.join(exprs)} FROM {R}")
    elif t == "text":
        c, op = q(p["column"]), p["op"]
        e = {"lower": f"lower(CAST({c} AS VARCHAR))", "upper": f"upper(CAST({c} AS VARCHAR))", "trim": f"trim(CAST({c} AS VARCHAR))",
             "length": f"length(CAST({c} AS VARCHAR))",
             "regex_replace": f"regexp_replace(CAST({c} AS VARCHAR), {lit(p['pattern'])}, {lit(p['replacement'])}, 'g')",
             "regex_extract": f"regexp_extract(CAST({c} AS VARCHAR), {lit(p['pattern'])})"}[op]
        name = p["new_name"] or p["column"]
        ctx.new(f"SELECT * REPLACE ({e} AS {q(name)}) FROM {R}" if name in cols else f"SELECT *, {e} AS {q(name)} FROM {R}")
    elif t == "math":
        f = {"log1p": "ln(1 + {})", "sqrt": "sqrt({})", "abs": "abs({})", "square": "({0} * {0})", "sign": "sign({})"}[p["fn"]]
        repl = ", ".join(f"{f.format(q(c))} AS {q(c)}" for c in p["columns"])
        ctx.new(f"SELECT * REPLACE ({repl}) FROM {R}")
    elif t == "bin":
        c, name = p["column"], p["new_name"] or p["column"] + "_bin"
        if p["method"] == "custom":
            edges = [float(x) for x in str(p["edges"]).split(",") if x.strip()]
        elif p["method"] == "equal_width":
            lo, hi = ctx.con.execute(f"SELECT min({q(c)}), max({q(c)}) FROM {ctx.fit_rel()}").fetchone()
            edges = [lo + (hi - lo) * i / p["bins"] for i in range(1, p["bins"])]
        else:
            qs = [i / p["bins"] for i in range(1, p["bins"])]
            edges = sorted(set(ctx.con.execute(f"SELECT quantile_cont({q(c)}, {qs}) FROM {ctx.fit_rel()}").fetchone()[0]))
        stats = {"edges": edges}
        case = " ".join(f"WHEN {q(c)} < {e!r} THEN {i}" for i, e in enumerate(edges))
        ctx.new(f"SELECT *, CASE WHEN {q(c)} IS NULL THEN NULL {case} ELSE {len(edges)} END AS {q(name)} FROM {R}")
    elif t == "impute":
        cs = p["columns"] or [c for c in cols if c != SPLIT and c != ctx.target and
                              ctx.con.execute(f"SELECT COUNT(*) FROM {R} WHERE {q(c)} IS NULL").fetchone()[0]]
        numeric = set(ctx.numeric(None, exclude_target=False))
        fills = {}
        for c in cs:
            if p["strategy"] == "constant":
                fills[c] = p["value"]
            elif p["strategy"] == "mode" or c not in numeric:
                fills[c] = ctx.con.execute(f"SELECT mode({q(c)}) FROM {ctx.fit_rel()}").fetchone()[0]
            else:
                agg = "avg" if p["strategy"] == "mean" else "median"
                fills[c] = ctx.con.execute(f"SELECT {agg}({q(c)}) FROM {ctx.fit_rel()}").fetchone()[0]
        stats = {"fill": {k: (v if not hasattr(v, "isoformat") else str(v)) for k, v in fills.items()}}
        ind = ", ".join(f"({q(c)} IS NULL) AS {q(c + '_was_missing')}" for c in cs) if p["add_indicator"] else ""
        repl = ", ".join(f"coalesce({q(c)}, CAST({lit(v)} AS {_type_of(ctx, c)})) AS {q(c)}" for c, v in fills.items())
        if repl:
            ctx.new(f"SELECT * REPLACE ({repl}){', ' + ind if ind else ''} FROM {R}")
    elif t == "clip":
        cs, bounds = ctx.numeric(p["columns"] or None), {}
        for c in cs:
            if p["method"] == "fixed":
                bounds[c] = (p["lower"], p["upper"])
            elif p["method"] == "quantile":
                bounds[c] = tuple(ctx.con.execute(f"SELECT quantile_cont({q(c)}, [{p['lower']}, {p['upper']}]) FROM {ctx.fit_rel()}").fetchone()[0])
            else:
                q1, q3 = ctx.con.execute(f"SELECT quantile_cont({q(c)}, [0.25, 0.75]) FROM {ctx.fit_rel()}").fetchone()[0]
                bounds[c] = (q1 - p["iqr_k"] * (q3 - q1), q3 + p["iqr_k"] * (q3 - q1))
        stats = {"bounds": bounds}
        repl = ", ".join(f"least(greatest({q(c)}, {lo!r}), {hi!r}) AS {q(c)}" for c, (lo, hi) in bounds.items() if lo is not None)
        if repl:
            ctx.new(f"SELECT * REPLACE ({repl}) FROM {R}")
    elif t == "scale":
        cs, params = ctx.numeric(p["columns"] or None), {}
        for c in cs:
            if p["method"] == "standard":
                m, s = ctx.con.execute(f"SELECT avg({q(c)}), stddev_samp({q(c)}) FROM {ctx.fit_rel()}").fetchone()
            elif p["method"] == "minmax":
                lo, hi = ctx.con.execute(f"SELECT min({q(c)}), max({q(c)}) FROM {ctx.fit_rel()}").fetchone()
                m, s = lo, (hi - lo) if hi is not None and lo is not None else None
            else:
                med, (q1, q3) = ctx.con.execute(f"SELECT median({q(c)}), quantile_cont({q(c)}, [0.25, 0.75]) FROM {ctx.fit_rel()}").fetchone()
                m, s = med, q3 - q1
            params[c] = (m, s if s else 1.0)
        stats = {"center_scale": params}
        repl = ", ".join(f"(({q(c)} - {m!r}) / {s!r}) AS {q(c)}" for c, (m, s) in params.items() if m is not None)
        if repl:
            ctx.new(f"SELECT * REPLACE ({repl}) FROM {R}")
    elif t == "one_hot":
        cats, exprs = {}, []
        for c in p["columns"]:
            top = [r[0] for r in ctx.con.execute(f"SELECT {q(c)} FROM {ctx.fit_rel()} WHERE {q(c)} IS NOT NULL GROUP BY 1 "
                                                  f"ORDER BY COUNT(*) DESC LIMIT {int(p['max_categories'])}").fetchall()]
            cats[c] = [str(v) for v in top]
            for v in top:
                exprs.append(f"CAST(CAST({q(c)} AS VARCHAR) = {lit(str(v))} AS INTEGER) AS {q(f'{c}={v}')}")
            exprs.append(f"CAST({q(c)} IS NOT NULL AND CAST({q(c)} AS VARCHAR) NOT IN ({', '.join(lit(str(v)) for v in top) or 'NULL'}) AS INTEGER) AS {q(c + '=other')}")
        stats = {"categories": cats}
        base = f"* EXCLUDE ({', '.join(q(c) for c in p['columns'])})" if p["drop_original"] and p["columns"] else "*"
        if exprs:
            ctx.new(f"SELECT {base}, {', '.join(exprs)} FROM {R}")
    elif t == "join":
        other = ctx.datasets.get(p["other"])
        if not other:
            raise ValueError("choose a dataset to join")
        from .engine import table_ref
        lo, ro = p["left_on"], p["right_on"] or p["left_on"]
        if not lo or len(lo) != len(ro):
            raise ValueError("pick the same number of key columns on both sides")
        ocols = [r[0] for r in ctx.con.execute(f"DESCRIBE SELECT * FROM {table_ref(other)}").fetchall()]
        sel = ", ".join(f"r.{q(c)} AS {q(c + p['suffix'] if c in cols else c)}" for c in ocols if c not in ro and c != SPLIT)
        on = " AND ".join(f"l.{q(a)} = r.{q(b)}" for a, b in zip(lo, ro))
        ctx.new(f"SELECT l.*{', ' + sel if sel else ''} FROM {R} l {p['how'].upper()} JOIN {table_ref(other)} r ON {on}")
    elif t == "union":
        from .engine import table_ref
        parts = [f"SELECT * FROM {R}"] + [f"SELECT * FROM {table_ref(ctx.datasets[o])}" for o in p["others"] if o in ctx.datasets]
        ctx.new(" UNION ALL BY NAME ".join(parts))
    elif t == "aggregate":
        aggs = []
        for a in p["aggs"]:
            col = "*" if a.get("column") in (None, "", "*") else q(a["column"])
            fn = a.get("fn", "count")
            e = f"count(DISTINCT {col})" if fn == "count_distinct" else f"{'stddev_samp' if fn == 'stddev' else fn}({col})"
            aggs.append(f"{e} AS {q(a.get('as') or (fn + '_' + str(a.get('column', 'n')).strip('*') or 'n'))}")
        gb = ", ".join(q(c) for c in p["group_by"])
        ctx.new(f"SELECT {gb + ', ' if gb else ''}{', '.join(aggs)} FROM {R}{' GROUP BY ' + gb if gb else ''}")
    elif t == "window":
        c, n = q(p["column"]), int(p["n"])
        part = f"PARTITION BY {', '.join(q(x) for x in p['partition_by'])}" if p["partition_by"] else ""
        order = f"ORDER BY {q(p['order_by'])}" if p["order_by"] else ""
        frame = f"ROWS BETWEEN {max(n - 1, 0)} PRECEDING AND CURRENT ROW"
        e = {"lag": f"lag({c}, {n}) OVER ({part} {order})", "lead": f"lead({c}, {n}) OVER ({part} {order})",
             "rolling_mean": f"avg({c}) OVER ({part} {order} {frame})", "rolling_sum": f"sum({c}) OVER ({part} {order} {frame})",
             "rolling_std": f"stddev_samp({c}) OVER ({part} {order} {frame})", "cumsum": f"sum({c}) OVER ({part} {order} ROWS UNBOUNDED PRECEDING)",
             "rank": f"rank() OVER ({part} ORDER BY {c})",
             "pct_change": f"({c} / NULLIF(lag({c}, {n}) OVER ({part} {order}), 0) - 1)"}[p["fn"]]
        ctx.new(f"SELECT *, {e} AS {q(p['new_name'] or p['column'] + '_' + p['fn'] + ('' if p['fn'] in ('cumsum', 'rank') else str(n)))} FROM {R}")
    elif t == "split":
        vf, tf, seed, col = float(p["val_fraction"]), float(p["test_fraction"]), int(p["seed"]), p["column"]
        if vf + tf >= 1:
            raise ValueError("val + test fractions must be < 1")
        src = f"(SELECT * EXCLUDE ({q(SPLIT)}) FROM {R})" if ctx.has_split() else R
        if p["method"] == "time":
            if not col:
                raise ValueError("choose the time column")
            inner = f"SELECT *, percent_rank() OVER (ORDER BY {q(col)}) AS __u FROM {src}"
            assign = f"CASE WHEN __u >= {1 - tf} THEN 'test' WHEN __u >= {1 - tf - vf} THEN 'val' ELSE 'train' END"
            excl = "__u"
        else:
            if p["method"] == "stratified" and not col:
                raise ValueError("choose the column to stratify by")
            numbered = f"SELECT *, row_number() OVER () AS __rn FROM {src}"      # Parquet scans keep row order
            u = (f"percent_rank() OVER (PARTITION BY {q(col)} ORDER BY hash(__rn, {seed}))" if p["method"] == "stratified"
                 else f"(hash(__rn, {seed}) % 1000000) / 1000000.0")
            inner = f"SELECT *, {u} AS __u FROM ({numbered})"
            assign = f"CASE WHEN __u < {tf} THEN 'test' WHEN __u < {tf + vf} THEN 'val' ELSE 'train' END"
            excl = "__u, __rn"
        ctx.new(f"SELECT * EXCLUDE ({excl}), {assign} AS {q(SPLIT)} FROM ({inner})")
        stats = {"counts": dict(ctx.con.execute(f"SELECT {q(SPLIT)}, COUNT(*) FROM {ctx.rel} GROUP BY 1").fetchall())}
    elif t == "sql":
        sql = (p["sql"] or "").strip().rstrip(";")
        if ";" in sql:
            raise ValueError("one SELECT statement only")
        ctx.new(f"WITH input AS (SELECT * FROM {R}) {sql}")
    return stats


def _type_of(ctx: Context, col: str) -> str:
    return next(t for c, t in ctx.columns() if c == col)


# ----------------------------------------------------------------------------- python steps
def _matrix(ctx: Context, cols: list[str], fit: bool):
    """Numeric matrix of `cols` for fitting (train rows, sampled) or for all rows (in table order)."""
    import numpy as np
    src = ctx.fit_rel() if fit else ctx.rel
    lim = f" USING SAMPLE reservoir({FIT_SAMPLE} ROWS) REPEATABLE (0)" if fit else ""
    df = ctx.con.execute(f"SELECT {', '.join('CAST(' + q(c) + ' AS DOUBLE) AS ' + q(c) for c in cols)} FROM {src}{lim}").fetchdf()
    return df.to_numpy(dtype=np.float64)


def _prep(X, med=None, mu=None, sd=None):
    import numpy as np
    if med is None:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0, med)
    X = np.where(np.isnan(X), med, X)
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd == 0, 1, sd)
    return (X - mu) / sd, med, mu, sd


def _attach(ctx: Context, new_cols: dict, drop: list[str] | None = None):
    """Add columns (numpy arrays aligned with the current row order) to the relation."""
    import pandas as pd
    df = ctx.con.execute(f"SELECT * FROM {ctx.rel}").fetchdf()
    if drop:
        df = df.drop(columns=drop)
    for k, v in new_cols.items():
        df[k] = v
    ctx.n += 1
    name = f"__s{ctx.n}"
    ctx.con.register(name + "_df", df)
    ctx.con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM {name}_df")
    ctx.con.unregister(name + "_df")
    ctx.rel = name


def _python(ctx: Context, t: str, p: dict, step: dict):
    import numpy as np
    cols = ctx.numeric(p.get("columns") or None)
    if t == "feature_select":
        return _feature_select(ctx, p, cols)
    if not cols:
        raise ValueError("no numeric columns (encode categoricals first, e.g. One-hot)")
    Xf, med, mu, sd = _prep(_matrix(ctx, cols, fit=True))
    Xa, *_ = _prep(_matrix(ctx, cols, fit=False), med, mu, sd)
    base = {"columns": cols, "fit_rows": int(len(Xf))}
    if t == "pca":
        from sklearn.decomposition import PCA
        n = int(p["n_components"]) or float(p["variance"])
        pca = PCA(n_components=n if isinstance(n, int) else min(max(n, 0.01), 0.9999), random_state=0).fit(Xf)
        Z = pca.transform(Xa)
        new = {f"{p['prefix']}{i + 1}": Z[:, i] for i in range(Z.shape[1])}
        _attach(ctx, new, None if p["keep_original"] else cols)
        return {**base, "n_components": int(Z.shape[1]), "explained_variance_ratio": pca.explained_variance_ratio_.round(5).tolist()}
    if t == "cluster":
        algo, k, seed = p["algorithm"], int(p["k"]), int(p["seed"])
        if algo == "kmeans":
            from sklearn.cluster import KMeans
            m = KMeans(k, n_init="auto", random_state=seed).fit(Xf)
            lab = m.predict(Xa)
        elif algo == "gmm":
            from sklearn.mixture import GaussianMixture
            m = GaussianMixture(k, random_state=seed).fit(Xf)
            lab = m.predict(Xa)
        else:
            from sklearn.cluster import DBSCAN, HDBSCAN
            from sklearn.neighbors import KNeighborsClassifier
            fit = (HDBSCAN(min_cluster_size=int(p["min_cluster_size"])) if algo == "hdbscan" else DBSCAN(eps=float(p["eps"]))).fit(Xf)
            # density methods can't label new points: assign each row the label of its nearest clustered training point
            core = fit.labels_ >= 0
            if not core.any():
                lab = np.full(len(Xa), -1)
            else:
                lab = KNeighborsClassifier(1).fit(Xf[core], fit.labels_[core]).predict(Xa)
        _attach(ctx, {p["name"]: lab.astype(int)})
        sizes = dict(zip(*np.unique(lab, return_counts=True)))
        return {**base, "algorithm": algo, "sizes": {str(int(a)): int(b) for a, b in sizes.items()}}
    if t == "anomaly":
        if p["algorithm"] == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            m = IsolationForest(contamination=float(p["contamination"]), random_state=int(p["seed"]), n_jobs=-1).fit(Xf)
        else:
            from sklearn.neighbors import LocalOutlierFactor
            m = LocalOutlierFactor(contamination=float(p["contamination"]), novelty=True).fit(Xf)
        score = -m.score_samples(Xa)                     # higher = more anomalous
        _attach(ctx, {p["name"] + "_score": score, p["name"] + "_flag": (m.predict(Xa) == -1).astype(int)})
        return {**base, "algorithm": p["algorithm"], "flagged": int((m.predict(Xa) == -1).sum())}
    raise ValueError(f"unknown python step {t}")


def _feature_select(ctx: Context, p: dict, cols: list[str]):
    import numpy as np
    X = _matrix(ctx, cols, fit=True)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    drop: list[str] = []
    scores = {}
    if p["method"] == "variance":
        v = X.var(0)
        scores = dict(zip(cols, v.round(6).tolist()))
        drop = [c for c, s in zip(cols, v) if s <= float(p["threshold"])]
    elif p["method"] == "correlation":
        corr = np.abs(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(corr, 0)
        for i in range(len(cols)):
            if cols[i] in drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] not in drop and corr[i, j] > float(p["threshold"]):
                    drop.append(cols[j])
        scores = {c: float(np.nanmax(corr[i])) for i, c in enumerate(cols)}
    else:
        target = p["target"] or ctx.target
        if not target:
            raise ValueError("choose the target column (or set the dataset's target)")
        y = ctx.con.execute(f"SELECT {q(target)} FROM {ctx.fit_rel()} USING SAMPLE reservoir({FIT_SAMPLE} ROWS) REPEATABLE (0)").fetchdf()[target]
        Xs = _matrix(ctx, cols, fit=True)
        Xs = np.where(np.isnan(Xs), np.nanmedian(Xs, axis=0), Xs)
        classif = y.dtype == object or y.nunique() <= 20
        if p["method"] == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            s = (mutual_info_classif if classif else mutual_info_regression)(Xs, y.astype(str) if classif else y, random_state=0)
        else:
            from sklearn.feature_selection import f_classif, f_regression
            s = (f_classif if classif else f_regression)(Xs, y.astype(str) if classif else y)[0]
        s = np.nan_to_num(s)
        scores = dict(zip(cols, np.round(s, 6).tolist()))
        keep = set(np.array(cols)[np.argsort(-s)[: int(p["k"])]])
        drop = [c for c in cols if c not in keep]
    if drop:
        ctx.new(f"SELECT * EXCLUDE ({', '.join(q(c) for c in drop)}) FROM {ctx.rel}")
    return {"method": p["method"], "dropped": drop, "scores": scores}


def to_sql_preview(steps: list[dict]) -> str:
    """Human-readable description of the recipe (for the UI's 'View as SQL / JSON')."""
    return json.dumps(steps, indent=2)
