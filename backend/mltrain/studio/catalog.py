"""Studio catalog: dataset records, imports (origins), versioned builds with lineage and staleness.

kinds
  source   rows come from an origin: a dashboard dataset, a local/cloud CSV/Parquet file, a live source snapshot,
           a SQL query against a database/warehouse connector, an uploaded file, or the dashboard's own experiment tables
  derived  a recipe (steps.py) applied to a parent dataset
  query    a saved SQL query over other Studio tables

Every successful build writes a new immutable Parquet version; derived datasets record the parent versions they were
built from, so a dataset is "stale" when an upstream dataset has been rebuilt since.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import db
from . import engine, steps

KEEP_VERSIONS = 5
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="studio-build")
_building: set[str] = set()
_lock = threading.Lock()


# ============================================================================= records
def all_datasets(c=None) -> list[dict]:
    if c is None:
        with db.get_conn() as c:
            return all_datasets(c)
    return db.rows(c, "SELECT * FROM studio_datasets ORDER BY created_at")


def get(ds_id: str, c=None) -> dict:
    if c is None:
        with db.get_conn() as c:
            return get(ds_id, c)
    d = db.one(c, "SELECT * FROM studio_datasets WHERE id=? OR slug=?", (ds_id, ds_id))
    if not d:
        raise KeyError(ds_id)
    return d


def _unique_slug(c, name: str) -> str:
    base = engine.slugify(name)
    slug, i = base, 2
    reserved = {"input", "dashboard_sessions", "dashboard_runs", "dashboard_metrics"}
    while slug in reserved or db.one(c, "SELECT 1 FROM studio_datasets WHERE slug=?", (slug,)):
        slug, i = f"{base}_{i}", i + 1
    return slug


def create(name: str, kind: str, origin: dict | None = None, parent_id: str | None = None, recipe: list | None = None,
           query: str | None = None, target: str | None = None, task: str | None = None, description: str = "",
           build_now: bool = True) -> dict:
    if kind not in ("source", "derived", "query"):
        raise ValueError("kind must be source, derived or query")
    with db.get_conn() as c:
        if parent_id:
            parent = get(parent_id, c)
            parent_id = parent["id"]
            target = target if target is not None else parent.get("target")
            task = task if task is not None else parent.get("task")
        did, now = db.new_id(), db.now()
        c.execute("INSERT INTO studio_datasets(id,name,slug,kind,parent_id,origin,recipe,query,target,task,description,status,created_at,updated_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (did, name, _unique_slug(c, name), kind, parent_id, json.dumps(origin or {}), json.dumps(recipe or []), query,
                   target, task, description, "new", now, now))
    if build_now:
        build_async(did)
    return get(did)


def update(ds_id: str, **fields) -> dict:
    allowed = {"name", "recipe", "query", "target", "task", "description", "tags", "origin"}
    sets = {k: (json.dumps(v) if k in ("recipe", "tags", "origin") else v) for k, v in fields.items() if k in allowed}
    if not sets:
        return get(ds_id)
    with db.get_conn() as c:
        d = get(ds_id, c)
        c.execute(f"UPDATE studio_datasets SET {', '.join(k + '=?' for k in sets)}, updated_at=? WHERE id=?",
                  (*sets.values(), db.now(), d["id"]))
    return get(d["id"])


def children(ds_id: str, c=None) -> list[dict]:
    return [d for d in all_datasets(c) if d.get("parent_id") == ds_id or ds_id in _query_deps(d)]


def _query_deps(d: dict) -> list[str]:
    return list((d.get("build") or {}).get("parents", {}).keys()) if d["kind"] == "query" else []


def delete(ds_id: str, force: bool = False) -> None:
    with db.get_conn() as c:
        d = get(ds_id, c)
        kids = children(d["id"], c)
        if kids and not force:
            raise ValueError(f"{len(kids)} dataset(s) are derived from it: {', '.join(k['name'] for k in kids[:5])}. Delete those first.")
        c.execute("DELETE FROM studio_datasets WHERE id=?", (d["id"],))
    shutil.rmtree(engine.root() / "tables" / d["id"], ignore_errors=True)


def stale(d: dict, by_id: dict) -> bool:
    """True when an upstream dataset has a newer version than the one this build used."""
    parents = (d.get("build") or {}).get("parents") or {}
    return any(by_id.get(pid, {}).get("version", 0) > v for pid, v in parents.items())


def describe(d: dict, by_id: dict | None = None) -> dict:
    by_id = by_id or {x["id"]: x for x in all_datasets()}
    b = d.get("build") or {}
    return {**{k: v for k, v in d.items() if k != "build"}, "rows": b.get("rows"), "n_columns": len(b.get("schema") or []),
            "schema": b.get("schema") or [], "bytes": b.get("bytes"), "warnings": b.get("warnings") or [],
            "fitted": b.get("fitted") or {}, "trace": b.get("trace") or [], "build_seconds": b.get("seconds"),
            "stale": stale(d, by_id), "building": d["id"] in _building or d["status"] == "building",
            "parent_name": by_id.get(d.get("parent_id") or "", {}).get("name")}


# ============================================================================= origins (source datasets)
def load_origin(origin: dict):
    """-> pandas DataFrame (or a DuckDB SQL string reading a local file) for a source dataset."""
    import pandas as pd
    t = origin.get("type")
    if t == "dashboard_dataset":
        return _from_plugin(origin)
    if t == "file":
        path = origin["path"]
        from .. import cloud
        if cloud.provider_of(path):
            fmt = origin.get("format") or ("parquet" if path.lower().endswith((".parquet", ".pq")) else "csv")
            files = cloud.glob(path, (".parquet", ".pq") if fmt == "parquet" else (".csv", ".csv.gz"))
            if not files:
                raise FileNotFoundError(f"No {fmt} files at {path}")
            opts = cloud.storage_options(path)
            read = (lambda f: pd.read_parquet(f, storage_options=opts)) if fmt == "parquet" else (lambda f: pd.read_csv(f, storage_options=opts))
            return pd.concat([read(f) for f in files], ignore_index=True)
        p = str(Path(path).expanduser())
        fmt = origin.get("format") or ("parquet" if p.lower().endswith((".parquet", ".pq")) or Path(p).is_dir() else "csv")
        glob = f"{p.rstrip('/')}/*.{'parquet' if fmt == 'parquet' else 'csv'}" if Path(p).is_dir() else p
        return f"SELECT * FROM {'read_parquet' if fmt == 'parquet' else 'read_csv_auto'}({steps.lit(glob)}{', union_by_name=true' if fmt == 'parquet' else ''})"
    if t == "upload":
        return load_origin({"type": "file", "path": origin["path"], "format": origin.get("format")})
    if t == "live":
        from ..data import live
        src = db.one(db.connect(), "SELECT * FROM data_sources WHERE id=?", (origin["source_id"],))
        if not src:
            raise ValueError("live source not found")
        cfg = live.cfg_of(src)
        conn = live.connector(cfg)
        end = time.time()
        start = end - float(origin["hours"]) * 3600 if origin.get("hours") else None
        df = conn.read_window(cfg.get("table"), cfg.get("query"), cfg["ts_col"], cfg["ts_format"], start, end)
        return df.drop(columns=["__ts"], errors="ignore")
    if t == "sql":
        from ..data.connectors import Connector
        conn = Connector(origin["connector"])
        if conn.is_sql:
            from sqlalchemy import text
            return pd.read_sql_query(text(origin["query"]), conn.engine)
        raise ValueError("SQL origins need a SQLite / SQL connector")
    if t == "dashboard_table":
        return dashboard_table(origin["table"])
    raise ValueError(f"unknown origin type {t!r}")


def _from_plugin(origin: dict):
    """Raw rows of a tabular dashboard dataset (all splits, original columns, readable labels)."""
    import pandas as pd

    from ..plugins.base import coerce_params
    from ..plugins.registry import get_registry
    plugin = get_registry().get("dataset", origin["plugin_id"])
    if plugin.modality != "tabular":
        raise ValueError(f"'{plugin.name}' is {plugin.modality} data; the Studio works with tables")
    params = coerce_params(plugin.params(), origin.get("params") or {})
    b = plugin.load(params, 0)
    if not b.raw:
        raise ValueError("this dataset doesn't expose raw rows")
    names = b.info.get("class_names")
    target = origin.get("target_name") or b.info.get("target_name") or "target"
    parts = []
    for split in ("train", "val", "test"):
        df, y = b.raw[split]
        df = df.copy()
        df[target] = [names[int(v)] for v in y] if names and b.task == "classification" else y
        if origin.get("keep_split"):
            df[steps.SPLIT] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def dashboard_table(name: str):
    """The dashboard's own records as tables: analyze experiments with SQL / EDA."""
    import pandas as pd
    with db.get_conn() as c:
        if name == "dashboard_sessions":
            rows = db.rows(c, "SELECT id, name, created_at, problem_id, model_id, framework, task, modality, mode, device, tags, "
                              "model_params, problem_params FROM sessions")
            for r in rows:
                for k in ("model_params", "problem_params"):
                    for pk, pv in (r.pop(k) or {}).items():
                        if isinstance(pv, (int, float, str, bool)) or pv is None:
                            r[f"{'param' if k == 'model_params' else 'data'}.{pk}"] = pv
                r["tags"] = ",".join(r["tags"] or [])
                r["created_at"] = pd.Timestamp(r["created_at"], unit="s")
            return pd.DataFrame(rows)
        if name == "dashboard_runs":
            rows = db.rows(c, "SELECT r.id AS run_id, r.session_id, s.name AS session_name, s.model_id, s.problem_id, r.seed, r.status, "
                              "r.started_at, r.ended_at, r.summary FROM runs r JOIN sessions s ON s.id = r.session_id")
            for r in rows:
                for k, v in (r.pop("summary") or {}).items():
                    if isinstance(v, (int, float)):
                        r[k] = v
                for k in ("started_at", "ended_at"):
                    r[k] = pd.Timestamp(r[k], unit="s") if r[k] else None
            return pd.DataFrame(rows)
        if name == "dashboard_metrics":
            return pd.read_sql_query("SELECT m.run_id, r.session_id, r.seed, m.step, m.key, m.value FROM metrics m "
                                     "JOIN runs r ON r.id = m.run_id", c)
    raise ValueError(f"unknown dashboard table {name}")


DASHBOARD_TABLES = {"dashboard_sessions": "Every training session with its parameters",
                    "dashboard_runs": "Every run (seed) with its final metrics",
                    "dashboard_metrics": "Every logged metric value (long format)"}


# ============================================================================= build
def build_async(ds_id: str) -> None:
    with _lock:
        if ds_id in _building:
            return
        _building.add(ds_id)
    _pool.submit(_build_guarded, ds_id)


def _build_guarded(ds_id: str) -> None:
    try:
        build(ds_id)
    except Exception:
        pass
    finally:
        with _lock:
            _building.discard(ds_id)


def build(ds_id: str, _depth: int = 0) -> dict:
    """Materialize a dataset (building stale / unbuilt parents first). Returns the updated record."""
    if _depth > 20:
        raise ValueError("lineage is too deep (cycle?)")
    d = get(ds_id)
    by_id = {x["id"]: x for x in all_datasets()}
    with db.get_conn() as c:
        c.execute("UPDATE studio_datasets SET status='building', error=NULL WHERE id=?", (d["id"],))
    t0 = time.time()
    try:
        parents: dict[str, int] = {}
        if d["kind"] == "derived":
            par = by_id.get(d["parent_id"])
            if not par:
                raise ValueError("parent dataset was deleted")
            if par["status"] != "ready" or stale(par, by_id):
                par = build(par["id"], _depth + 1)
                by_id[par["id"]] = par
            for s in d["recipe"] or []:           # datasets referenced by join / union steps
                for ref in [s.get("params", {}).get("other")] + list(s.get("params", {}).get("others") or []):
                    if ref and ref in by_id and (by_id[ref]["status"] != "ready" or stale(by_id[ref], by_id)):
                        by_id[ref] = build(ref, _depth + 1)
        ready = [x for x in by_id.values() if x["status"] == "ready" and x["id"] != d["id"]]
        version = (d.get("version") or 0) + 1
        out = engine.table_path(d["id"], version)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta: dict = {}
        if d["kind"] == "source":
            con = engine.connect(sandbox=False)
            data = load_origin(d["origin"])
            if isinstance(data, str):
                con.execute(f"COPY ({data}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            else:
                con.register("__src", data)
                con.execute(f"COPY (SELECT * FROM __src) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        elif d["kind"] == "derived":
            par = by_id[d["parent_id"]]
            parents[par["id"]] = par["version"]
            con = engine.connect(ready)
            ctx = steps.Context(con, engine.table_ref(par), d.get("target"), {x["id"]: x for x in ready})
            steps.run(ctx, d["recipe"] or [])
            con.execute(f"COPY (SELECT * FROM {ctx.rel}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            for s in d["recipe"] or []:
                for ref in [s.get("params", {}).get("other")] + list(s.get("params", {}).get("others") or []):
                    if ref in by_id:
                        parents[ref] = by_id[ref]["version"]
            meta = {"fitted": ctx.fitted, "warnings": ctx.warnings, "trace": ctx.trace}
        else:
            con = engine.connect(ready)
            sql = (d.get("query") or "").strip().rstrip(";")
            if ";" in sql:
                raise ValueError("one SELECT statement only")
            con.execute(f"COPY ({sql}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            low = sql.lower()
            parents = {x["id"]: x["version"] for x in ready if x["slug"] in low}
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
        schema = [{"name": r[0], "type": r[1]} for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()]
        con.close()
        build_meta = {**meta, "version": version, "rows": rows, "schema": schema, "bytes": out.stat().st_size, "parents": parents,
                      "seconds": round(time.time() - t0, 2), "built_at": time.time()}
        with db.get_conn() as c:
            c.execute("UPDATE studio_datasets SET status='ready', version=?, build=?, error=NULL, built_at=? WHERE id=?",
                      (version, json.dumps(build_meta, default=str), db.now(), d["id"]))
        _prune(d["id"], version)
    except Exception as e:
        msg = str(e) if isinstance(e, (ValueError, FileNotFoundError, KeyError)) else f"{type(e).__name__}: {e}"
        with db.get_conn() as c:
            c.execute("UPDATE studio_datasets SET status=?, error=? WHERE id=?",
                      ("ready" if d.get("version") else "error", msg[-2000:], d["id"]))
        traceback.print_exc()
        raise
    return get(d["id"])


def _prune(ds_id: str, current: int) -> None:
    folder = engine.root() / "tables" / ds_id
    for f in folder.glob("v*.parquet"):
        try:
            v = int(f.stem[1:])
        except ValueError:
            continue
        if v <= current - KEEP_VERSIONS:
            f.unlink(missing_ok=True)


# ============================================================================= preview (recipe editing)
def preview(parent_id: str, recipe: list[dict], target: str | None, rows: int = 50_000, upto: int | None = None) -> dict:
    """Run a recipe on a sample of the parent (fast feedback while editing). Fitted statistics are approximate."""
    by_id = {x["id"]: x for x in all_datasets()}
    par = get(parent_id)
    if par["status"] != "ready":
        raise ValueError(f"'{par['name']}' isn't built yet")
    ready = [x for x in by_id.values() if x["status"] == "ready"]
    con = engine.connect(ready)
    total = (par.get("build") or {}).get("rows") or 0
    src = engine.table_ref(par)
    if total > rows:
        con.execute(f"CREATE TEMP TABLE __sample AS SELECT * FROM {src} USING SAMPLE reservoir({rows} ROWS) REPEATABLE (0)")
        src = "__sample"
    ctx = steps.Context(con, src, target, {x["id"]: x for x in ready})
    steps.run(ctx, recipe[: upto] if upto is not None else recipe)
    head = con.execute(f"SELECT * FROM {ctx.rel} LIMIT 200").fetchdf()
    schema = [{"name": c, "type": t} for c, t in ctx.columns()]
    n = con.execute(f"SELECT COUNT(*) FROM {ctx.rel}").fetchone()[0]
    con.close()
    return {"sampled": total > rows, "sample_rows": min(total, rows), "rows": n, "schema": schema, "warnings": ctx.warnings,
            "trace": ctx.trace, "fitted": ctx.fitted, "head": json.loads(head.to_json(orient="split", date_format="iso", default_handler=str))}


VALUE_STEPS = {"scale": "scaled", "math": "math transform", "clip": "clipped", "bin": "binned"}


def value_transforms(d: dict, column: str, by_id: dict | None = None, _seen=None) -> list[str]:
    """Recipe steps (in this dataset or any ancestor) that changed the *values* of a column — e.g. standardization.
    Comparing a scaled column with a raw one is meaningless, so the comparison view warns about mismatches."""
    by_id = by_id or {x["id"]: x for x in all_datasets()}
    seen = _seen or set()
    if d["id"] in seen:
        return []
    seen.add(d["id"])
    parents = [d["parent_id"]] if d.get("parent_id") else []
    parents += list(((d.get("build") or {}).get("parents") or {}).keys()) if d["kind"] == "query" else []
    out: list[str] = []
    for pid in dict.fromkeys(parents):
        if pid in by_id:
            out += value_transforms(by_id[pid], column, by_id, seen)
    for st in d.get("recipe") or []:
        if not st.get("enabled", True) or st.get("type") not in VALUE_STEPS:
            continue
        p = st.get("params") or {}
        if st["type"] == "bin":
            hit = p.get("column") == column and not p.get("new_name")
        else:
            hit = column in (p.get("columns") or []) or not p.get("columns")
        if hit:
            out.append(VALUE_STEPS[st["type"]] + (f" ({p['method']})" if p.get("method") else f" ({p['fn']})" if p.get("fn") else ""))
    return out
