"""Data Studio API: datasets (import / derive / query), recipes, EDA, charts, subset comparison, unsupervised
exploration, sandboxed SQL and lineage."""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import db
from ..studio import catalog, charts, eda, engine, explore, steps
from .state import protected_router

router = protected_router()


def _ds(ds_id: str) -> dict:
    try:
        return catalog.get(ds_id)
    except KeyError:
        raise HTTPException(404, "dataset not found")


def _ready(ds_id: str) -> dict:
    d = _ds(ds_id)
    if not d.get("version"):
        raise HTTPException(409, f"'{d['name']}' isn't built yet" + (f": {d['error']}" if d.get("error") else ""))
    return d


async def _run(fn, *a, **kw):
    try:
        return await run_in_threadpool(fn, *a, **kw)
    except HTTPException:
        raise
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        msg = str(e)
        if "Interrupt" in type(e).__name__ or "INTERRUPT" in msg.upper():
            raise HTTPException(408, f"Query cancelled after {engine.QUERY_TIMEOUT_S}s")
        from .. import cloud
        raise HTTPException(400, cloud.redact(f"{type(e).__name__}: {msg}"))


# ===================================================================== catalog
@router.get("/studio/datasets")
def list_datasets():
    all_ = catalog.all_datasets()
    by_id = {d["id"]: d for d in all_}
    return [catalog.describe(d, by_id) for d in all_]


class CreateBody(BaseModel):
    name: str
    kind: str = "source"                 # source | derived | query
    origin: dict[str, Any] | None = None
    parent_id: str | None = None
    recipe: list[dict[str, Any]] = []
    query: str | None = None
    target: str | None = None
    task: str | None = None
    description: str = ""


@router.post("/studio/datasets")
async def create_dataset(body: CreateBody):
    if body.kind == "derived" and not body.parent_id:
        raise HTTPException(400, "derived datasets need a parent")
    if body.kind == "query" and not (body.query or "").strip():
        raise HTTPException(400, "query datasets need SQL")
    d = await _run(catalog.create, body.name.strip() or "dataset", body.kind, body.origin, body.parent_id, body.recipe, body.query,
                   body.target or None, body.task or None, body.description)
    return catalog.describe(d)


@router.get("/studio/datasets/{ds_id}")
def get_dataset(ds_id: str):
    return catalog.describe(_ds(ds_id))


class PatchBody(BaseModel):
    name: str | None = None
    recipe: list[dict[str, Any]] | None = None
    query: str | None = None
    target: str | None = None
    task: str | None = None
    description: str | None = None
    rebuild: bool = False


@router.patch("/studio/datasets/{ds_id}")
async def patch_dataset(ds_id: str, body: PatchBody):
    d = _ds(ds_id)
    fields = body.model_dump(exclude_none=True, exclude={"rebuild"})
    if "target" in fields and fields["target"] == "":
        fields["target"], fields["task"] = None, None
    d = catalog.update(d["id"], **fields)
    if body.rebuild or "recipe" in fields or "query" in fields:
        catalog.build_async(d["id"])
    from ..plugins.registry import get_registry
    get_registry(reload=True)              # target / task changes affect the training dataset list
    return catalog.describe(catalog.get(d["id"]))


@router.post("/studio/datasets/{ds_id}/build")
def build_dataset(ds_id: str):
    d = _ds(ds_id)
    catalog.build_async(d["id"])
    return {"ok": True}


@router.delete("/studio/datasets/{ds_id}")
async def delete_dataset(ds_id: str):
    d = _ds(ds_id)
    with db.get_conn() as c:
        used = c.execute("SELECT COUNT(*) FROM sessions WHERE problem_id=?", (f"studio.{d['slug']}",)).fetchone()[0]
    await _run(catalog.delete, d["id"])
    from ..plugins.registry import get_registry
    get_registry(reload=True)
    return {"ok": True, "sessions_using": used}


class PreviewBody(BaseModel):
    parent_id: str
    recipe: list[dict[str, Any]]
    target: str | None = None
    upto: int | None = None
    rows: int = 50_000


@router.post("/studio/preview")
async def preview(body: PreviewBody):
    return await _run(catalog.preview, body.parent_id, body.recipe, body.target, min(body.rows, 500_000), body.upto)


@router.get("/studio/steps")
def step_catalog():
    return steps.catalog()


# ===================================================================== imports
@router.get("/studio/origins")
def origins():
    """What can be imported: tabular dashboard datasets, live sources, the dashboard's own tables."""
    from ..plugins.registry import get_registry
    reg = get_registry()
    ds = [{"id": p.id, "name": p.name, "description": p.description, "task": p.task, "source": p.source,
           "params": [x.to_dict() for x in p.params()]}
          for p in reg.plugins["dataset"].values() if p.modality == "tabular" and not p.id.startswith("studio.")]
    with db.get_conn() as c:
        live = db.rows(c, "SELECT id, name FROM data_sources ORDER BY name")
    return {"datasets": sorted(ds, key=lambda x: x["name"]), "live_sources": live,
            "dashboard_tables": [{"name": k, "description": v} for k, v in catalog.DASHBOARD_TABLES.items()]}


@router.post("/studio/upload")
async def upload(file: UploadFile = File(...), name: str = Form("")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".csv", ".parquet", ".pq", ".tsv", ".json", ".jsonl"):
        raise HTTPException(400, "Upload a CSV, TSV, Parquet or JSON-lines file")
    d = engine.root() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{int(time.time())}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', file.filename or 'upload' + ext)}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    fmt = "parquet" if ext in (".parquet", ".pq") else "csv"
    if ext in (".json", ".jsonl"):
        import pandas as pd
        df = pd.read_json(dest, lines=ext == ".jsonl")
        dest2 = dest.with_suffix(".parquet")
        df.to_parquet(dest2)
        dest, fmt = dest2, "parquet"
    ds = await _run(catalog.create, name or Path(file.filename or "upload").stem, "source", {"type": "upload", "path": str(dest), "format": fmt})
    return catalog.describe(ds)


# ===================================================================== exploring a dataset
@router.get("/studio/datasets/{ds_id}/rows")
async def dataset_rows(ds_id: str, offset: int = 0, limit: int = 100, sort: str | None = None, desc: bool = False, where: str = ""):
    return await _run(eda.rows, _ready(ds_id), offset, min(limit, 1000), sort, desc, where)


@router.get("/studio/datasets/{ds_id}/profile")
async def dataset_profile(ds_id: str, refresh: bool = False):
    return await _run(eda.profile, _ready(ds_id), refresh)


@router.get("/studio/datasets/{ds_id}/column")
async def dataset_column(ds_id: str, name: str, by: str | None = None, bins: int = 40):
    return await _run(eda.column_detail, _ready(ds_id), name, by, min(max(bins, 5), 200))


class ChartBody(BaseModel):
    datasets: list[str]
    spec: dict[str, Any]


@router.post("/studio/chart")
async def chart(body: ChartBody):
    if not body.datasets:
        raise HTTPException(400, "choose a dataset")
    return await _run(charts.chart, [_ready(i) for i in body.datasets[:12]], body.spec)


class CompareBody(BaseModel):
    datasets: list[str]
    column: str
    filter: str | None = None


@router.post("/studio/compare")
async def compare(body: CompareBody):
    if len(body.datasets) < 2:
        raise HTTPException(400, "pick at least two datasets to compare")
    return await _run(charts.compare, [_ready(i) for i in body.datasets[:12]], body.column, body.filter)


class ExploreBody(BaseModel):
    dataset: str
    spec: dict[str, Any]


@router.post("/studio/explore")
async def explore_run(body: ExploreBody):
    return await _run(explore.run, _ready(body.dataset), body.spec)


class ApplyBody(BaseModel):
    dataset: str
    spec: dict[str, Any]
    name: str
    subset_value: int | None = None        # cluster id / anomaly flag to keep, for "save as subset"


@router.post("/studio/explore/apply")
async def explore_apply(body: ApplyBody):
    """Save an exploration as a derived dataset: the fitted step (cluster / anomaly / pca), optionally filtered to one group."""
    d = _ready(body.dataset)
    step = {**explore.recipe_step(body.spec), "id": db.new_id(), "enabled": True}
    recipe = [step]
    if body.subset_value is not None:
        col = step["params"]["name"] + ("_flag" if step["type"] == "anomaly" else "")
        recipe.append({"id": db.new_id(), "type": "filter", "enabled": True,
                       "params": {"conditions": [{"column": col, "op": "=", "value": str(body.subset_value)}]}})
    new = await _run(catalog.create, body.name, "derived", None, d["id"], recipe)
    return catalog.describe(new)


# ===================================================================== SQL
class SQLBody(BaseModel):
    sql: str
    limit: int = 1000


def _sql_connection(sql: str):
    datasets = catalog.all_datasets()
    low = sql.lower()
    extra = {name: catalog.dashboard_table(name) for name in catalog.DASHBOARD_TABLES if name in low}
    return engine.connect(datasets, extra)


def _run_sql(sql: str, limit: int) -> dict:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("Write a query")
    if ";" in sql:
        raise ValueError("Run one statement at a time")
    con = _sql_connection(sql)
    t0 = time.time()
    try:
        res = engine.run_with_timeout(con, sql)
        if res.description is None:
            return {"columns": [], "rows": [], "row_count": 0, "seconds": round(time.time() - t0, 3), "truncated": False}
        cols = [d[0] for d in res.description]
        types = [str(d[1]) for d in res.description]
        rows = res.fetchmany(limit + 1)
        truncated = len(rows) > limit
        rows = rows[:limit]
        return {"columns": cols, "types": types, "rows": [[_cell(v) for v in r] for r in rows], "row_count": len(rows),
                "truncated": truncated, "seconds": round(time.time() - t0, 3)}
    finally:
        con.close()


def _cell(v):
    import math
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


@router.post("/studio/sql")
async def run_sql(body: SQLBody):
    return await _run(_run_sql, body.sql, min(max(body.limit, 1), engine.MAX_RESULT_ROWS))


@router.get("/studio/schema")
async def schema():
    """Tables and columns for the SQL editor's schema browser."""
    def build():
        out = []
        for d in catalog.all_datasets():
            if not d.get("version"):
                continue
            b = d.get("build") or {}
            out.append({"name": d["slug"], "label": d["name"], "kind": d["kind"], "rows": b.get("rows"), "id": d["id"],
                        "columns": b.get("schema") or []})
        for name, desc in catalog.DASHBOARD_TABLES.items():
            out.append({"name": name, "label": desc, "kind": "dashboard", "rows": None, "columns": []})
        return out
    return await _run(build)


class QueryBody(BaseModel):
    name: str
    sql: str


@router.get("/studio/queries")
def list_queries():
    with db.get_conn() as c:
        return db.rows(c, "SELECT * FROM studio_queries ORDER BY updated_at DESC")


@router.post("/studio/queries")
def save_query(body: QueryBody):
    with db.get_conn() as c:
        ex = db.one(c, "SELECT id FROM studio_queries WHERE name=?", (body.name,))
        if ex:
            c.execute("UPDATE studio_queries SET sql=?, updated_at=? WHERE id=?", (body.sql, db.now(), ex["id"]))
            return {"id": ex["id"]}
        qid = db.new_id()
        c.execute("INSERT INTO studio_queries(id,name,sql,created_at,updated_at) VALUES (?,?,?,?,?)", (qid, body.name, body.sql, db.now(), db.now()))
        return {"id": qid}


@router.delete("/studio/queries/{qid}")
def delete_query(qid: str):
    with db.get_conn() as c:
        c.execute("DELETE FROM studio_queries WHERE id=?", (qid,))
    return {"ok": True}


# ===================================================================== export / lineage
@router.get("/studio/datasets/{ds_id}/download")
async def download(ds_id: str, format: str = "parquet"):
    d = _ready(ds_id)
    src = engine.table_path(d["id"], d["version"])
    if format == "parquet":
        return FileResponse(src, filename=f"{d['slug']}_v{d['version']}.parquet")

    def to_csv():
        out = engine.root() / "tmp" / f"{d['slug']}_v{d['version']}.csv"
        if not out.exists():
            con = engine.connect(sandbox=False)
            con.execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{out}' (HEADER, DELIMITER ',')")
            con.close()
        return out
    out = await _run(to_csv)
    return FileResponse(out, filename=out.name, media_type="text/csv")


@router.get("/studio/datasets/{ds_id}/lineage")
def lineage(ds_id: str):
    d = _ds(ds_id)
    all_ = catalog.all_datasets()
    by_id = {x["id"]: x for x in all_}
    up, cur = [], d
    while cur.get("parent_id") and cur["parent_id"] in by_id:
        cur = by_id[cur["parent_id"]]
        up.append({"id": cur["id"], "name": cur["name"], "kind": cur["kind"], "version": cur["version"]})
    down = []

    def walk(pid, depth):
        for x in all_:
            if x.get("parent_id") == pid or (x["kind"] == "query" and pid in ((x.get("build") or {}).get("parents") or {})):
                down.append({"id": x["id"], "name": x["name"], "kind": x["kind"], "depth": depth, "parent": pid})
                if depth < 10:
                    walk(x["id"], depth + 1)
    walk(d["id"], 1)
    with db.get_conn() as c:
        sessions = db.rows(c, "SELECT id, name, model_id, task, created_at FROM sessions WHERE problem_id=? ORDER BY created_at DESC",
                           (f"studio.{d['slug']}",))
    return {"ancestors": list(reversed(up)), "descendants": down, "sessions": sessions,
            "joins": [s["params"].get("other") for s in (d.get("recipe") or []) if s.get("type") == "join"]}
