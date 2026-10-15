"""Data API: static dataset quality reports, live data sources (health), training schedules, demo stream."""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import db
from ..data import live, schedules, service, static_report
from ..plugins.registry import get_registry
from .state import protected_router

router = protected_router()


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, KeyError) as e:
        from .. import cloud
        raise HTTPException(400, cloud.redact(str(e)))
    except Exception as e:                  # driver errors can echo connection URLs: never return passwords
        from .. import cloud
        raise HTTPException(400, cloud.redact(f"{type(e).__name__}: {e}"))


# ===================================================================== static datasets
@router.get("/data/static")
def static_datasets():
    out = []
    for p in get_registry().list("dataset"):
        if p.id.startswith(("live.", "deployment.")):
            continue
        out.append({**{k: v for k, v in p.meta().items() if k != "params"}, "report": static_report.cached_summary(p.id)})
    return out


class ReportBody(BaseModel):
    params: dict[str, Any] = {}
    refresh: bool = False


@router.post("/data/static/{dataset_id}/report")
async def static_dataset_report(dataset_id: str, body: ReportBody):
    """Data quality report (cached per parameter set). First run may download the dataset."""
    return await run_in_threadpool(_guard, static_report.report, dataset_id, body.params, body.refresh)


# ===================================================================== live sources
class ConnBody(BaseModel):
    connector: dict[str, Any]
    table: str | None = None
    query: str | None = None


@router.post("/data/connect/tables")
async def connect_tables(body: ConnBody):
    from ..data.connectors import Connector
    return {"tables": await run_in_threadpool(_guard, lambda: Connector(body.connector).tables())}


@router.post("/data/connect/inspect")
async def connect_inspect(body: ConnBody):
    return await run_in_threadpool(_guard, live.inspect, body.connector, body.table, body.query)


class SourceBody(BaseModel):
    name: str
    config: dict[str, Any]


@router.get("/data/sources")
def list_sources():
    with db.get_conn() as c:
        rows = db.rows(c, "SELECT * FROM data_sources ORDER BY created_at DESC")
        out = []
        for s in rows:
            snap = db.one(c, "SELECT metrics FROM source_snapshots WHERE source_id=? ORDER BY ts DESC LIMIT 1", (s["id"],))
            m = snap["metrics"] if snap else {}
            out.append({**s, "latest": {k: m.get(k) for k in ("rows", "last_ts", "lag_s", "volume", "label_coverage", "recent_rows")}})
    return out


@router.post("/data/sources")
async def create_source(body: SourceBody):
    sid = await run_in_threadpool(_guard, live.create_source, body.name, body.config)
    get_registry(reload=True)          # its training dataset `live.<id>` appears in New training
    with db.get_conn() as c:
        src = db.one(c, "SELECT * FROM data_sources WHERE id=?", (sid,))
    await run_in_threadpool(_guard, live.profile_source, src)
    return {"id": sid}


def _source(sid: str) -> dict:
    with db.get_conn() as c:
        s = db.one(c, "SELECT * FROM data_sources WHERE id=?", (sid,))
    if not s:
        raise HTTPException(404, "source not found")
    return s


@router.get("/data/sources/{sid}")
def get_source(sid: str):
    s = _source(sid)
    with db.get_conn() as c:
        snap = db.one(c, "SELECT * FROM source_snapshots WHERE source_id=? ORDER BY ts DESC LIMIT 1", (sid,))
        scheds = [x for x in db.rows(c, "SELECT * FROM training_schedules ORDER BY created_at DESC")
                  if (x["template"] or {}).get("problem_id") == f"live.{sid}"]
        sessions = db.rows(c, "SELECT id, name, created_at FROM sessions WHERE problem_id=? ORDER BY created_at DESC LIMIT 20", (f"live.{sid}",))
    for x in scheds:
        x["cadence_text"] = schedules.describe(x["cadence"])
    return {**s, "config": live.cfg_of(s), "latest": snap["metrics"] if snap else None, "schedules": scheds, "sessions": sessions,
            "dataset_id": f"live.{sid}", "is_demo": (s["config"].get("connector") or {}).get("path") == str(service.DEMO_DB)}


class SourcePatch(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    reset_schema_baseline: bool = False


@router.patch("/data/sources/{sid}")
async def patch_source(sid: str, body: SourcePatch):
    s = _source(sid)
    cfg = dict(s["config"])
    if body.config:
        protected = {"connector", "ts_format"}
        cfg.update({k: v for k, v in body.config.items() if k not in protected})
    if body.reset_schema_baseline:
        conn = live.connector(live.cfg_of({**s, "config": cfg}))
        sample = await run_in_threadpool(_guard, conn.sample, cfg.get("table"), cfg.get("query"), 500)
        cfg["schema_baseline"] = live.schema_of(sample)
    with db.get_conn() as c:
        c.execute("UPDATE data_sources SET name=COALESCE(?, name), config=?, last_profiled_at=NULL WHERE id=?",
                  (body.name, json.dumps(cfg), sid))
    get_registry(reload=True)
    return get_source(sid)


@router.delete("/data/sources/{sid}")
def delete_source(sid: str):
    _source(sid)
    with db.get_conn() as c:
        c.execute("DELETE FROM data_sources WHERE id=?", (sid,))
    get_registry(reload=True)
    return {"ok": True}


@router.post("/data/sources/{sid}/profile")
async def profile_now(sid: str):
    return await run_in_threadpool(_guard, live.profile_source, _source(sid))


@router.get("/data/sources/{sid}/snapshots")
def source_snapshots(sid: str, hours: float = 72):
    with db.get_conn() as c:
        snaps = db.rows(c, "SELECT ts, health, metrics FROM source_snapshots WHERE source_id=? AND ts>=? ORDER BY ts",
                        (sid, time.time() - hours * 3600))
    out = []
    for s in snaps:
        m = s["metrics"] or {}
        out.append({"ts": s["ts"], "health": s["health"], "lag_s": m.get("lag_s"), "rows": m.get("rows"),
                    "last_hour": (m.get("volume") or {}).get("last_hour"), "typical_hour": (m.get("volume") or {}).get("typical_hour"),
                    "label_coverage": m.get("label_coverage"),
                    "missing": {k: v.get("missing") for k, v in (m.get("columns") or {}).items()},
                    "psi": {k: v.get("psi") for k, v in (m.get("columns") or {}).items()}})
    return out


@router.get("/data/sources/{sid}/volume")
async def source_volume(sid: str, hours: float = 168):
    return await run_in_threadpool(_guard, live.hourly_volume, _source(sid), hours)


@router.get("/data/sources/{sid}/preview")
async def source_preview(sid: str, n: int = 50):
    return await run_in_threadpool(_guard, live.preview, _source(sid), min(n, 500))


@router.get("/data/sources/{sid}/columns/{col}")
async def source_column(sid: str, col: str, hours: float = 24):
    return await run_in_threadpool(_guard, live.column_detail, _source(sid), col, hours)


# ===================================================================== schedules
class ScheduleBody(BaseModel):
    name: str
    template: dict[str, Any]
    cadence: dict[str, Any]
    register_as: str | None = None
    start_now: bool = False


@router.get("/schedules")
def list_schedules():
    with db.get_conn() as c:
        rows = db.rows(c, "SELECT * FROM training_schedules ORDER BY created_at DESC")
        for r in rows:
            r["cadence_text"] = schedules.describe(r["cadence"])
            r["runs"] = db.rows(c, "SELECT r.session_id, r.created_at, r.status, r.registered_version, s.name FROM schedule_runs r "
                                   "LEFT JOIN sessions s ON s.id=r.session_id WHERE r.schedule_id=? ORDER BY r.created_at DESC LIMIT 10", (r["id"],))
    return rows


@router.post("/schedules")
def create_schedule(body: ScheduleBody):
    return {"id": _guard(schedules.create, body.name, body.template, body.cadence, body.register_as, body.start_now)}


class SchedulePatch(BaseModel):
    enabled: bool | None = None
    cadence: dict[str, Any] | None = None
    register_as: str | None = None
    name: str | None = None


@router.patch("/schedules/{sid}")
def patch_schedule(sid: str, body: SchedulePatch):
    with db.get_conn() as c:
        s = db.one(c, "SELECT * FROM training_schedules WHERE id=?", (sid,))
        if not s:
            raise HTTPException(404, "schedule not found")
        if body.enabled is not None:
            c.execute("UPDATE training_schedules SET enabled=? WHERE id=?", (int(body.enabled), sid))
        if body.cadence is not None:
            c.execute("UPDATE training_schedules SET cadence=?, next_run_at=? WHERE id=?",
                      (json.dumps(body.cadence), schedules.next_run(body.cadence, time.time()), sid))
        if body.register_as is not None:
            c.execute("UPDATE training_schedules SET register_as=? WHERE id=?", (body.register_as or None, sid))
        if body.name:
            c.execute("UPDATE training_schedules SET name=? WHERE id=?", (body.name, sid))
    return {"ok": True}


@router.post("/schedules/{sid}/run")
def run_schedule(sid: str):
    with db.get_conn() as c:
        s = db.one(c, "SELECT * FROM training_schedules WHERE id=?", (sid,))
    if not s:
        raise HTTPException(404, "schedule not found")
    return {"session_id": _guard(schedules.launch, s)}


@router.delete("/schedules/{sid}")
def delete_schedule(sid: str):
    with db.get_conn() as c:
        c.execute("DELETE FROM training_schedules WHERE id=?", (sid,))
    return {"ok": True}


# ===================================================================== demo live database
class StreamBody(BaseModel):
    rate: float = 120 / 3600     # rows per second — matches the demo history (120 rows/hour)
    incident: str = "none"
    label_delay: float = 60


@router.post("/data/demo/setup")
async def demo_setup():
    """Create a growing SQLite database of loan applications and connect it as a live source."""
    with db.get_conn() as c:
        if db.one(c, "SELECT id FROM data_sources WHERE id='loan-applications'"):
            raise HTTPException(409, "The demo source already exists")
    service.start()
    await run_in_threadpool(_guard, service.wait_for_backfill)
    cfg = {"connector": {"kind": "sqlite", "path": str(service.DEMO_DB)}, "table": "applications", "ts_col": "created_at",
           "target": "outcome", "task": "classification", "id_col": "id", "profile_every_minutes": 2, "recent_hours": 6,
           "reference_days": 7, "description": "Demo: synthetic loan applications streaming in; outcomes arrive ~1 min later"}
    sid = await run_in_threadpool(_guard, live.create_source, "Loan applications", cfg)
    get_registry(reload=True)
    with db.get_conn() as c:
        src = db.one(c, "SELECT * FROM data_sources WHERE id=?", (sid,))
    await run_in_threadpool(_guard, live.profile_source, src)
    return {"id": sid}


@router.get("/data/demo/stream")
def demo_stream_state():
    return service.state() or {}


@router.post("/data/demo/stream")
def demo_stream_start(body: StreamBody):
    if body.incident not in ("none", "missing_values", "new_category"):
        raise HTTPException(400, "incident must be none, missing_values or new_category")
    if not service.DEMO_DB.exists():
        raise HTTPException(400, "Set up the demo source first")
    return service.start(body.rate, body.incident, body.label_delay)


@router.delete("/data/demo/stream")
def demo_stream_stop():
    return {"stopped": service.stop()}


# ===================================================================== cloud (AWS / GCP / Azure)
@router.get("/cloud/status")
async def cloud_status():
    from .. import cloud
    return await run_in_threadpool(cloud.status)


class CloudSettings(BaseModel):
    aws: dict[str, Any] = {}
    gcp: dict[str, Any] = {}
    azure: dict[str, Any] = {}


@router.put("/cloud/settings")
def put_cloud_settings(body: CloudSettings):
    from .. import cloud
    return cloud.save_settings(body.model_dump())


class CloudTest(BaseModel):
    path: str
    auth: dict[str, Any] | None = None


@router.post("/cloud/test")
async def cloud_test(body: CloudTest):
    from .. import cloud
    if not cloud.provider_of(body.path):
        raise HTTPException(400, "Enter a cloud URL: s3://…, gs://… or az://…")
    try:
        return await run_in_threadpool(cloud.test, body.path, body.auth)
    except Exception as e:
        raise HTTPException(400, cloud.redact(f"{type(e).__name__}: {e}"))
