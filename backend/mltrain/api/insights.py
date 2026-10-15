"""Explainability (SHAP / Grad-CAM), hyperparameter searches (Optuna), notifications."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import config, db, events, explain, search
from ..data import live, schedules
from ..notify import CHANNEL_TYPES, EVENTS, get_notifier
from .state import CFG, protected_router

router = protected_router()
log = logging.getLogger("mltrain")


# ===================================================================== background loop
class Background:
    """Drives hyperparameter searches and completion notifications."""

    def __init__(self):
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        events.mark_existing_notified()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            for fn in (search.tick, events.tick, schedules.tick, live.tick):
                try:
                    await asyncio.to_thread(fn)
                except Exception:
                    log.exception("background %s failed", fn.__module__)
            await asyncio.sleep(4)


BACKGROUND = Background()


# ===================================================================== explainability
def _path(kind: str, ref: str) -> str:
    if kind in ("run", "run_final"):
        try:
            return str(explain.model_path_for_run(ref, "final" if kind == "run_final" else "best"))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
    if kind == "version":
        with db.get_conn() as c:
            v = db.one(c, "SELECT path FROM model_versions WHERE id=?", (ref,))
        if not v:
            raise HTTPException(404, "version not found")
        return v["path"]
    raise HTTPException(400, "kind must be run, run_final or version")


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@router.post("/explain/{kind}/{ref}/shap")
async def shap_global(kind: str, ref: str, n: int = 150):
    """Global SHAP: feature importance, beeswarm data and per-example explanations on held-out data (cached)."""
    return await run_in_threadpool(_guard, explain.tabular_global, _path(kind, ref), min(max(n, 20), 500))


class RecordBody(BaseModel):
    record: dict[str, Any]


@router.post("/explain/{kind}/{ref}/record")
async def shap_record(kind: str, ref: str, body: RecordBody):
    """SHAP waterfall for one raw input record."""
    return await run_in_threadpool(_guard, explain.tabular_record, _path(kind, ref), body.record)


class ImagesBody(BaseModel):
    images: list[str] | None = None   # base64 images; omit to use test-set samples


@router.post("/explain/{kind}/{ref}/gradcam")
async def gradcam(kind: str, ref: str, body: ImagesBody):
    if body.images and len(body.images) > 24:
        raise HTTPException(400, "At most 24 images")
    return await run_in_threadpool(_guard, explain.gradcam, _path(kind, ref), body.images)


@router.get("/deployments/{did}/predictions/{pid}/explain")
async def explain_prediction(did: str, pid: int):
    """Why did the deployed model make this prediction? (SHAP on the logged raw input)"""
    with db.get_conn() as c:
        row = db.one(c, "SELECT p.payload, v.path, v.modality FROM predictions p JOIN model_versions v ON v.id=p.version_id "
                        "WHERE p.id=? AND p.deployment_id=?", (pid, did))
    if not row:
        raise HTTPException(404, "prediction not found")
    if row["modality"] != "tabular" or not isinstance(row["payload"], dict):
        raise HTTPException(400, "Per-prediction explanations are available for tabular models")
    return await run_in_threadpool(_guard, explain.tabular_record, row["path"], row["payload"])


# ===================================================================== hyperparameter search
class SearchCreate(BaseModel):
    name: str | None = None
    base: dict[str, Any]
    space: dict[str, dict[str, Any]]
    n_trials: int = 20
    parallel: int = 2
    metric: str = "auto"
    direction: str = "auto"
    pruning: bool = True
    sampler: str = "tpe"


@router.post("/searches")
def create_search(body: SearchCreate):
    sid = search.create(body.model_dump())
    return {"id": sid}


@router.get("/searches")
def list_searches():
    with db.get_conn() as c:
        rows = db.rows(c, "SELECT s.*, (SELECT COUNT(*) FROM search_trials t WHERE t.search_id=s.id) AS n_asked, "
                          "(SELECT COUNT(*) FROM search_trials t WHERE t.search_id=s.id AND t.state='complete') AS n_complete, "
                          "(SELECT COUNT(*) FROM search_trials t WHERE t.search_id=s.id AND t.state='pruned') AS n_pruned "
                          "FROM searches s ORDER BY created_at DESC")
    return rows


@router.get("/searches/{sid}")
def get_search(sid: str):
    return search.detail(sid)


@router.post("/searches/{sid}/stop")
def stop_search(sid: str):
    search.stop(sid)
    return {"ok": True}


# ===================================================================== notifications
class Channel(BaseModel):
    id: str | None = None
    type: str
    name: str = ""
    enabled: bool = True
    events: list[str] = []
    url: str | None = None          # discord / slack / webhook
    server: str | None = None       # ntfy
    topic: str | None = None        # ntfy
    token: str | None = None        # ntfy (optional)
    host: str | None = None         # email
    port: int | None = None
    username: str | None = None
    password: str | None = None
    to: str | None = None
    from_: str | None = None


def _validate(ch: Channel) -> dict:
    if ch.type not in CHANNEL_TYPES:
        raise HTTPException(400, f"type must be one of {CHANNEL_TYPES}")
    need = {"discord": ["url"], "slack": ["url"], "webhook": ["url"], "ntfy": ["topic"], "email": ["host", "username", "password", "to"]}[ch.type]
    missing = [f for f in need if not getattr(ch, f)]
    if missing:
        raise HTTPException(400, f"{ch.type} channel needs: {', '.join(missing)}")
    bad = [e for e in ch.events if e not in EVENTS]
    if bad:
        raise HTTPException(400, f"unknown events {bad}")
    d = ch.model_dump(exclude_none=True)
    if "from_" in d:
        d["from"] = d.pop("from_")
    d["id"] = ch.id or uuid.uuid4().hex[:8]
    d["name"] = ch.name or ch.type
    return d


def _mask(ch: dict) -> dict:
    out = dict(ch)
    for k in ("password", "token"):
        if out.get(k):
            out[k] = "••••••"
    return out


def _channels() -> list[dict]:
    return list((CFG.get("notifications") or {}).get("channels") or [])


def _save(channels: list[dict], dashboard_url: str | None = None) -> None:
    n = dict(CFG.get("notifications") or {})
    n["channels"] = channels
    if dashboard_url is not None:
        n["dashboard_url"] = dashboard_url
    CFG["notifications"] = n
    config.save_config(CFG)


@router.get("/notifications")
def get_notifications():
    nt = get_notifier()
    return {"channels": [_mask(c) for c in _channels()], "events": EVENTS, "types": CHANNEL_TYPES,
            "dashboard_url": nt.dashboard_url(), "dashboard_url_set": bool((CFG.get("notifications") or {}).get("dashboard_url")),
            "history": nt.history[:20]}


@router.post("/notifications/channels")
def upsert_channel(ch: Channel):
    d = _validate(ch)
    chans = _channels()
    old = next((c for c in chans if c["id"] == d["id"]), None)
    if old:   # keep secrets that came back masked
        for k in ("password", "token"):
            if d.get(k) == "••••••":
                d[k] = old.get(k)
        chans = [d if c["id"] == d["id"] else c for c in chans]
    else:
        chans.append(d)
    _save(chans)
    return _mask(d)


@router.delete("/notifications/channels/{cid}")
def delete_channel(cid: str):
    _save([c for c in _channels() if c["id"] != cid])
    return {"ok": True}


@router.post("/notifications/channels/{cid}/test")
async def test_channel(cid: str):
    ch = next((c for c in _channels() if c["id"] == cid), None)
    if not ch:
        raise HTTPException(404, "channel not found")
    try:
        await run_in_threadpool(get_notifier().send_test, ch)
    except Exception as e:
        raise HTTPException(502, f"Sending failed: {e}")
    return {"ok": True}


class DashUrl(BaseModel):
    dashboard_url: str = ""


@router.put("/notifications/dashboard-url")
def set_dashboard_url(body: DashUrl):
    _save(_channels(), body.dashboard_url.strip())
    return {"ok": True}

