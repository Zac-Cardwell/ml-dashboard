"""Training sessions: create (train / fine-tune / evaluate), list, control, metrics, media, live updates."""
from __future__ import annotations

import asyncio
import importlib.metadata as md
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .. import analysis, db, storage
from ..plugins.base import coerce_params
from ..plugins.registry import get_registry
from ..training.options import options_for
from .state import AUTH, GPU_AVAILABLE, protected_router

router = protected_router()


# ===================================================================== creation
class InitFrom(BaseModel):
    session_id: str | None = None
    strategy: str = "match_seed"   # match_seed | best | run | version
    run_id: str | None = None
    version_id: str | None = None  # a model-registry version (immutable copy)
    variant: str = "best"          # best | final — which saved weights of the source run


class SessionCreate(BaseModel):
    name: str | None = None
    problem_kind: str  # 'dataset' | 'env'
    problem_id: str
    problem_params: dict[str, Any] = {}
    model_id: str
    model_params: dict[str, Any] = {}
    seeds: list[int] = [0]
    device: str = "auto"  # auto | cuda | cpu
    mode: str = "train"   # train | evaluate
    options: dict[str, Any] = {}
    init_from: InitFrom | None = None
    parent_id: str | None = None
    notes: str = ""
    tags: list[str] = []
    priority: int = 0                 # -1 low, 0 normal, 1 high, 2 urgent
    scheduled_at: float | None = None  # unix time; the session waits in the queue until then


def lib_versions() -> dict:
    out = {}
    for lib in ("torch", "torchvision", "scikit-learn", "xgboost", "lightgbm", "stable-baselines3", "sb3-contrib", "gymnasium",
                "transformers", "datasets", "numpy"):
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    return out


def _saved_runs(c, session_id: str, variant: str = "best") -> list[dict]:
    runs = db.rows(c, "SELECT id, seed, summary, status FROM runs WHERE session_id=? ORDER BY seed", (session_id,))
    return [r for r in runs if storage.model_meta(r["id"], variant)]


def _pick_init_run(saved: list[dict], seed: int, init: InitFrom, task: str | None) -> str:
    if init.strategy == "run":
        if not any(r["id"] == init.run_id for r in saved):
            raise HTTPException(400, f"Run {init.run_id} has no saved model")
        return init.run_id
    if init.strategy == "match_seed":
        for r in saved:
            if r["seed"] == seed:
                return r["id"]
    key, hib = analysis.HEADLINE.get(task, (None, True))
    scored = [r for r in saved if key and isinstance((r["summary"] or {}).get(key), (int, float))]
    if scored:
        return (max if hib else min)(scored, key=lambda r: r["summary"][key])["id"]
    return saved[0]["id"]


def create_session(c, body: SessionCreate, group_id: str | None = None, group_kind: str | None = None,
                   group_params: dict | None = None) -> str:
    reg = get_registry()
    try:
        problem = reg.problem(body.problem_kind, body.problem_id)
        model = reg.get("model", body.model_id)
    except KeyError as e:
        raise HTTPException(400, str(e))
    if not model.compatible(problem.task, problem.modality):
        raise HTTPException(400, f"{model.name} does not support {problem.task}/{problem.modality} problems")
    if body.mode not in ("train", "evaluate"):
        raise HTTPException(400, "mode must be train or evaluate")

    source, saved, version = None, [], None
    if body.init_from and body.init_from.version_id:
        version = db.one(c, "SELECT * FROM model_versions WHERE id=?", (body.init_from.version_id,))
        if not version:
            raise HTTPException(400, "model version not found")
        if version["model_id"] != model.id:
            raise HTTPException(400, f"That version is a '{version['model_id']}' model; select that model type")
        body.init_from.strategy = "version"
        body.init_from.session_id = version["source_session_id"]
        source = db.one(c, "SELECT * FROM sessions WHERE id=?", (version["source_session_id"],)) or {
            "id": None, "model_params": json.loads((Path(version["path"]) / "meta.json").read_text())["model_params"],
            "task": version["task"], "name": f"{version['model_name']} v{version['version']}"}
    elif body.init_from:
        source = db.one(c, "SELECT * FROM sessions WHERE id=?", (body.init_from.session_id,))
        if not source:
            raise HTTPException(400, "init_from session not found")
        if source["model_id"] != model.id:
            raise HTTPException(400, f"That saved model is a '{source['model_id']}'; select that model type to use it")
        if body.init_from.variant not in storage.VARIANT_DIRS:
            raise HTTPException(400, "variant must be best or final")
        saved = _saved_runs(c, source["id"], body.init_from.variant)
        if not saved:
            what = "final-epoch models" if body.init_from.variant == "final" else "saved models"
            raise HTTPException(400, f"Session '{source['name']}' has no {what} (deleted, or saving was off)")
        if body.mode == "train" and not model.supports_init:
            raise HTTPException(400, f"{model.name} cannot start from a saved model")
    if body.mode == "evaluate":
        if not model.supports_eval:
            raise HTTPException(400, f"{model.name} does not support evaluation-only runs")
        if not source and not model.eval_without_init:
            raise HTTPException(400, "Choose a saved model to evaluate")

    try:
        pparams = coerce_params(problem.params(), body.problem_params)
        base = source["model_params"] if source and body.mode == "evaluate" else {}
        mparams = coerce_params(model.params(), {**base, **body.model_params})
        options = coerce_params(options_for(getattr(model, "session_task", None) or problem.task, body.mode), body.options)
    except ValueError as e:
        raise HTTPException(400, str(e))
    seeds = list(dict.fromkeys(body.seeds))
    if not seeds or len(seeds) > 100:
        raise HTTPException(400, "Provide 1-100 seeds")
    device = body.device
    if device == "auto":
        device = "cuda" if model.uses_gpu and GPU_AVAILABLE else "cpu"
    if device not in ("cuda", "cpu"):
        raise HTTPException(400, "device must be auto, cuda or cpu")
    if not -1 <= body.priority <= 2:
        raise HTTPException(400, "priority must be -1 (low) .. 2 (urgent)")
    parent_id = body.parent_id
    if parent_id and not db.one(c, "SELECT id FROM sessions WHERE id=?", (parent_id,)):
        raise HTTPException(400, "parent session not found")
    if source and source.get("id") and body.mode == "train" and not parent_id:
        parent_id = source["id"]      # fine-tuning continues that session's lineage

    sid, now = db.new_id(), db.now()
    name = body.name
    if not name:
        n = c.execute("SELECT COUNT(*) FROM sessions WHERE model_id=? AND problem_id=?", (model.id, problem.id)).fetchone()[0]
        prefix = "Eval · " if body.mode == "evaluate" else ""
        name = f"{prefix}{model.name} · {problem.name} #{n + 1}"
        if group_params:
            name += " · " + ", ".join(f"{k.split('.', 1)[-1]}={v}" for k, v in group_params.items())
    hashes = {src: h for src, h in reg.hashes.items()
              if src.endswith(Path(model.source).name) or src.endswith(Path(problem.source).name)}
    init_json = body.init_from.model_dump() if body.init_from else None
    c.execute("INSERT INTO sessions(id,name,created_at,problem_kind,problem_id,problem_params,model_id,model_params,"
              "task,modality,framework,seeds,device,parent_id,notes,tags,env_info,mode,options,init_from,group_id,group_kind,group_params,"
              "priority,scheduled_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (sid, name, now, body.problem_kind, problem.id, json.dumps(pparams), model.id, json.dumps(mparams),
               getattr(model, "session_task", None) or problem.task, problem.modality, model.framework, json.dumps(seeds), device, parent_id,
               body.notes, json.dumps(body.tags), json.dumps({"versions": lib_versions(), "plugin_hashes": hashes}),
               body.mode, json.dumps(options), json.dumps(init_json) if init_json else None,
               group_id, group_kind, json.dumps(group_params or {}), body.priority,
               body.scheduled_at if body.scheduled_at and body.scheduled_at > now else None))
    for seed in seeds:
        init_run = _pick_init_run(saved, seed, body.init_from, source["task"]) if saved else None
        init_path = version["path"] if version else None
        if init_run and body.init_from.variant == "final":
            init_path = str(storage.model_dir(init_run, "final"))
        c.execute("INSERT INTO runs(id,session_id,seed,status,created_at,init_run_id,init_path) VALUES (?,?,?,?,?,?,?)",
                  (db.new_id(), sid, seed, "queued", now, init_run, init_path))
    return sid


@router.post("/sessions")
def post_session(body: SessionCreate):
    with db.get_conn() as c:
        c.execute("BEGIN")
        try:
            sid = create_session(c, body)
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
    return {"id": sid}


@router.get("/options")
def get_options(task: str, mode: str = "train"):
    return [p.to_dict() for p in options_for(task, mode)]


# ===================================================================== listing / detail
def runs_by_session(c, ids: list[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        if not chunk:
            continue
        q = ",".join("?" * len(chunk))
        for r in db.rows(c, f"SELECT * FROM runs WHERE session_id IN ({q}) ORDER BY seed", tuple(chunk)):
            out.setdefault(r["session_id"], []).append(r)
    return out


@router.get("/sessions")
def list_sessions(status: str | None = None, q: str | None = None, group: str | None = None, limit: int = 1000):
    with db.get_conn() as c:
        if group:
            sessions = db.rows(c, "SELECT * FROM sessions WHERE group_id=? ORDER BY created_at", (group,))
        else:
            sessions = db.rows(c, "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,))
        rb = runs_by_session(c, [s["id"] for s in sessions])
    out = []
    for s in sessions:
        b = analysis.session_brief(s, rb.get(s["id"], []))
        if status and b["status"] != status:
            continue
        if q and q.lower() not in (s["name"] + s["model_id"] + s["problem_id"] + " ".join(s["tags"])).lower():
            continue
        out.append(b)
    return out


def get_session_row(c, sid: str) -> dict:
    s = db.one(c, "SELECT * FROM sessions WHERE id=?", (sid,))
    if not s:
        raise HTTPException(404, "session not found")
    return s


def session_detail(c, sid: str) -> dict:
    s = get_session_row(c, sid)
    runs = db.rows(c, "SELECT * FROM runs WHERE session_id=? ORDER BY seed", (sid,))
    for r in runs:
        r["storage"] = storage.run_storage(r["id"])
    stats = analysis.summary_stats([r for r in runs if r["status"] == "completed"])
    parent = db.one(c, "SELECT * FROM sessions WHERE id=?", (s["parent_id"],)) if s["parent_id"] else None
    children = db.rows(c, "SELECT id, name, created_at, mode FROM sessions WHERE parent_id=? ORDER BY created_at", (sid,))
    evaluations = []
    run_ids = [r["id"] for r in runs]
    if run_ids:
        q = ",".join("?" * len(run_ids))
        evaluations = db.rows(c, f"SELECT DISTINCT s.id, s.name, s.problem_id, s.created_at FROM sessions s JOIN runs r ON r.session_id=s.id "
                                 f"WHERE r.init_run_id IN ({q}) AND s.id != ? ORDER BY s.created_at", (*run_ids, sid))
    source = None
    if s.get("init_from"):
        src = db.one(c, "SELECT id, name FROM sessions WHERE id=?", (s["init_from"].get("session_id"),))
        source = src or {"id": s["init_from"].get("session_id"), "name": "(deleted session)"}
        if s["init_from"].get("version_id"):
            v = db.one(c, "SELECT model_name, version FROM model_versions WHERE id=?", (s["init_from"]["version_id"],))
            if v:
                source = {**source, "version": f"{v['model_name']} v{v['version']}"}
    reg = get_registry()
    plugin_meta = {}
    for kind, pid in (("model", s["model_id"]), (s["problem_kind"], s["problem_id"])):
        try:
            plugin_meta[kind] = reg.get(kind, pid).meta()
        except KeyError:
            plugin_meta[kind] = None
    return {
        **analysis.session_brief(s, runs),
        "problem_params": s["problem_params"], "model_params": s["model_params"], "seeds": s["seeds"],
        "notes": s["notes"], "env_info": s["env_info"], "options": s.get("options") or {},
        "init_from": s.get("init_from"), "source": source,
        "runs": runs, "summary_stats": stats,
        "parent": {"id": parent["id"], "name": parent["name"]} if parent else None,
        "changes_from_parent": analysis.param_diff(analysis.session_config(parent), analysis.session_config(s)) if parent else [],
        "children": children, "evaluations": evaluations,
        "plugins": plugin_meta,
        # sessions that reuse a part of this model (e.g. its decoder as a GAN generator)
        "component_users": db.rows(c, "SELECT id, name, model_id FROM sessions WHERE model_params LIKE ? AND id != ? ORDER BY created_at",
                                   (f'%"session:{sid}:%', sid)),
    }


@router.get("/sessions/{sid}")
def get_session(sid: str):
    with db.get_conn() as c:
        return session_detail(c, sid)


class SessionPatch(BaseModel):
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None


@router.patch("/sessions/{sid}")
def patch_session(sid: str, body: SessionPatch):
    with db.get_conn() as c:
        get_session_row(c, sid)
        for k, v in body.model_dump(exclude_none=True).items():
            c.execute(f"UPDATE sessions SET {k}=? WHERE id=?", (json.dumps(v) if k == "tags" else v, sid))
        return session_detail(c, sid)


# ===================================================================== control
@router.post("/sessions/{sid}/stop")
def stop_session(sid: str):
    with db.get_conn() as c:
        get_session_row(c, sid)
        c.execute("UPDATE runs SET status='stopped', ended_at=? WHERE session_id=? AND status='queued'", (db.now(), sid))
        c.execute("UPDATE runs SET status='stopping' WHERE session_id=? AND status='running'", (sid,))
    return {"ok": True}


def _clear_run_outputs(c, ids: list[str]) -> None:
    if not ids:
        return
    q = ",".join("?" * len(ids))
    for t in ("metrics", "artifacts", "media"):
        c.execute(f"DELETE FROM {t} WHERE run_id IN ({q})", tuple(ids))
    for rid in ids:
        for sub in ("media", "checkpoint", "model"):
            shutil.rmtree(storage.run_dir(rid) / sub, ignore_errors=True)
        (storage.run_dir(rid) / "live.jpg").unlink(missing_ok=True)


@router.post("/sessions/{sid}/retry")
def retry_failed(sid: str):
    """Re-run failed/stopped seeds from scratch (same config)."""
    with db.get_conn() as c:
        get_session_row(c, sid)
        ids = [r["id"] for r in db.rows(c, "SELECT id FROM runs WHERE session_id=? AND status IN ('failed','stopped')", (sid,))]
        _clear_run_outputs(c, ids)
        for rid in ids:
            db.update_run(c, rid, status="queued", error=None, progress=0, progress_text="", started_at=None, ended_at=None,
                          summary={}, created_at=db.now(), resume=0)
    return {"requeued": len(ids)}


@router.post("/sessions/{sid}/resume")
def resume_session(sid: str):
    """Continue stopped/failed seeds from their last checkpoint."""
    with db.get_conn() as c:
        s = get_session_row(c, sid)
        model = get_registry().plugins["model"].get(s["model_id"])
        if not model or not model.supports_resume:
            raise HTTPException(400, "This model type cannot resume from checkpoints; use Retry instead")
        runs = db.rows(c, "SELECT id FROM runs WHERE session_id=? AND status IN ('failed','stopped')", (sid,))
        ids = [r["id"] for r in runs if storage.run_storage(r["id"])["has_checkpoint"]]
        if not ids:
            raise HTTPException(400, "No stopped/failed seeds with a checkpoint to resume")
        for rid in ids:
            db.update_run(c, rid, status="queued", error=None, ended_at=None, created_at=db.now(), resume=1,
                          progress_text="resuming")
    return {"resumed": len(ids)}


@router.delete("/sessions/{sid}")
def delete_session(sid: str):
    with db.get_conn() as c:
        get_session_row(c, sid)
        runs = db.rows(c, "SELECT id, status FROM runs WHERE session_id=?", (sid,))
        if any(r["status"] in ("running", "stopping") for r in runs):
            raise HTTPException(409, "Stop the session before deleting it")
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    for r in runs:
        shutil.rmtree(storage.run_dir(r["id"]), ignore_errors=True)
    return {"ok": True}


# ===================================================================== queue
class QueuePatch(BaseModel):
    priority: int | None = None
    scheduled_at: float | None = None
    clear_schedule: bool = False      # start as soon as a slot is free


@router.get("/queue")
def get_queue():
    """Running and waiting sessions in the order the scheduler will process them."""
    import time as _t
    with db.get_conn() as c:
        rows = db.rows(c, "SELECT s.id, s.name, s.priority, s.scheduled_at, s.device, s.model_id, s.problem_id, s.created_at, "
                          "SUM(r.status='running') AS running, SUM(r.status='queued') AS queued, COUNT(r.id) AS runs "
                          "FROM sessions s JOIN runs r ON r.session_id=s.id GROUP BY s.id "
                          "HAVING SUM(r.status IN ('running','queued','stopping'))>0 "
                          "ORDER BY running DESC, (s.scheduled_at IS NOT NULL AND s.scheduled_at>?) ASC, s.priority DESC, s.created_at",
                       (_t.time(),))
    return rows


@router.patch("/sessions/{sid}/queue")
def patch_queue(sid: str, body: QueuePatch):
    with db.get_conn() as c:
        get_session_row(c, sid)
        if body.priority is not None:
            if not -1 <= body.priority <= 2:
                raise HTTPException(400, "priority must be -1 .. 2")
            c.execute("UPDATE sessions SET priority=? WHERE id=?", (body.priority, sid))
        if body.clear_schedule:
            c.execute("UPDATE sessions SET scheduled_at=NULL WHERE id=?", (sid,))
        elif body.scheduled_at is not None:
            c.execute("UPDATE sessions SET scheduled_at=? WHERE id=?", (body.scheduled_at, sid))
    return {"ok": True}


# ===================================================================== saved models of a session
@router.delete("/sessions/{sid}/models")
def delete_session_models(sid: str, checkpoints: bool = True):
    with db.get_conn() as c:
        get_session_row(c, sid)
        runs = db.rows(c, "SELECT id, status FROM runs WHERE session_id=?", (sid,))
    if any(r["status"] in ("running", "stopping") for r in runs):
        raise HTTPException(409, "Stop the session first")
    return {"freed_bytes": sum(storage.delete_model(r["id"], checkpoints) for r in runs)}


@router.delete("/runs/{rid}/model")
def delete_run_model(rid: str, checkpoints: bool = True, variant: str | None = None):
    with db.get_conn() as c:
        r = db.one(c, "SELECT status FROM runs WHERE id=?", (rid,))
    if not r:
        raise HTTPException(404, "run not found")
    if r["status"] in ("running", "stopping"):
        raise HTTPException(409, "Stop the run first")
    if variant and variant not in storage.VARIANT_DIRS:
        raise HTTPException(400, "variant must be best or final")
    return {"freed_bytes": storage.delete_model(rid, checkpoints, variant)}


# ===================================================================== metrics / lineage / artifacts / media
@router.get("/sessions/{sid}/metrics")
def session_metrics(sid: str, since_id: int = 0, aggregate: bool = True):
    with db.get_conn() as c:
        get_session_row(c, sid)
        runs = db.rows(c, "SELECT id, seed FROM runs WHERE session_id=?", (sid,))
        series, last = analysis.load_series(c, [r["id"] for r in runs], since_id)
    out = {"last_id": last, "runs": {r["id"]: r["seed"] for r in runs}, "series": series}
    if aggregate:
        out["aggregate"] = analysis.aggregate_series(series)
    return out


@router.get("/sessions/{sid}/lineage")
def session_lineage(sid: str):
    with db.get_conn() as c:
        get_session_row(c, sid)
        return analysis.lineage(c, sid)


@router.get("/sessions/{sid}/artifacts")
def session_artifacts(sid: str):
    with db.get_conn() as c:
        return db.rows(c, "SELECT a.run_id, r.seed, a.name, a.kind, a.data FROM artifacts a JOIN runs r ON r.id=a.run_id "
                          "WHERE r.session_id=? ORDER BY a.name, r.seed", (sid,))


@router.get("/sessions/{sid}/media")
def session_media(sid: str, since_id: int = 0):
    """Learning snapshots (videos, decision boundaries, generated text, ...) ordered by step."""
    with db.get_conn() as c:
        return db.rows(c, "SELECT m.id, m.run_id, r.seed, m.step, m.name, m.kind, m.path, m.data, m.t FROM media m "
                          "JOIN runs r ON r.id=m.run_id WHERE r.session_id=? AND m.id>? ORDER BY m.name, r.seed, m.step",
                       (sid, since_id))


@router.get("/runs/{rid}/log", response_class=PlainTextResponse)
def run_log(rid: str, tail: int = 400):
    path = storage.run_dir(rid) / "log.txt"
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-tail:])


@router.get("/runs/{rid}/files")
def run_files(rid: str):
    d = storage.run_dir(rid)
    if not d.exists():
        return []
    return [{"name": str(p.relative_to(d)), "size": p.stat().st_size} for p in sorted(d.rglob("*"))
            if p.is_file() and not p.name.endswith(".tmp")]


def _safe_path(rid: str, name: str) -> Path:
    base = storage.run_dir(rid).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, "not found")
    return path


@router.get("/runs/{rid}/files/{name:path}")
def run_file(rid: str, name: str, inline: bool = False):
    path = _safe_path(rid, name)
    return FileResponse(path) if inline else FileResponse(path, filename=path.name)


@router.get("/runs/{rid}/live.jpg")
def run_live_frame(rid: str):
    path = storage.run_dir(rid) / "live.jpg"
    if not path.exists():
        raise HTTPException(404, "no live frame")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ===================================================================== JSON export
@router.get("/sessions/{sid}/export")
def export_session(sid: str, max_points: int = 200):
    """Everything about a session in one JSON document (for LLMs, notebooks, archiving)."""
    with db.get_conn() as c:
        detail = session_detail(c, sid)
        series, _ = analysis.load_series(c, [r["id"] for r in detail["runs"]])
        arts = db.rows(c, "SELECT a.run_id, a.name, a.kind, a.data FROM artifacts a JOIN runs r ON r.id=a.run_id WHERE r.session_id=?", (sid,))
        media = db.rows(c, "SELECT m.run_id, m.step, m.name, m.kind, m.path FROM media m JOIN runs r ON r.id=m.run_id "
                           "WHERE r.session_id=? ORDER BY m.step", (sid,))
        lin = analysis.lineage(c, sid)
    seed_of = {r["id"]: r["seed"] for r in detail["runs"]}
    detail.pop("plugins", None)
    return {
        "schema": "mltrain.session/v2",
        "session": detail,
        "curves": {
            "aggregate_across_seeds": analysis.aggregate_series(series, max_points),
            "per_seed": {str(seed_of[rid]): s for rid, s in analysis.downsample(series, max_points).items()},
        },
        "artifacts": [{**a, "seed": seed_of.get(a["run_id"])} for a in arts],
        "snapshots": [{**m, "seed": seed_of.get(m["run_id"])} for m in media],
        "lineage": lin,
    }


# ===================================================================== websocket (live updates)
def register_ws(app, guard) -> None:
    @app.websocket("/api/ws/sessions/{sid}")
    async def ws_session(ws: WebSocket, sid: str):
        if not guard.allowed(ws.client.host if ws.client else None) or not AUTH.is_authed(ws):
            await ws.close(code=4401)
            return
        await ws.accept()
        last_id = int(ws.query_params.get("since_id", 0))
        last_media = 0
        last_runs, done_sent = None, False
        try:
            while True:
                def poll():
                    with db.get_conn() as c:
                        runs = db.rows(c, "SELECT id, seed, status, progress, progress_text, error, started_at, ended_at "
                                          "FROM runs WHERE session_id=? ORDER BY seed", (sid,))
                        series, last = analysis.load_series(c, [r["id"] for r in runs], last_id)
                        ids = [r["id"] for r in runs]
                        m = c.execute(f"SELECT MAX(id) FROM media WHERE run_id IN ({','.join('?' * len(ids))})", ids).fetchone()[0] if ids else 0
                    return runs, series, last, m or 0

                runs, series, new_last, media_max = await asyncio.to_thread(poll)
                if new_last != last_id or runs != last_runs or media_max != last_media:
                    await ws.send_json({"type": "update", "runs": runs, "series": series, "last_id": new_last,
                                        "media_last_id": media_max, "status": analysis.session_status(runs)})
                    last_id, last_runs, last_media = new_last, runs, media_max
                finished = all(r["status"] in ("completed", "failed", "stopped") for r in runs)
                if finished and not done_sent:
                    await ws.send_json({"type": "done"})
                done_sent = finished
                # wait for the next poll — but wake immediately if the client goes away (tab closed, server shutdown),
                # otherwise an idle socket for a finished session would loop forever and block shutdown
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5 if finished else 1.5)
                    if msg.get("type") == "websocket.disconnect":
                        break
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, RuntimeError):
            pass

