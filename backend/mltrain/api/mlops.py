"""MLOps API: model registry, deployments, serving, monitoring, alerts, automation, demo."""
from __future__ import annotations

import hmac
import json
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import db
from ..analysis import session_status
from ..mlops import automation
from ..mlops.deployments import analyze, create as create_dep, event, get_deployment, promote, rollback, run_monitor
from ..mlops.export import build_export
from ..mlops.monitoring import DEFAULT_CONFIG
from ..mlops.registry import get_version, load_profile, register_version, set_stage
from ..mlops.serving import SERVING
from ..mlops.service import MLOPS
from .state import AUTH, protected_router

router = protected_router()
serve = APIRouter(prefix="/api/serve")


# ===================================================================== registry
class RegisterBody(BaseModel):
    model_name: str
    session_id: str
    run_id: str | None = None
    description: str = ""
    variant: str = "best"


@router.get("/registry")
def list_registry():
    with db.get_conn() as c:
        models = db.rows(c, "SELECT * FROM registered_models ORDER BY created_at DESC")
        versions = db.rows(c, "SELECT * FROM model_versions ORDER BY version DESC")
        deps = db.rows(c, "SELECT id, name, model_name, champion_version_id, challenger_version_id FROM deployments")
    for m in models:
        m["versions"] = [v for v in versions if v["model_name"] == m["name"]]
        m["deployments"] = [d for d in deps if d["model_name"] == m["name"]]
    return models


@router.post("/registry")
async def register(body: RegisterBody):
    v = await run_in_threadpool(register_version, body.model_name, body.session_id, body.run_id, body.description, "user", body.variant)
    return v


def _version_detail(c, vid: str) -> dict:
    v = get_version(c, vid)
    p = Path(v["path"])
    v["card"] = json.loads((p / "card.json").read_text()) if (p / "card.json").exists() else {}
    prof = load_profile(v)
    v["profile"] = {"kind": prof["kind"], "task": prof["task"], "classes": prof.get("classes"), "baseline": prof.get("baseline"),
                    "n_reference": prof.get("n_reference"), "schema": prof.get("schema"), "example": prof.get("example"),
                    "features": {k: {kk: vv for kk, vv in f.items() if kk != "sample"} for k, f in (prof.get("features") or {}).items()},
                    "prediction": prof.get("prediction")}
    v["size_bytes"] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    v["deployed_in"] = db.rows(c, "SELECT id, name, CASE WHEN champion_version_id=? THEN 'champion' ELSE 'challenger' END AS role "
                                  "FROM deployments WHERE champion_version_id=? OR challenger_version_id=?", (vid, vid, vid))
    return v


@router.get("/registry/versions/{vid}")
def version_detail(vid: str):
    with db.get_conn() as c:
        return _version_detail(c, vid)


class VersionPatch(BaseModel):
    stage: str | None = None
    description: str | None = None


@router.patch("/registry/versions/{vid}")
def patch_version(vid: str, body: VersionPatch):
    with db.get_conn() as c:
        get_version(c, vid)
        if body.stage is not None:
            if body.stage == "production":
                raise HTTPException(400, "Use a deployment's Promote action to put a version in production")
            set_stage(c, vid, body.stage)
        if body.description is not None:
            c.execute("UPDATE model_versions SET description=? WHERE id=?", (body.description, vid))
        return _version_detail(c, vid)


@router.delete("/registry/versions/{vid}")
def delete_version(vid: str):
    with db.get_conn() as c:
        v = get_version(c, vid)
        if db.one(c, "SELECT id FROM deployments WHERE champion_version_id=? OR challenger_version_id=?", (vid, vid)):
            raise HTTPException(409, "This version is used by a deployment")
        c.execute("DELETE FROM model_versions WHERE id=?", (vid,))
        if not db.one(c, "SELECT id FROM model_versions WHERE model_name=?", (v["model_name"],)) and \
                not db.one(c, "SELECT id FROM deployments WHERE model_name=?", (v["model_name"],)):
            c.execute("DELETE FROM registered_models WHERE name=?", (v["model_name"],))
    SERVING.evict(vid)
    shutil.rmtree(v["path"], ignore_errors=True)
    return {"ok": True}


@router.get("/registry/versions/{vid}/export")
def export_version(vid: str, request: Request, deployment: str | None = None):
    with db.get_conn() as c:
        v = get_version(c, vid)
    data = build_export(v, deployment, str(request.base_url).rstrip("/"))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{v["model_name"]}-v{v["version"]}.zip"'})


# ===================================================================== deployments
class DeploymentCreate(BaseModel):
    name: str
    version_id: str
    kind: str = "internal"
    config: dict[str, Any] = {}
    automation: dict[str, Any] = {}


@router.post("/deployments")
def create_deployment(body: DeploymentCreate):
    if body.kind not in ("internal", "external"):
        raise HTTPException(400, "kind must be internal or external")
    with db.get_conn() as c:
        return {"id": create_dep(c, body.name, body.version_id, body.kind, body.config, body.automation)}


def _dep_summary(c, d: dict) -> dict:
    now = time.time()
    n24 = c.execute("SELECT COUNT(*), AVG(error IS NOT NULL) FROM predictions WHERE deployment_id=? AND shadow=0 AND ts>=?",
                    (d["id"], now - 86400)).fetchone()
    nl = c.execute("SELECT COUNT(*) FROM predictions WHERE deployment_id=? AND shadow=0 AND label IS NOT NULL AND ts>=?",
                   (d["id"], now - 86400)).fetchone()[0]
    champ = db.one(c, "SELECT id, version, stage FROM model_versions WHERE id=?", (d["champion_version_id"],)) if d["champion_version_id"] else None
    chall = db.one(c, "SELECT id, version, stage FROM model_versions WHERE id=?", (d["challenger_version_id"],)) if d["challenger_version_id"] else None
    alerts = db.rows(c, "SELECT severity FROM alerts WHERE deployment_id=? AND status='open'", (d["id"],))
    snap = db.one(c, "SELECT ts, metrics FROM monitor_snapshots WHERE deployment_id=? ORDER BY ts DESC LIMIT 1", (d["id"],))
    m = snap["metrics"] if snap else {}
    out = {k: v for k, v in d.items() if k != "api_key"}
    out.update(champion=champ, challenger=chall, requests_24h=n24[0], error_rate_24h=n24[1] or 0, labeled_24h=nl,
               open_alerts=len(alerts), critical_alerts=sum(a["severity"] == "critical" for a in alerts),
               traffic=MLOPS.traffic_state(d["id"]),
               latest={"ts": snap["ts"] if snap else None, "n": m.get("n"), "max_psi": (m.get("drift") or {}).get("max_psi"),
                       "pred_psi": (m.get("prediction") or {}).get("psi"), "latency_p95": (m.get("latency") or {}).get("p95"),
                       "perf": (m.get("performance") or {}).get("headline")})
    return out


@router.get("/deployments")
def list_deployments():
    with db.get_conn() as c:
        return [_dep_summary(c, d) for d in db.rows(c, "SELECT * FROM deployments ORDER BY created_at DESC")]


@router.get("/deployments/{did}")
def deployment_detail(did: str, request: Request):
    with db.get_conn() as c:
        d = get_deployment(c, did)
        out = _dep_summary(c, d)
        out["api_key"] = d["api_key"]
        out["config"] = {**DEFAULT_CONFIG, **(d["config"] or {})}
        out["automation"] = automation.automation_of(d)
        out["versions"] = db.rows(c, "SELECT id, version, stage, metrics, created_at, created_by, description, source_session_id "
                                     "FROM model_versions WHERE model_name=? ORDER BY version DESC", (d["model_name"],))
        champ = db.one(c, "SELECT * FROM model_versions WHERE id=?", (d["champion_version_id"],)) if d["champion_version_id"] else None
        if champ:
            prof = load_profile(champ)
            out["schema"] = {"kind": prof["kind"], "task": prof["task"], "classes": prof.get("classes"),
                             "input_schema": prof.get("schema"), "example": prof.get("example"), "baseline": prof.get("baseline")}
        out["endpoint"] = f"{str(request.base_url).rstrip('/')}/api/serve/{did}"
        return out


class DeploymentPatch(BaseModel):
    name: str | None = None
    status: str | None = None
    config: dict[str, Any] | None = None
    automation: dict[str, Any] | None = None
    challenger_version_id: str | None = None
    challenger_mode: str | None = None
    canary_fraction: float | None = None


@router.patch("/deployments/{did}")
def patch_deployment(did: str, body: DeploymentPatch, request: Request):
    with db.get_conn() as c:
        d = get_deployment(c, did)
        data = body.model_dump(exclude_unset=True)
        if "status" in data and data["status"] not in ("active", "paused"):
            raise HTTPException(400, "status must be active or paused")
        if "challenger_mode" in data and data["challenger_mode"] not in ("none", "shadow", "canary"):
            raise HTTPException(400, "challenger_mode must be none, shadow or canary")
        if data.get("challenger_version_id"):
            v = get_version(c, data["challenger_version_id"])
            if v["model_name"] != d["model_name"]:
                raise HTTPException(400, "Challenger must be a version of the same registered model")
        if "canary_fraction" in data and not 0 < data["canary_fraction"] <= 0.5:
            raise HTTPException(400, "canary_fraction must be in (0, 0.5]")
        for k in ("config", "automation"):
            if k in data:
                merged = {**(d[k] or {}), **data[k]}
                if k == "automation" and "data" in data[k]:
                    merged["data"] = {**((d[k] or {}).get("data") or {}), **data[k]["data"]}
                data[k] = json.dumps(merged)
        if data.get("challenger_version_id") is None and "challenger_version_id" in data:
            data["challenger_mode"] = "none"
        for k, v in data.items():
            c.execute(f"UPDATE deployments SET {k}=? WHERE id=?", (v, did))
        changed = ", ".join(data)
        if changed:
            event(c, did, "config", f"Updated {changed}")
    return deployment_detail(did, request)


@router.delete("/deployments/{did}")
def delete_deployment(did: str):
    MLOPS.stop_traffic(did)
    with db.get_conn() as c:
        get_deployment(c, did)
        c.execute("DELETE FROM deployments WHERE id=?", (did,))
    return {"ok": True}


class PromoteBody(BaseModel):
    version_id: str


@router.post("/deployments/{did}/promote")
def promote_version(did: str, body: PromoteBody):
    with db.get_conn() as c:
        promote(c, did, body.version_id, reason="manual")
        for r in db.rows(c, "SELECT id FROM automation_runs WHERE deployment_id=? AND challenger_version_id=? AND status='awaiting_approval'",
                         (did, body.version_id)):
            c.execute("UPDATE automation_runs SET status='promoted', updated_at=? WHERE id=?", (time.time(), r["id"]))
    return {"ok": True}


@router.post("/deployments/{did}/rollback")
def rollback_version(did: str):
    with db.get_conn() as c:
        return {"version_id": rollback(c, did)}


@router.post("/deployments/{did}/monitor")
async def monitor_now(did: str):
    m = await run_in_threadpool(run_monitor, did)
    return {"health": m["health"] if m else None}


# ---------------------------------------------------------------- monitoring data
@router.get("/deployments/{did}/snapshots")
def snapshots(did: str, hours: float = 24):
    """Time series of monitoring snapshots (slimmed)."""
    with db.get_conn() as c:
        get_deployment(c, did)
        snaps = db.rows(c, "SELECT ts, n, health, metrics FROM monitor_snapshots WHERE deployment_id=? AND ts>=? ORDER BY ts",
                        (did, time.time() - hours * 3600))
    out = []
    for s in snaps:
        m = s["metrics"] or {}
        perf = m.get("performance") or {}
        out.append({"ts": s["ts"], "n": s["n"], "health": s["health"], "error_rate": m.get("error_rate"),
                    "throughput_per_min": m.get("throughput_per_min"), "latency": m.get("latency"),
                    "max_psi": (m.get("drift") or {}).get("max_psi"), "mean_psi": (m.get("drift") or {}).get("mean_psi"),
                    "n_drifted": (m.get("drift") or {}).get("n_drifted"),
                    "feature_psi": {k: v.get("psi") for k, v in (m.get("features") or {}).items()},
                    "feature_mean": {k: v.get("mean") for k, v in (m.get("features") or {}).items()},
                    "prediction": {k: (m.get("prediction") or {}).get(k) for k in ("psi", "cur_hist", "classes", "mean")},
                    "score_mean": (m.get("score") or {}).get("mean"),
                    "performance": {"n_labeled": perf.get("n_labeled"), "metrics": perf.get("metrics"), "headline": perf.get("headline")},
                    "shadow": m.get("shadow"), "quality_rate": (m.get("quality") or {}).get("worst_rate")})
    return out


@router.get("/deployments/{did}/analysis")
def analysis(did: str, minutes: float = 60, start: float | None = None, end: float | None = None):
    """Full monitoring analysis for a time range, with reference histograms for side-by-side plots."""
    with db.get_conn() as c:
        d = get_deployment(c, did)
        t1 = end or time.time()
        t0 = start or t1 - minutes * 60
        m = analyze(c, d, t0, t1)
        prof = load_profile(get_version(c, d["champion_version_id"]))
    m["reference"] = {k: {kk: vv for kk, vv in f.items() if kk != "sample"} for k, f in (prof.get("features") or {}).items()}
    m["reference_prediction"] = prof.get("prediction")
    m["baseline"] = prof.get("baseline")
    return m


@router.get("/deployments/{did}/predictions")
def recent_predictions(did: str, limit: int = 100, labeled: bool | None = None, shadow: int = 0):
    with db.get_conn() as c:
        q = "SELECT p.id, p.request_id, p.ts, p.payload, p.features, p.prediction, p.score, p.latency_ms, p.error, p.label, p.label_ts, " \
            "p.source, v.version FROM predictions p LEFT JOIN model_versions v ON v.id=p.version_id WHERE p.deployment_id=? AND p.shadow=?"
        if labeled is not None:
            q += " AND p.label IS " + ("NOT NULL" if labeled else "NULL")
        return db.rows(c, q + " ORDER BY p.ts DESC LIMIT ?", (did, shadow, min(limit, 1000)))


@router.get("/deployments/{did}/events")
def events(did: str, limit: int = 200):
    with db.get_conn() as c:
        return db.rows(c, "SELECT * FROM deployment_events WHERE deployment_id=? ORDER BY ts DESC LIMIT ?", (did, limit))


@router.get("/deployments/{did}/alerts")
def alerts(did: str, include_resolved: bool = False):
    with db.get_conn() as c:
        q = "SELECT * FROM alerts WHERE deployment_id=?" + ("" if include_resolved else " AND status!='resolved'")
        return db.rows(c, q + " ORDER BY CASE severity WHEN 'critical' THEN 0 ELSE 1 END, updated_at DESC LIMIT 200", (did,))


@router.get("/alerts")
def all_alerts():
    with db.get_conn() as c:
        return db.rows(c, "SELECT a.*, d.name AS deployment_name FROM alerts a JOIN deployments d ON d.id=a.deployment_id "
                          "WHERE a.status='open' ORDER BY a.updated_at DESC LIMIT 200")


@router.post("/deployments/{did}/alerts/{aid}/{action}")
def alert_action(did: str, aid: int, action: str):
    if action not in ("acknowledge", "resolve"):
        raise HTTPException(400, "action must be acknowledge or resolve")
    with db.get_conn() as c:
        c.execute("UPDATE alerts SET status=?, updated_at=?, resolved_at=CASE WHEN ?='resolved' THEN ? ELSE resolved_at END "
                  "WHERE id=? AND deployment_id=?", ("acknowledged" if action == "acknowledge" else "resolved", time.time(),
                                                     "resolved" if action == "resolve" else "", time.time(), aid, did))
    return {"ok": True}


# ---------------------------------------------------------------- automation
class AutomationRunBody(BaseModel):
    kind: str = "retrain"


@router.post("/deployments/{did}/automation/run")
def automation_run(did: str, body: AutomationRunBody):
    if body.kind not in ("retrain", "evaluate"):
        raise HTTPException(400, "kind must be retrain or evaluate")
    with db.get_conn() as c:
        d = get_deployment(c, did)
    rid = automation.start(d, body.kind, "manual")
    if not rid:
        with db.get_conn() as c:
            last = db.one(c, "SELECT message FROM deployment_events WHERE deployment_id=? ORDER BY ts DESC LIMIT 1", (did,))
        raise HTTPException(409, (last or {}).get("message") or "Could not start (another automation run is in progress?)")
    return {"id": rid}


@router.get("/deployments/{did}/automation/runs")
def automation_runs(did: str):
    with db.get_conn() as c:
        return db.rows(c, "SELECT * FROM automation_runs WHERE deployment_id=? ORDER BY created_at DESC LIMIT 100", (did,))


@router.post("/deployments/{did}/automation/runs/{rid}/{action}")
def automation_decide(did: str, rid: str, action: str):
    with db.get_conn() as c:
        r = db.one(c, "SELECT * FROM automation_runs WHERE id=? AND deployment_id=?", (rid, did))
        if not r or r["status"] != "awaiting_approval":
            raise HTTPException(400, "Run is not awaiting approval")
        if action == "approve":
            promote(c, did, r["challenger_version_id"], reason="approved challenger")
            status = "promoted"
        elif action == "reject":
            set_stage(c, r["challenger_version_id"], "archived")
            d = get_deployment(c, did)
            if d["challenger_version_id"] == r["challenger_version_id"]:
                c.execute("UPDATE deployments SET challenger_version_id=NULL, challenger_mode='none' WHERE id=?", (did,))
            event(c, did, "automation", "Challenger rejected by user")
            status = "rejected"
        else:
            raise HTTPException(400, "action must be approve or reject")
        c.execute("UPDATE automation_runs SET status=?, updated_at=? WHERE id=?", (status, time.time(), rid))
        c.execute("UPDATE alerts SET status='resolved', resolved_at=? WHERE deployment_id=? AND key=?", (time.time(), did, f"automation:{rid}"))
    return {"status": status}


# ---------------------------------------------------------------- demo traffic
class TrafficBody(BaseModel):
    scenario: str = "none"
    rps: float = 3.0
    ramp_minutes: float = 15.0
    label_delay: float = 20.0
    label_fraction: float = 0.8


@router.post("/deployments/{did}/traffic")
def start_traffic(did: str, body: TrafficBody, request: Request):
    from ..mlops.demo import SCENARIOS
    if body.scenario not in SCENARIOS:
        raise HTTPException(400, f"scenario must be one of {SCENARIOS}")
    with db.get_conn() as c:
        d = get_deployment(c, did)
        v = get_version(c, d["champion_version_id"])
    meta = json.loads((Path(v["path"]) / "meta.json").read_text())
    if not str(meta.get("problem_id", "")).startswith(("demo.loan", "deployment.feedback")):
        raise HTTPException(400, "The traffic simulator generates loan applications; it only works for models trained on the loan demo data")
    MLOPS.start_traffic(d, f"http://127.0.0.1:{request.url.port or 8000}", body.scenario, body.rps, body.ramp_minutes,
                        body.label_delay, body.label_fraction)
    return MLOPS.traffic_state(did)


@router.delete("/deployments/{did}/traffic")
def stop_traffic(did: str):
    return {"stopped": MLOPS.stop_traffic(did)}


@router.post("/mlops/demo/setup")
def demo_setup():
    """One click: train the loan-default demo model; it is registered and deployed when training finishes."""
    from .sessions import SessionCreate, create_session
    with db.get_conn() as c:
        if db.one(c, "SELECT id FROM deployments WHERE id='loan-default'"):
            raise HTTPException(409, "The demo deployment 'loan-default' already exists")
        c.execute("BEGIN")
        sid = create_session(c, SessionCreate(problem_kind="dataset", problem_id="demo.loan_default", model_id="xgboost",
                                              model_params={"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4},
                                              seeds=[0], tags=["demo", "mlops"], name="Loan default model (MLOps demo)",
                                              options={"snapshots": False}))
        c.execute("COMMIT")
    MLOPS.pending_demo = sid
    return {"session_id": sid}


@router.get("/mlops/demo")
def demo_state():
    sid = getattr(MLOPS, "pending_demo", None)
    with db.get_conn() as c:
        dep = db.one(c, "SELECT id FROM deployments WHERE id='loan-default'")
        st = None
        if sid:
            st = session_status(db.rows(c, "SELECT status FROM runs WHERE session_id=?", (sid,)))
    return {"deployment": dep["id"] if dep else None, "session_id": sid, "session_status": st}


# ===================================================================== serving (API key or dashboard login)
def _serve_auth(did: str, request: Request) -> dict:
    with db.get_conn() as c:
        d = get_deployment(c, did)
    key = request.headers.get("x-api-key", "")
    if not (key and hmac.compare_digest(key, d["api_key"])) and not AUTH.is_authed(request):
        raise HTTPException(401, "Missing or invalid X-API-Key")
    return d


class PredictBody(BaseModel):
    instances: list[Any]
    request_ids: list[str] | None = None


@serve.post("/{did}/predict")
async def serve_predict(did: str, body: PredictBody, request: Request):
    d = _serve_auth(did, request)
    if len(body.instances) > 1000:
        raise HTTPException(413, "At most 1000 instances per request")
    return await run_in_threadpool(SERVING.predict, d, body.instances, body.request_ids)


class LogBody(BaseModel):
    records: list[dict[str, Any]]


@serve.post("/{did}/log")
async def serve_log(did: str, body: LogBody, request: Request):
    d = _serve_auth(did, request)
    return {"logged": await run_in_threadpool(SERVING.log_external, d, body.records)}


class FeedbackBody(BaseModel):
    items: list[dict[str, Any]]


@serve.post("/{did}/feedback")
def serve_feedback(did: str, body: FeedbackBody, request: Request):
    d = _serve_auth(did, request)
    return {"updated": SERVING.feedback(d, body.items)}


@serve.get("/{did}/info")
def serve_info(did: str, request: Request):
    d = _serve_auth(did, request)
    with db.get_conn() as c:
        v = get_version(c, d["champion_version_id"])
    prof = load_profile(v)
    return {"deployment": did, "model": v["model_name"], "version": v["version"], "task": prof["task"], "input_kind": prof["kind"],
            "classes": prof.get("classes"), "input_schema": prof.get("schema"), "example": prof.get("example")}

