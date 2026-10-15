"""Deployment operations: monitoring runs, alerts, promotion / rollback, audit events."""
from __future__ import annotations

import json
import time

from fastapi import HTTPException

from .. import db
from .monitoring import DEFAULT_CONFIG, compute_window
from .registry import load_profile, set_stage

LOG_COLUMNS = "id, version_id, request_id, ts, features, prediction, score, latency_ms, error, label, shadow"


def event(c, dep_id: str, kind: str, message: str, data: dict | None = None) -> None:
    c.execute("INSERT INTO deployment_events(deployment_id,ts,kind,message,data) VALUES (?,?,?,?,?)",
              (dep_id, time.time(), kind, message, json.dumps(data or {}, default=str)))


def create(c, name: str, version_id: str, kind: str = "internal", cfg: dict | None = None, auto: dict | None = None) -> str:
    import secrets

    from .registry import get_version, slug
    did = slug(name)
    if db.one(c, "SELECT id FROM deployments WHERE id=?", (did,)):
        raise HTTPException(409, f"Deployment '{did}' already exists")
    v = get_version(c, version_id)
    c.execute("INSERT INTO deployments(id,name,model_name,kind,api_key,config,automation,created_at) VALUES (?,?,?,?,?,?,?,?)",
              (did, name, v["model_name"], kind, "mk_" + secrets.token_urlsafe(24), json.dumps(cfg or {}), json.dumps(auto or {}), time.time()))
    event(c, did, "created", f"Deployment created ({kind}) for {v['model_name']}")
    promote(c, did, v["id"], reason="initial deployment")
    return did


def get_deployment(c, dep_id: str) -> dict:
    d = db.one(c, "SELECT * FROM deployments WHERE id=?", (dep_id,))
    if not d:
        raise HTTPException(404, "deployment not found")
    return d


def cfg_of(dep: dict) -> dict:
    return {**DEFAULT_CONFIG, **(dep.get("config") or {})}


def load_rows(c, dep_id: str, t0: float, t1: float, shadow: int = 0, version_id: str | None = None, limit: int = 100_000) -> list[dict]:
    q = f"SELECT {LOG_COLUMNS} FROM predictions WHERE deployment_id=? AND ts>=? AND ts<? AND shadow=?"
    args: list = [dep_id, t0, t1, shadow]
    if version_id:
        q += " AND version_id=?"
        args.append(version_id)
    return db.rows(c, q + " ORDER BY ts DESC LIMIT ?", (*args, limit))


def analyze(c, dep: dict, t0: float, t1: float, version_id: str | None = None) -> dict:
    """Monitoring metrics for any time range, against the champion's reference profile."""
    vid = version_id or dep["champion_version_id"]
    if not vid:
        raise HTTPException(409, "Deployment has no production model")
    version = db.one(c, "SELECT * FROM model_versions WHERE id=?", (vid,))
    profile = load_profile(version)
    all_rows = load_rows(c, dep["id"], t0, t1)
    # judge each version against its own training profile: right after a promotion (or with a canary) the window
    # contains several versions — only the one being analyzed counts here
    rows = [r for r in all_rows if r["version_id"] == vid]
    m = compute_window(rows, profile, cfg_of(dep), t0, t1)
    m["by_version"] = {}
    for r in all_rows:
        m["by_version"][r["version_id"]] = m["by_version"].get(r["version_id"], 0) + 1
    # champion vs challenger (shadow) on the same requests
    if dep["challenger_version_id"] and dep["challenger_mode"] == "shadow":
        sh = load_rows(c, dep["id"], t0, t1, shadow=1, version_id=dep["challenger_version_id"])
        if sh:
            main = {r["request_id"]: r for r in rows}
            pairs = [(main[s["request_id"]], s) for s in sh if s["request_id"] in main]
            agree = sum(str(a["prediction"]) == str(b["prediction"]) for a, b in pairs) / max(len(pairs), 1)
            # labels live on the production row; copy them across to score the shadow model
            for a, b in pairs:
                b["label"] = a["label"]
            from .monitoring import performance
            m["shadow"] = {"n": len(sh), "agreement": agree, "performance": performance([b for _, b in pairs], profile["task"], profile.get("classes")),
                           "latency_p50": sorted(x["latency_ms"] for x in sh if x["latency_ms"] is not None)[len(sh) // 2] if sh else None}
    return m


def run_monitor(dep_id: str) -> dict | None:
    """Compute and store one monitoring snapshot; update health and alerts."""
    with db.get_conn() as c:
        dep = get_deployment(c, dep_id)
        if not dep["champion_version_id"]:
            return None
        cfg = cfg_of(dep)
        now = time.time()
        m = analyze(c, dep, now - cfg["window_minutes"] * 60, now)
        slim = {k: v for k, v in m.items()}
        slim["features"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("unseen_examples",)} for k, v in m["features"].items()}
        c.execute("INSERT INTO monitor_snapshots(deployment_id,ts,window_start,window_end,n,health,metrics) VALUES (?,?,?,?,?,?,?)",
                  (dep_id, now, now - cfg["window_minutes"] * 60, now, m["n"], m["health"], json.dumps(slim, default=float)))
        c.execute("UPDATE deployments SET health=?, health_reasons=?, last_monitor_at=? WHERE id=?",
                  (m["health"], json.dumps(m["reasons"]), now, dep_id))
        if m["health"] != "no_data":
            sync_alerts(c, dep_id, m["reasons"])
        if dep["health"] != m["health"]:
            event(c, dep_id, "health", f"Health changed {dep['health']} → {m['health']}", {"reasons": [r["message"] for r in m["reasons"]]})
        retention = now - cfg["retention_days"] * 86400
        c.execute("DELETE FROM monitor_snapshots WHERE deployment_id=? AND ts<?", (dep_id, retention))
        c.execute("DELETE FROM predictions WHERE deployment_id=? AND ts<?", (dep_id, retention))
        return m


def sync_alerts(c, dep_id: str, reasons: list[dict]) -> None:
    """Open new alerts, refresh ongoing ones, auto-resolve those whose condition cleared."""
    now = time.time()
    open_ = {a["key"]: a for a in db.rows(c, "SELECT * FROM alerts WHERE deployment_id=? AND status!='resolved'", (dep_id,))}
    active = {r["key"]: r for r in reasons if r["severity"] in ("warning", "critical")}
    for key, r in active.items():
        if key in open_:
            c.execute("UPDATE alerts SET severity=?, message=?, value=?, updated_at=? WHERE id=?",
                      (r["severity"], r["message"], r.get("value"), now, open_[key]["id"]))
        else:
            c.execute("INSERT INTO alerts(deployment_id,key,kind,severity,message,status,created_at,updated_at,value) VALUES (?,?,?,?,?,?,?,?,?)",
                      (dep_id, key, r["kind"], r["severity"], r["message"], "open", now, now, r.get("value")))
            from ..notify import notify
            notify(f"alert_{'critical' if r['severity'] == 'critical' else 'warning'}", f"{dep_id}: {r['severity']}", r["message"],
                   f"/deployments/{dep_id}", r["severity"], key=f"alert:{dep_id}:{key}")
    for key, a in open_.items():
        if key not in active and not key.startswith("automation"):
            c.execute("UPDATE alerts SET status='resolved', resolved_at=?, updated_at=? WHERE id=?", (now, now, a["id"]))


def raise_alert(c, dep_id: str, key: str, kind: str, severity: str, message: str) -> None:
    from ..notify import notify
    notify("automation", f"{dep_id}: action needed", message, f"/deployments/{dep_id}", "warning", key=f"alert:{dep_id}:{key}")
    now = time.time()
    c.execute("INSERT INTO alerts(deployment_id,key,kind,severity,message,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
              (dep_id, key, kind, severity, message, "open", now, now))


def promote(c, dep_id: str, version_id: str, reason: str = "manual") -> None:
    dep = get_deployment(c, dep_id)
    v = db.one(c, "SELECT * FROM model_versions WHERE id=? AND model_name=?", (version_id, dep["model_name"]))
    if not v:
        raise HTTPException(400, "Version doesn't belong to this deployment's registered model")
    prev = dep["champion_version_id"]
    if prev == version_id:
        return
    if prev:
        set_stage(c, prev, "archived")
    set_stage(c, version_id, "production")
    chall = None if dep["challenger_version_id"] == version_id else dep["challenger_version_id"]
    c.execute("UPDATE deployments SET champion_version_id=?, challenger_version_id=?, challenger_mode=? WHERE id=?",
              (version_id, chall, dep["challenger_mode"] if chall else "none", dep_id))
    pv = db.one(c, "SELECT version FROM model_versions WHERE id=?", (prev,)) if prev else None
    event(c, dep_id, "promote", f"v{v['version']} promoted to production" + (f" (replacing v{pv['version']})" if pv else "") + f" — {reason}",
          {"from": prev, "to": version_id, "reason": reason})
    for a in db.rows(c, "SELECT id FROM alerts WHERE deployment_id=? AND key LIKE 'automation%' AND status!='resolved'", (dep_id,)):
        c.execute("UPDATE alerts SET status='resolved', resolved_at=? WHERE id=?", (time.time(), a["id"]))


def rollback(c, dep_id: str) -> str:
    ev = db.one(c, "SELECT * FROM deployment_events WHERE deployment_id=? AND kind='promote' ORDER BY ts DESC LIMIT 1", (dep_id,))
    if not ev or not ev["data"].get("from"):
        raise HTTPException(400, "Nothing to roll back to")
    promote(c, dep_id, ev["data"]["from"], reason="rollback")
    return ev["data"]["from"]
