"""Continuous training / evaluation (champion-challenger).

Triggers: schedule, input/prediction drift, live performance drop, or manual.
  1. retrain a challenger on fresh data (production feedback or an updated dataset) — a normal training session
  2. evaluate the current champion on the *same* data split — a normal evaluation session
  3. quality gate: challenger must beat the champion by >= min_improvement on the headline metric
  4. register the challenger as a new version; promote automatically or wait for approval
     (optionally run it in shadow first)
Scheduled re-evaluation re-tests the champion on an updated dataset and records the result.
"""
from __future__ import annotations

import json
import time

from .. import analysis, db
from .deployments import event, promote, raise_alert

DEFAULT_AUTOMATION = {
    "enabled": False,
    "triggers": ["drift", "performance"],    # also: "schedule"
    "retrain_every_hours": 0,                # schedule trigger (0 = off)
    "evaluate_every_hours": 0,               # scheduled re-test of the champion (0 = off)
    "cooldown_hours": 6,                     # min time between automatic retrains
    "data": {"source": "feedback", "feedback_days": 30, "min_labeled": 300,
             "problem_kind": None, "problem_id": None, "problem_params": {}},
    "min_improvement": 0.005,                # absolute gain on the headline metric required to promote
    "auto_promote": False,
    "shadow_first": False,                   # put a passing challenger in shadow instead of promoting
    "warm_start": False,                     # fine-tune from the champion instead of training from scratch
}


def automation_of(dep: dict) -> dict:
    a = {**DEFAULT_AUTOMATION, **(dep.get("automation") or {})}
    a["data"] = {**DEFAULT_AUTOMATION["data"], **((dep.get("automation") or {}).get("data") or {})}
    return a


def _problem(dep: dict, champ_meta: dict, auto: dict) -> dict:
    d = auto["data"]
    if d["source"] == "feedback":
        pid = {"text": "deployment.feedback_text"}.get(champ_meta["modality"], "deployment.feedback_regression"
                                                       if champ_meta["task"] == "regression" else "deployment.feedback")
        return {"problem_kind": "dataset", "problem_id": pid,
                "problem_params": {"deployment_id": dep["id"], "days": d["feedback_days"], "split_seed": 0, "until": time.time()}}
    if d["source"] == "live":
        # a live data source: train on its most recent window, weighting recent rows, test on the newest
        return {"problem_kind": "dataset", "problem_id": f"live.{d['source_id']}",
                "problem_params": {"lookback_hours": d.get("lookback_hours", 24 * 30), "weighting": d.get("weighting", "exponential"),
                                   "half_life_hours": d.get("half_life_hours", 24 * 7), "split": "time", "until": time.time()}}
    if d["source"] == "original":
        return {"problem_kind": champ_meta["problem_kind"], "problem_id": champ_meta["problem_id"],
                "problem_params": {**champ_meta["problem_params"], "split_seed": 0}}
    return {"problem_kind": d["problem_kind"], "problem_id": d["problem_id"], "problem_params": {**(d["problem_params"] or {}), "split_seed": 0}}


def labeled_count(c, dep_id: str, days: float) -> int:
    return c.execute("SELECT COUNT(*) FROM predictions WHERE deployment_id=? AND shadow=0 AND label IS NOT NULL AND ts>=?",
                     (dep_id, time.time() - days * 86400)).fetchone()[0]


def start(dep: dict, kind: str, trigger: str) -> str | None:
    """Launch an automation run (retrain or evaluate). Returns its id, or None if skipped."""
    from ..api.sessions import InitFrom, SessionCreate, create_session
    auto = automation_of(dep)
    with db.get_conn() as c:
        if db.one(c, "SELECT id FROM automation_runs WHERE deployment_id=? AND status='running'", (dep["id"],)):
            return None
        champ = db.one(c, "SELECT * FROM model_versions WHERE id=?", (dep["champion_version_id"],))
        if not champ:
            return None
        meta = json.loads(open(f"{champ['path']}/meta.json").read())
        if auto["data"]["source"] == "feedback":
            n = labeled_count(c, dep["id"], auto["data"]["feedback_days"])
            if n < auto["data"]["min_labeled"]:
                event(c, dep["id"], "automation", f"Skipped {kind} ({trigger}): only {n} labeled production examples "
                                                  f"(need {auto['data']['min_labeled']})")
                return None
        prob = _problem(dep, meta, auto)
        rid = db.new_id()
        tag = ["automation", dep["id"]]
        c.execute("BEGIN")
        try:
            champ_eval = create_session(c, SessionCreate(
                **prob, model_id=meta["model_id"], mode="evaluate", seeds=[0], tags=tag,
                init_from=InitFrom(version_id=champ["id"]), options={"snapshots": False, "eval_splits": "test"},
                name=f"[auto] {dep['id']} · champion v{champ['version']} on fresh data"))
            chall = None
            if kind == "retrain":
                chall = create_session(c, SessionCreate(
                    **prob, model_id=meta["model_id"], model_params=meta["model_params"], seeds=[0], tags=tag,
                    init_from=InitFrom(version_id=champ["id"]) if auto["warm_start"] else None,
                    options={"save_model": True, "snapshots": False},
                    name=f"[auto] {dep['id']} · challenger ({trigger})", notes=f"Automatic retrain triggered by {trigger}"))
            now = time.time()
            c.execute("INSERT INTO automation_runs(id,deployment_id,kind,trigger,status,created_at,updated_at,challenger_session_id,"
                      "champion_session_id,data) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (rid, dep["id"], kind, trigger, "running", now, now, chall, champ_eval,
                       json.dumps({"champion_version": champ["id"], "problem": prob})))
            event(c, dep["id"], "automation", f"{'Retraining a challenger' if kind == 'retrain' else 'Re-evaluating the champion'} ({trigger})",
                  {"run": rid})
            c.execute("COMMIT")
        except Exception as e:
            c.execute("ROLLBACK")
            with db.get_conn() as c2:
                event(c2, dep["id"], "automation", f"Could not start {kind}: {e}")
            return None
    return rid


def _headline(c, session_id: str, task: str, key: str | None = None) -> float | None:
    key = key or analysis.HEADLINE.get(task, (None, True))[0]
    runs = db.rows(c, "SELECT seed, status, summary FROM runs WHERE session_id=?", (session_id,))
    st = analysis.summary_stats([r for r in runs if r["status"] == "completed"])
    return st[key]["mean"] if key in st else None


def _session_state(c, sid: str | None) -> str:
    if not sid:
        return "completed"
    runs = db.rows(c, "SELECT status FROM runs WHERE session_id=?", (sid,))
    return analysis.session_status(runs)


def progress(run: dict) -> None:
    """Advance a running automation run once its sessions finish."""
    from .registry import register_version
    with db.get_conn() as c:
        dep = db.one(c, "SELECT * FROM deployments WHERE id=?", (run["deployment_id"],))
        states = [_session_state(c, run["champion_session_id"]), _session_state(c, run["challenger_session_id"])]
        if any(s in ("running", "queued", "stopping") for s in states):
            return
        if any(s not in ("completed",) for s in states):
            c.execute("UPDATE automation_runs SET status='failed', updated_at=?, message=? WHERE id=?",
                      (time.time(), f"session states: {states}", run["id"]))
            event(c, dep["id"], "automation", "Automation run failed (a session did not complete)", {"run": run["id"]})
            return
        champ_v = db.one(c, "SELECT * FROM model_versions WHERE id=?", (run["data"]["champion_version"],))
        task = champ_v["task"]
        key, hib = analysis.HEADLINE.get(task, (None, True))
        # binary classifiers are gated on AUC (threshold-free, matches live monitoring) when both sides report it
        if task == "classification" and _headline(c, run["champion_session_id"], task, "test/auc") is not None and \
                (run["kind"] == "evaluate" or _headline(c, run["challenger_session_id"], task, "test/auc") is not None):
            key = "test/auc"
        champ_score = _headline(c, run["champion_session_id"], task, key)

    if run["kind"] == "evaluate":
        with db.get_conn() as c:
            c.execute("UPDATE automation_runs SET status='completed', updated_at=?, decision=?, data=? WHERE id=?",
                      (time.time(), f"{key}={champ_score}", json.dumps({**run["data"], "champion_score": champ_score}), run["id"]))
            event(c, dep["id"], "evaluation", f"Scheduled re-test: champion v{champ_v['version']} {key} = {champ_score:.4f}"
                  if champ_score is not None else "Scheduled re-test finished", {"run": run["id"], "score": champ_score, "key": key})
        return

    with db.get_conn() as c:
        chall_score = _headline(c, run["challenger_session_id"], task, key)
    auto = automation_of(dep)
    gain = None if chall_score is None or champ_score is None else (chall_score - champ_score) * (1 if hib else -1)
    data = {**run["data"], "champion_score": champ_score, "challenger_score": chall_score, "gain": gain, "metric": key}
    if gain is None or gain < auto["min_improvement"]:
        with db.get_conn() as c:
            c.execute("UPDATE automation_runs SET status='rejected', updated_at=?, decision=?, data=? WHERE id=?",
                      (time.time(), f"challenger did not beat champion by {auto['min_improvement']}", json.dumps(data), run["id"]))
            event(c, dep["id"], "automation", f"Challenger rejected: {key} {_fmt(chall_score)} vs champion {_fmt(champ_score)} "
                  f"(needed +{auto['min_improvement']})", {"run": run["id"], **data})
        from ..notify import notify
        notify("automation", f"{dep['id']}: challenger rejected", f"{key} {_fmt(chall_score)} vs champion {_fmt(champ_score)} "
               f"(trigger: {run['trigger']})", f"/deployments/{dep['id']}", "info")
        return
    try:
        v = register_version(dep["model_name"], run["challenger_session_id"], None,
                             f"Auto-retrained ({run['trigger']}); {key} {_fmt(chall_score)} vs {_fmt(champ_score)}", "automation")
    except Exception as e:
        with db.get_conn() as c:
            c.execute("UPDATE automation_runs SET status='failed', updated_at=?, message=? WHERE id=?", (time.time(), str(e), run["id"]))
        return
    with db.get_conn() as c:
        msg = f"Challenger v{v['version']} beats champion: {key} {_fmt(chall_score)} vs {_fmt(champ_score)} (+{gain:.4f})"
        if auto["auto_promote"] and not auto["shadow_first"]:
            promote(c, dep["id"], v["id"], reason=f"auto-promoted after {run['trigger']} retrain")
            status = "promoted"
            from ..notify import notify
            notify("automation", f"{dep['id']}: v{v['version']} auto-promoted", msg, f"/deployments/{dep['id']}", "success")
        else:
            c.execute("UPDATE model_versions SET stage='staging' WHERE id=?", (v["id"],))
            if auto["shadow_first"]:
                c.execute("UPDATE deployments SET challenger_version_id=?, challenger_mode='shadow' WHERE id=?", (v["id"], dep["id"]))
                msg += " — running in shadow"
            raise_alert(c, dep["id"], f"automation:{run['id']}", "approval", "warning", msg + ". Approve to promote.")
            status = "awaiting_approval"
        c.execute("UPDATE automation_runs SET status=?, updated_at=?, challenger_version_id=?, decision=?, data=? WHERE id=?",
                  (status, time.time(), v["id"], msg, json.dumps(data), run["id"]))
        event(c, dep["id"], "automation", msg, {"run": run["id"], **data})


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def check_triggers(dep: dict, snapshot: dict | None) -> None:
    auto = automation_of(dep)
    if not auto["enabled"] or not dep["champion_version_id"]:
        return
    now = time.time()
    with db.get_conn() as c:
        last = db.one(c, "SELECT * FROM automation_runs WHERE deployment_id=? AND kind='retrain' ORDER BY created_at DESC LIMIT 1", (dep["id"],))
        last_eval = db.one(c, "SELECT * FROM automation_runs WHERE deployment_id=? AND kind='evaluate' ORDER BY created_at DESC LIMIT 1", (dep["id"],))
    since = now - (last["created_at"] if last else dep["created_at"])
    reasons = {r["kind"]: r for r in (snapshot or {}).get("reasons", [])}
    trigger = None
    if "schedule" in auto["triggers"] and auto["retrain_every_hours"] and since >= auto["retrain_every_hours"] * 3600:
        trigger = "schedule"
    elif since >= auto["cooldown_hours"] * 3600 or not last:
        if "performance" in auto["triggers"] and reasons.get("performance", {}).get("severity") == "critical":
            trigger = "performance"
        elif "drift" in auto["triggers"] and ({"feature_drift", "prediction_drift"} & set(reasons)):
            trigger = "drift"
    if trigger:
        start(dep, "retrain", trigger)
    if auto["evaluate_every_hours"] and now - (last_eval["created_at"] if last_eval else dep["created_at"]) >= auto["evaluate_every_hours"] * 3600:
        start(dep, "evaluate", "schedule")
