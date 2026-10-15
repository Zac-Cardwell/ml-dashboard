"""Recurring training: run a session template every N hours, daily at HH:MM, or weekly.

Every run is a normal session (group_kind='schedule', so all runs of a schedule can be compared over time).
For live sources the data snapshot is frozen at launch (`until` = launch time). Optionally each finished run is
registered in the model registry under a fixed name, and a notification is sent.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

from fastapi import HTTPException

from .. import analysis, db

log = logging.getLogger("mltrain.schedules")
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def next_run(cadence: dict, after: float) -> float:
    kind = cadence.get("kind", "interval")
    if kind == "interval":
        hours = float(cadence.get("every_hours") or 24)
        if hours < 0.25:
            raise HTTPException(400, "Interval must be at least 0.25 hours")
        return after + hours * 3600
    hh, mm = (int(x) for x in str(cadence.get("at", "02:00")).split(":"))
    base = dt.datetime.fromtimestamp(after).astimezone()          # local time — "daily at 02:00" means your 2 am
    cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if kind == "daily":
        if cand.timestamp() <= after:
            cand += dt.timedelta(days=1)
        return cand.timestamp()
    if kind == "weekly":
        wd = WEEKDAYS.index(cadence.get("weekday", "mon"))
        cand += dt.timedelta(days=(wd - cand.weekday()) % 7)
        if cand.timestamp() <= after:
            cand += dt.timedelta(days=7)
        return cand.timestamp()
    raise HTTPException(400, "cadence kind must be interval, daily or weekly")


def describe(cadence: dict) -> str:
    k = cadence.get("kind")
    if k == "interval":
        return f"every {cadence.get('every_hours')} h"
    if k == "daily":
        return f"daily at {cadence.get('at')}"
    return f"every {cadence.get('weekday', 'mon').capitalize()} at {cadence.get('at')}"


def create(name: str, template: dict, cadence: dict, register_as: str | None, start_now: bool = False) -> str:
    from ..api.sessions import SessionCreate
    SessionCreate(**template)   # validate shape
    sid = "sch" + db.new_id()
    first = time.time() if start_now else next_run(cadence, time.time())
    with db.get_conn() as c:
        c.execute("INSERT INTO training_schedules(id,name,enabled,template,cadence,register_as,next_run_at,created_at) VALUES (?,?,?,?,?,?,?,?)",
                  (sid, name, 1, json.dumps(template), json.dumps(cadence), register_as or None, first, time.time()))
    return sid


def launch(s: dict) -> str:
    from ..api.sessions import SessionCreate, create_session
    t = dict(s["template"])
    now = time.time()
    if str(t.get("problem_id", "")).startswith(("live.", "deployment.")):
        t["problem_params"] = {**(t.get("problem_params") or {}), "until": now}   # freeze the data snapshot
    t["name"] = f"{s['name']} · {dt.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M')}"
    t["tags"] = list(dict.fromkeys([*(t.get("tags") or []), "schedule"]))
    t.pop("scheduled_at", None)
    with db.get_conn() as c:
        c.execute("BEGIN")
        session_id = create_session(c, SessionCreate(**t), s["id"], "schedule", {"run_at": now})
        c.execute("INSERT INTO schedule_runs(schedule_id,session_id,created_at) VALUES (?,?,?)", (s["id"], session_id, now))
        c.execute("UPDATE training_schedules SET last_run_at=?, last_session_id=?, next_run_at=? WHERE id=?",
                  (now, session_id, next_run(s["cadence"], now), s["id"]))
        c.execute("COMMIT")
    log.info("schedule %s launched session %s", s["id"], session_id)
    return session_id


def tick() -> None:
    now = time.time()
    with db.get_conn() as c:
        due = db.rows(c, "SELECT * FROM training_schedules WHERE enabled=1 AND next_run_at<=?", (now,))
        pending = db.rows(c, "SELECT r.*, s.register_as, s.name AS schedule_name FROM schedule_runs r "
                             "JOIN training_schedules s ON s.id=r.schedule_id WHERE r.status='running'")
    for s in due:
        with db.get_conn() as c:
            busy = db.one(c, "SELECT r.session_id FROM schedule_runs r JOIN runs x ON x.session_id=r.session_id "
                             "WHERE r.schedule_id=? AND x.status IN ('queued','running','stopping') LIMIT 1", (s["id"],))
        if busy:   # previous run still going: skip this slot rather than piling up
            with db.get_conn() as c:
                c.execute("UPDATE training_schedules SET next_run_at=? WHERE id=?", (next_run(s["cadence"], now), s["id"]))
            continue
        try:
            launch(s)
        except Exception as e:
            log.exception("schedule %s failed to launch", s["id"])
            from ..notify import notify
            notify("schedule", f"Schedule '{s['name']}' could not start", str(e)[:500], "/data?tab=schedules", "critical",
                   key=f"schedule-fail:{s['id']}")
            with db.get_conn() as c:
                c.execute("UPDATE training_schedules SET next_run_at=? WHERE id=?", (next_run(s["cadence"], now), s["id"]))
    for r in pending:
        _finish_run(r)


def _finish_run(r: dict) -> None:
    from ..notify import notify
    with db.get_conn() as c:
        runs = db.rows(c, "SELECT * FROM runs WHERE session_id=?", (r["session_id"],))
        sess = db.one(c, "SELECT * FROM sessions WHERE id=?", (r["session_id"],))
    st = analysis.session_status(runs)
    if st in ("running", "queued", "stopping", "scheduled"):
        return
    version = None
    msg = ""
    if st == "completed":
        brief = analysis.session_brief(sess, runs)
        h = brief["headline"]
        msg = f"{h['key']} = {h['mean']:.4g}" if h else "completed"
        if r["register_as"]:
            try:
                from ..mlops.registry import register_version
                v = register_version(r["register_as"], r["session_id"], None, f"Scheduled run of '{r['schedule_name']}'", "schedule")
                version = v["id"]
                msg += f" · registered as {r['register_as']} v{v['version']}"
            except Exception as e:
                msg += f" · registration failed: {e}"
    with db.get_conn() as c:
        c.execute("UPDATE schedule_runs SET status=?, registered_version=? WHERE schedule_id=? AND session_id=?",
                  (st, version, r["schedule_id"], r["session_id"]))
    if st != "completed":
        err = next((x["error"] for x in runs if x["error"]), "") or ""
        lines = err.strip().splitlines()
        msg = lines[-1][:500] if lines else f"Run ended as '{st}'"
    notify("schedule", f"{'✓' if st == 'completed' else '✗'} Scheduled training '{r['schedule_name']}' {st}", msg,
           f"/sessions/{r['session_id']}", "success" if st == "completed" else "critical")
