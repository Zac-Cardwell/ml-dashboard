"""Watches for finished sessions / groups and sends notifications (one message per session, one per group)."""
from __future__ import annotations

from . import analysis, db
from .api.sessions import runs_by_session
from .notify import notify

TERMINAL = {"completed", "failed", "partial", "stopped", "empty"}


def mark_existing_notified() -> None:
    """On startup: don't notify about sessions that finished before notifications existed / while offline."""
    with db.get_conn() as c:
        c.execute("UPDATE sessions SET notified=1 WHERE notified=0 AND id NOT IN "
                  "(SELECT session_id FROM runs WHERE status IN ('queued','running','stopping'))")


def _fmt_headline(h: dict | None) -> str:
    if not h:
        return ""
    v = h["mean"]
    s = f"{v:.2%}" if h["key"].split("/")[-1] in ("acc", "f1", "precision", "recall") and v <= 1 else f"{v:.4g}"
    return f"{h['key']} = {s}" + (f" ± {h['std']:.3g} over {h['n']} seeds" if h["n"] > 1 else "")


def tick() -> None:
    with db.get_conn() as c:
        pending = db.rows(c, "SELECT * FROM sessions WHERE notified=0")
        if not pending:
            return
        rb = runs_by_session(c, [s["id"] for s in pending])
    done_groups: set[str] = set()
    for s in pending:
        runs = rb.get(s["id"], [])
        status = analysis.session_status(runs)
        if status not in TERMINAL:
            continue
        with db.get_conn() as c:
            c.execute("UPDATE sessions SET notified=1 WHERE id=?", (s["id"],))
        if "automation" in (s["tags"] or []) or s.get("group_kind") in ("search", "schedule"):
            continue            # automation, searches and schedules send their own summaries
        if s.get("group_id"):
            done_groups.add(s["group_id"])
            continue
        brief = analysis.session_brief(s, runs)
        if status == "completed":
            notify("session_completed", f"✓ {s['name']} finished", _fmt_headline(brief["headline"]) or "All seeds completed.",
                   f"/sessions/{s['id']}", "success")
        elif status in ("failed", "partial"):
            err = next((r["error"] for r in runs if r["error"]), "") or ""
            notify("session_failed", f"✗ {s['name']} {'failed' if status == 'failed' else 'partly failed'}",
                   (err.strip().splitlines() or ["(no error message)"])[-1][:500], f"/sessions/{s['id']}", "critical")
    for gid in done_groups:
        _group_done(gid)


def _group_done(gid: str) -> None:
    with db.get_conn() as c:
        members = db.rows(c, "SELECT * FROM sessions WHERE group_id=?", (gid,))
        rb = runs_by_session(c, [m["id"] for m in members])
    statuses = [analysis.session_status(rb.get(m["id"], [])) for m in members]
    if any(st not in TERMINAL for st in statuses):
        return
    briefs = [analysis.session_brief(m, rb.get(m["id"], [])) for m in members]
    scored = [b for b in briefs if b["headline"]]
    best = None
    if scored:
        hib = scored[0]["headline"]["higher_is_better"]
        best = (max if hib else min)(scored, key=lambda b: b["headline"]["mean"])
    kind = members[0].get("group_kind") or "group"
    failed = sum(st in ("failed", "partial") for st in statuses)
    msg = f"{len(members)} sessions, {failed} failed." + (f"\nBest: {best['name']} — {_fmt_headline(best['headline'])}" if best else "")
    notify("group_completed", f"{kind.capitalize()} finished", msg, f"/compare?group={gid}", "success" if not failed else "warning",
           key=f"group:{gid}")
