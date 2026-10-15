"""Saved models, hyperparameter sweeps and session groups (benchmarks / multi-model comparisons)."""
from __future__ import annotations

import itertools
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from .. import analysis, db, storage
from ..plugins.registry import get_registry
from .sessions import SessionCreate, create_session, runs_by_session
from .state import protected_router

router = protected_router()
MAX_GROUP = 100


# ===================================================================== saved models
@router.get("/models/saved")
def saved_models(task: str | None = None, modality: str | None = None, model_id: str | None = None):
    """Trained models kept on disk, grouped by session (newest first). Filter by what they can run on."""
    reg = get_registry()
    with db.get_conn() as c:
        q, args = "SELECT * FROM sessions WHERE mode='train'", []
        for col, val in (("task", task), ("modality", modality), ("model_id", model_id)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        sessions = db.rows(c, q + " ORDER BY created_at DESC", tuple(args))
        rb = runs_by_session(c, [s["id"] for s in sessions])
    out = []
    for s in sessions:
        runs = []
        for r in rb.get(s["id"], []):
            st = storage.run_storage(r["id"])
            if st["has_model"]:
                runs.append({"run_id": r["id"], "seed": r["seed"], "bytes": st["model_bytes"] + st["final_bytes"], "summary": r["summary"],
                             "has_final": st["has_final"]})
        if not runs:
            continue
        model = reg.plugins["model"].get(s["model_id"])
        out.append({"session_id": s["id"], "session_name": s["name"], "created_at": s["created_at"],
                    "model_id": s["model_id"], "model_name": model.name if model else s["model_id"],
                    "problem_id": s["problem_id"], "task": s["task"], "modality": s["modality"], "framework": s["framework"],
                    "model_params": s["model_params"], "problem_params": s["problem_params"],
                    "headline": analysis.headline(s["task"], analysis.summary_stats([{**r, "seed": r["seed"]} for r in runs])),
                    "runs": runs, "bytes": sum(r["bytes"] for r in runs), "has_final": any(r["has_final"] for r in runs)})
    return out


@router.get("/storage")
def storage_usage():
    """Disk used per session (models, checkpoints, media) — for cleaning up."""
    with db.get_conn() as c:
        sessions = db.rows(c, "SELECT id, name, created_at FROM sessions ORDER BY created_at DESC")
        rb = runs_by_session(c, [s["id"] for s in sessions])
    out = []
    for s in sessions:
        tot = {"model_bytes": 0, "checkpoint_bytes": 0, "media_bytes": 0, "total_bytes": 0}
        for r in rb.get(s["id"], []):
            st = storage.run_storage(r["id"])
            for k in tot:
                tot[k] += st[k] + (st["final_bytes"] if k == "model_bytes" else 0)
        out.append({**s, **tot})
    return out


# ===================================================================== groups
class GroupCreate(BaseModel):
    kind: str = "benchmark"                       # benchmark | compare | sweep
    sessions: list[SessionCreate]
    group_params: list[dict[str, Any]] | None = None   # per-session labels (optional)


def _create_group(kind: str, bodies: list[SessionCreate], labels: list[dict] | None) -> dict:
    if not bodies:
        raise HTTPException(400, "No sessions to create")
    if len(bodies) > MAX_GROUP:
        raise HTTPException(400, f"At most {MAX_GROUP} sessions per group ({len(bodies)} requested)")
    gid = "g" + db.new_id()
    ids = []
    with db.get_conn() as c:
        c.execute("BEGIN")
        try:
            for i, b in enumerate(bodies):
                ids.append(create_session(c, b, gid, kind, (labels or [None] * len(bodies))[i]))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
    return {"group_id": gid, "session_ids": ids}


@router.post("/groups")
def create_group(body: GroupCreate):
    """Create several sessions at once — e.g. evaluate many saved models on one dataset, or train several model types."""
    return _create_group(body.kind, body.sessions, body.group_params)


class SweepCreate(BaseModel):
    base: SessionCreate
    grid: dict[str, list[Any]]   # keys like "model.lr", "problem.val_fraction", "options.snapshot_count"


@router.post("/sweeps")
def create_sweep(body: SweepCreate):
    """Grid search: one session per combination of the listed values (each with the base's seeds)."""
    if not body.grid:
        raise HTTPException(400, "Sweep grid is empty")
    for k, vals in body.grid.items():
        if k.split(".", 1)[0] not in ("model", "problem", "options") or "." not in k:
            raise HTTPException(400, f"Grid key '{k}' must look like model.<param>, problem.<param> or options.<param>")
        if not vals:
            raise HTTPException(400, f"No values for '{k}'")
    keys = list(body.grid)
    combos = list(itertools.product(*(body.grid[k] for k in keys)))
    bodies, labels = [], []
    for combo in combos:
        b = body.base.model_copy(deep=True)
        label = {}
        for k, v in zip(keys, combo):
            section, name = k.split(".", 1)
            getattr(b, f"{section}_params" if section != "options" else "options")[name] = v
            label[k] = v
        bodies.append(b)
        labels.append(label)
    return _create_group("sweep", bodies, labels)


@router.get("/groups/{gid}")
def get_group(gid: str):
    with db.get_conn() as c:
        sessions = db.rows(c, "SELECT * FROM sessions WHERE group_id=? ORDER BY created_at", (gid,))
        if not sessions:
            raise HTTPException(404, "group not found")
        rb = runs_by_session(c, [s["id"] for s in sessions])
    members = []
    for s in sessions:
        runs = rb.get(s["id"], [])
        members.append({**analysis.session_brief(s, runs), "group_params": s.get("group_params") or {},
                        "model_params": s["model_params"],
                        "final_metrics": {k: {"mean": v["mean"], "std": v["std"], "n": v["n"]}
                                          for k, v in analysis.summary_stats([r for r in runs if r["status"] == "completed"]).items()}})
    swept = sorted({k for m in members for k in m["group_params"]})
    return {"group_id": gid, "kind": sessions[0]["group_kind"], "swept": swept, "sessions": members}
