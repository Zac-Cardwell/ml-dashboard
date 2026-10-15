"""Aggregation across seeds, lineage, and parameter diffs. Used by the API and the JSON export."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from . import db

STATUS_ORDER = ["running", "stopping", "queued", "failed", "stopped", "completed"]

HEADLINE = {  # task -> (metric, higher_is_better)
    "classification": ("test/acc", True),
    "regression": ("test/rmse", False),
    "rl": ("final/mean_reward", True),
    "seq2seq": ("test/bleu", True),
    "generation": ("test/fid", False),
}
# tasks whose headline depends on what was measured (images: FID; text: perplexity; otherwise test loss)
HEADLINE_FALLBACK = {"generation": [("test/fid", False), ("test/perplexity", False), ("test/loss", False)],
                     "clustering": [("test/silhouette", True), ("test/ari", True)],
                     "anomaly_detection": [("test/auc", True), ("test/average_precision", True)],
                     "dimensionality_reduction": [("test/explained_variance", True), ("test/reconstruction_rmse", False)]}


def session_status(runs: list[dict]) -> str:
    st = {r["status"] for r in runs}
    for s in STATUS_ORDER[:3]:
        if s in st:
            return s
    if "failed" in st:
        return "failed" if st == {"failed"} else "partial"
    if "stopped" in st:
        return "stopped"
    return "completed" if st else "empty"


def summary_stats(runs: list[dict]) -> dict[str, dict]:
    """Per final-metric stats across seeds (mean/std/min/max/n + per-seed values)."""
    vals: dict[str, list] = defaultdict(list)
    for r in runs:
        for k, v in (r.get("summary") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals[k].append((r["seed"], float(v)))
    out = {}
    for k, sv in sorted(vals.items()):
        a = np.array([v for _, v in sv])
        out[k] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                  "min": float(a.min()), "max": float(a.max()), "n": len(a),
                  "per_seed": {str(s): v for s, v in sv}}
    return out


def headline(task: str | None, stats: dict) -> dict | None:
    if task in HEADLINE_FALLBACK:
        key, hib = next(((k, h) for k, h in HEADLINE_FALLBACK[task] if k in stats), (None, True))
        if key is None:
            return None
    elif task not in HEADLINE:
        return None
    else:
        key, hib = HEADLINE[task]
    if key not in stats:
        return None
    return {"key": key, "higher_is_better": hib, **{k: stats[key][k] for k in ("mean", "std", "n")}}


def load_series(c, run_ids: list[str], since_id: int = 0) -> tuple[dict, int]:
    """{run_id: {key: {"steps": [...], "values": [...]}}}, last metric id."""
    out: dict = {rid: defaultdict(lambda: {"steps": [], "values": []}) for rid in run_ids}
    last = since_id
    if not run_ids:
        return {}, last
    q = ",".join("?" * len(run_ids))
    for row in c.execute(f"SELECT id, run_id, step, key, value FROM metrics WHERE run_id IN ({q}) AND id > ? ORDER BY id",
                         (*run_ids, since_id)):
        s = out[row["run_id"]][row["key"]]
        s["steps"].append(row["step"])
        s["values"].append(row["value"])
        last = row["id"]
    return {rid: dict(v) for rid, v in out.items()}, last


def aggregate_series(per_run: dict, max_points: int = 300) -> dict:
    """Mean/std across runs for every metric key.

    Steps that line up exactly (epochs) are averaged directly; irregular steps (RL episodes)
    are linearly interpolated onto a shared grid over the overlapping step range.
    """
    by_key: dict[str, list] = defaultdict(list)
    for series in per_run.values():
        for k, s in series.items():
            if s["steps"]:
                by_key[k].append((np.asarray(s["steps"], float), np.asarray(s["values"], float)))
    out = {}
    for k, lst in by_key.items():
        step_sets = [tuple(st) for st, _ in lst]
        if len(set(step_sets)) == 1 and len(step_sets[0]) <= max_points:
            grid = np.asarray(step_sets[0])
            mat = np.stack([v for _, v in lst])
        else:
            lo = max(st.min() for st, _ in lst)
            hi = min(st.max() for st, _ in lst)
            if hi < lo:
                lo, hi = min(st.min() for st, _ in lst), max(st.max() for st, _ in lst)
            n = int(min(max_points, max(len(st) for st, _ in lst)))
            grid = np.linspace(lo, hi, max(n, 2))
            mat = np.stack([np.interp(grid, *_dedupe(st, v)) for st, v in lst])
        out[k] = {"steps": grid.round(3).tolist(), "mean": mat.mean(0).tolist(),
                  "std": (mat.std(0, ddof=1) if len(lst) > 1 else np.zeros(mat.shape[1])).tolist(), "n": len(lst)}
    return out


def _dedupe(steps, vals):
    """np.interp needs increasing x; average values that share a step."""
    order = np.argsort(steps, kind="stable")
    steps, vals = steps[order], vals[order]
    uniq, idx = np.unique(steps, return_inverse=True)
    if len(uniq) == len(steps):
        return steps, vals
    sums = np.bincount(idx, vals)
    return uniq, sums / np.bincount(idx)


def downsample(per_run: dict, max_points: int = 500) -> dict:
    out = {}
    for rid, series in per_run.items():
        out[rid] = {}
        for k, s in series.items():
            n = len(s["steps"])
            stride = max(1, n // max_points)
            out[rid][k] = {"steps": s["steps"][::stride], "values": s["values"][::stride]}
    return out


def param_diff(a: dict, b: dict) -> list[dict]:
    """Differences between two flat param dicts (b relative to a)."""
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append({"param": k, "from": a.get(k), "to": b.get(k)})
    return out


def session_config(s: dict) -> dict:
    """Flattened configuration for diffs: problem/model ids and all params."""
    d = {"problem": s["problem_id"], "model": s["model_id"], "seeds": s["seeds"], "device": s["device"]}
    d.update({f"problem.{k}": v for k, v in (s["problem_params"] or {}).items()})
    d.update({f"model.{k}": v for k, v in (s["model_params"] or {}).items()})
    return d


def lineage(c, session_id: str) -> dict:
    """The full family tree containing a session: walk to the root, then collect descendants."""
    cur = db.one(c, "SELECT id, parent_id FROM sessions WHERE id=?", (session_id,))
    seen = set()
    while cur and cur["parent_id"] and cur["parent_id"] not in seen:
        seen.add(cur["id"])
        parent = db.one(c, "SELECT id, parent_id FROM sessions WHERE id=?", (cur["parent_id"],))
        if parent is None:
            break
        cur = parent
    root = cur["id"] if cur else session_id
    nodes, frontier = {}, [root]
    while frontier:
        q = ",".join("?" * len(frontier))
        found = db.rows(c, f"SELECT * FROM sessions WHERE id IN ({q})", tuple(frontier))
        for s in found:
            nodes[s["id"]] = s
        kids = db.rows(c, f"SELECT id FROM sessions WHERE parent_id IN ({q})", tuple(frontier))
        frontier = [k["id"] for k in kids if k["id"] not in nodes]
    out_nodes = []
    for s in sorted(nodes.values(), key=lambda s: s["created_at"]):
        runs = db.rows(c, "SELECT seed, status, summary FROM runs WHERE session_id=?", (s["id"],))
        stats = summary_stats(runs)
        parent = nodes.get(s["parent_id"])
        out_nodes.append({
            "id": s["id"], "name": s["name"], "parent_id": s["parent_id"], "created_at": s["created_at"],
            "status": session_status(runs), "model_id": s["model_id"], "problem_id": s["problem_id"],
            "headline": headline(s["task"], stats),
            "changes": param_diff(session_config(parent), session_config(s)) if parent else [],
        })
    return {"root": root, "current": session_id, "nodes": out_nodes}


def session_brief(s: dict, runs: list[dict]) -> dict[str, Any]:
    stats = summary_stats([r for r in runs if r["status"] == "completed"])
    active = [r for r in runs if r["status"] in ("running", "stopping", "queued")]
    status = session_status(runs)
    if status == "queued" and s.get("scheduled_at") and s["scheduled_at"] > time.time():
        status = "scheduled"
    return {
        "id": s["id"], "name": s["name"], "created_at": s["created_at"], "status": status,
        "priority": s.get("priority", 0), "scheduled_at": s.get("scheduled_at"),
        "problem_kind": s["problem_kind"], "problem_id": s["problem_id"], "model_id": s["model_id"],
        "task": s["task"], "modality": s["modality"], "framework": s["framework"], "device": s["device"],
        "parent_id": s["parent_id"], "tags": s["tags"], "n_seeds": len(runs),
        "n_completed": sum(r["status"] == "completed" for r in runs),
        "progress": float(np.mean([r["progress"] or 0 for r in runs])) if runs else 0.0,
        "progress_text": next((r["progress_text"] for r in active if r["status"] == "running"), ""),
        "started_at": min((r["started_at"] for r in runs if r["started_at"]), default=None),
        "ended_at": max((r["ended_at"] for r in runs if r["ended_at"]), default=None) if not active else None,
        "headline": headline(s["task"], stats),
        "mode": s.get("mode") or "train",
        "group_id": s.get("group_id"), "group_kind": s.get("group_kind"), "group_params": s.get("group_params") or {},
        "init_session_id": (s.get("init_from") or {}).get("session_id"),
    }
