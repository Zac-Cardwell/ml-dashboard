"""Runs one training/evaluation run (one seed of a session) in its own process.

    python -m mltrain.worker <run_id>

Launched by the scheduler; stdout/stderr go to data/runs/<run_id>/log.txt.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from . import config, db
from .plugins.registry import get_registry
from .training.context import RunContext
from .preprocessing import apply_saved_preprocessor  # noqa: F401  (re-exported)
from .training.utils import seed_everything


def main(run_id: str) -> int:
    conn = db.connect()
    run = db.one(conn, "SELECT * FROM runs WHERE id=?", (run_id,))
    session = db.one(conn, "SELECT * FROM sessions WHERE id=?", (run["session_id"],))
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    db.update_run(conn, run_id, status="running", started_at=run["started_at"] if run["resume"] else db.now(),
                  pid=os.getpid(), progress_text="loading", error=None)
    mode = session.get("mode") or "train"
    print(f"run {run_id} | session {session['id']} '{session['name']}' | {mode} | seed {run['seed']} | device {session['device']}"
          + (" | resume" if run["resume"] else ""), flush=True)

    init_path = None
    if run.get("init_path"):
        init_path = Path(run["init_path"])
    elif run.get("init_run_id"):
        init_path = config.RUNS_DIR / run["init_run_id"] / "model"
    if init_path is not None:
        if not (init_path / "meta.json").exists():
            raise_msg = f"Saved model not found at {init_path} (was it deleted?)"
            db.update_run(conn, run_id, status="failed", ended_at=db.now(), error=raise_msg, progress_text="failed")
            print(raise_msg, file=sys.stderr, flush=True)
            return 1
        print(f"starting from saved model {init_path}", flush=True)

    ctx = RunContext(run, session, run_dir, init_path=init_path)
    try:
        reg = get_registry()
        model = reg.get("model", session["model_id"])
        problem_plugin = reg.problem(session["problem_kind"], session["problem_id"])
        seed_everything(run["seed"])
        if session["problem_kind"] == "dataset":
            ctx.set_progress(0, "loading dataset")
            problem = problem_plugin.load(session["problem_params"], run["seed"])
            if init_path is not None:
                problem = apply_saved_preprocessor(problem, init_path)
            ctx.info = problem.info
            ctx.log_artifact("dataset_info", "json", {k: v for k, v in problem.info.items() if k != "feature_names"})
        else:
            problem = problem_plugin
        summary = (model.evaluate(problem, ctx) if mode == "evaluate" else model.train(problem, ctx)) or {}
        ctx.flush()
        status = "stopped" if ctx.should_stop() else "completed"
        if ctx.save_model and ctx.model_dir.exists():
            write_model_meta(ctx, session, run, summary, "final" if ctx.keep == "final" else "best")
            if ctx.final_dir.exists():
                fin = {k.removeprefix("final_model/"): v for k, v in summary.items() if k.startswith("final_model/")}
                fin.update({k: v for k, v in summary.items() if "/" not in k})
                write_model_meta(ctx, session, run, fin, "final", ctx.final_dir)
        else:
            shutil.rmtree(ctx.model_dir, ignore_errors=True)
            shutil.rmtree(ctx.final_dir, ignore_errors=True)
        if status == "completed" or not ctx.opt("keep_checkpoint", True):
            ctx.clear_checkpoint()
        extra = {"progress": 1.0} if status == "completed" else {}
        db.update_run(conn, run_id, status=status, ended_at=db.now(), progress_text=status, resume=0,
                      summary={k: v for k, v in summary.items() if v is not None}, **extra)
        print(f"{status}: {summary}", flush=True)
        return 0
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            ctx.flush()
        except Exception:
            pass
        status = "stopped" if ctx.should_stop() else "failed"   # crashes caused by a stop request are just stops
        db.update_run(conn, run_id, status=status, ended_at=db.now(), error=tb[-4000:], progress_text=status, resume=0)
        return 1


def write_model_meta(ctx: RunContext, session: dict, run: dict, summary: dict, variant: str = "best", target_dir: Path | None = None) -> None:
    """model/meta.json makes a saved model self-describing (used to fine-tune and evaluate it later)."""
    meta = {
        "run_id": run["id"], "session_id": session["id"], "session_name": session["name"], "seed": run["seed"],
        "model_id": session["model_id"], "framework": session["framework"], "model_params": session["model_params"],
        "problem_kind": session["problem_kind"], "problem_id": session["problem_id"], "problem_params": session["problem_params"],
        "task": session["task"], "modality": session["modality"],
        "info": {k: v for k, v in ctx.info.items() if k in ("n_features", "n_classes", "class_names", "input_shape", "feature_names",
                                                             "preprocessor", "raw_columns", "image_norm", "obs_space", "action_space", "window")},
        "summary": {k: v for k, v in summary.items() if isinstance(v, (int, float)) and not k.startswith("final_model/")},
        "variant": variant,
        "created_at": db.now(),
    }
    ((target_dir or ctx.model_dir) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
