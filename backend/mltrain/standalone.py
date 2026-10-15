"""Run training sessions without the dashboard (used by exported portfolio repos).

LocalContext implements the same API as the dashboard's RunContext, but writes everything to files:

    runs/<experiment>/seed_<n>/
        metrics.csv        step,key,value  (every logged curve)
        summary.json       final metrics
        artifacts/*.json   confusion matrices, tables, ...
        media/             learning snapshots (videos, images) + media.jsonl
        model/             saved model (best weights, or final if only final was kept)
        model_final/       last-epoch weights (when both were kept)
"""
from __future__ import annotations

import csv
import json
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

from .training.context import RunContext, _json_default


class LocalContext(RunContext):
    def __init__(self, experiment: dict, seed: int, run_dir: Path, device: str, init_path: Path | None = None, quiet: bool = False):
        # deliberately not calling RunContext.__init__ (that one talks to the dashboard database)
        self.run_id = f"{experiment['name']}-seed{seed}"
        self.session_id = experiment["name"]
        self.seed = seed
        self.params = experiment["model_params"]
        self.problem_params = experiment["problem_params"]
        self.device = device
        self.mode = experiment.get("mode", "train")
        self.options = dict(experiment.get("options") or {})
        self.resume = False
        self.run_dir = run_dir
        self.model_dir = run_dir / "model"
        self.checkpoint_dir = run_dir / "checkpoint"
        self.media_dir = run_dir / "media"
        self.init_path = init_path
        self.info: dict = {}
        self.quiet = quiet
        self._buf: list[tuple] = []
        self._last_flush = time.time()
        self._last_frame = 0.0
        self._last_progress = 0.0
        self._stop = False
        run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = open(run_dir / "metrics.csv", "w", newline="")
        self._writer = csv.writer(self._metrics)
        self._writer.writerow(["step", "key", "value"])
        signal.signal(signal.SIGINT, self._on_sigterm)

    def log_artifact(self, name, kind, data):
        d = self.run_dir / "artifacts"
        d.mkdir(exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps({"kind": kind, "data": data}, default=_json_default))

    def log_media(self, step, name, kind, data=None, file=None):
        rec = {"step": int(step), "name": name, "kind": kind,
               "path": str(Path(file).relative_to(self.run_dir)) if file else None}
        self.media_dir.mkdir(parents=True, exist_ok=True)
        if data is not None and kind not in ("video", "image"):
            rec["data"] = data
        with open(self.media_dir / "media.jsonl", "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def live_frame(self, frame):
        pass

    def set_progress(self, frac, text=""):
        now = time.time()
        if self.quiet or (now - self._last_progress < 2 and frac < 1):
            return
        self._last_progress = now
        bar = "#" * int(30 * max(0.0, min(1.0, frac)))
        print(f"\r  [{bar:<30}] {frac * 100:5.1f}%  {text[:60]:<60}", end="", flush=True)

    def flush(self):
        for _, step, k, v, _t in self._buf:
            self._writer.writerow([step, k, repr(v)])
        self._buf.clear()
        self._metrics.flush()
        self._last_flush = time.time()

    def truncate_after(self, step):
        pass

    def print(self, *args):
        if not self.quiet:
            print(*args, flush=True)

    def close(self):
        self.flush()
        self._metrics.close()


def run_seed(experiment: dict, seed: int, out_dir: Path, device: str = "auto", quiet: bool = False) -> dict:
    """Train (or evaluate) one seed of an experiment; returns the final metrics."""
    from .plugins.registry import get_registry
    from .training.utils import seed_everything

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    run_dir = out_dir / f"seed_{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    init_path = Path(experiment["init_model"]) if experiment.get("init_model") else None
    ctx = LocalContext(experiment, seed, run_dir, device, init_path, quiet)
    reg = get_registry()
    if reg.errors and not quiet:
        for e in reg.errors:
            print(f"warning: could not load {e['file']}:\n{e['error']}", file=sys.stderr)
    model = reg.get("model", experiment["model_id"])
    problem = reg.problem(experiment["problem_kind"], experiment["problem_id"])
    seed_everything(seed)
    t0 = time.time()
    try:
        if experiment["problem_kind"] == "dataset":
            ctx.set_progress(0, "loading dataset")
            problem = problem.load(experiment["problem_params"], seed)
            if init_path is not None:
                from .preprocessing import apply_saved_preprocessor
                problem = apply_saved_preprocessor(problem, init_path)
            ctx.info = problem.info
        summary = (model.evaluate(problem, ctx) if ctx.mode == "evaluate" else model.train(problem, ctx)) or {}
    except KeyboardInterrupt:
        summary = {}
        ctx._stop = True
    except Exception:
        ctx.close()
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    ctx.close()
    if not quiet:
        print()
    summary = {k: v for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    summary.setdefault("wall_time_s", round(time.time() - t0, 2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    meta = {"experiment": experiment["name"], "seed": seed, "model_id": experiment["model_id"], "framework": experiment.get("framework"),
            "model_params": experiment["model_params"], "problem_kind": experiment["problem_kind"],
            "problem_id": experiment["problem_id"], "problem_params": experiment["problem_params"],
            "task": experiment.get("task"), "modality": experiment.get("modality"),
            "info": {k: v for k, v in ctx.info.items() if k in ("n_features", "n_classes", "class_names", "input_shape", "feature_names",
                                                                 "preprocessor", "raw_columns", "image_norm", "obs_space", "action_space")},
            "summary": {k: v for k, v in summary.items() if not k.startswith("final_model/")}}
    if ctx.save_model and ctx.model_dir.exists():
        (ctx.model_dir / "meta.json").write_text(json.dumps({**meta, "variant": "final" if ctx.keep == "final" else "best"}, indent=2, default=str))
        if ctx.final_dir.exists():
            fin = {k.removeprefix("final_model/"): v for k, v in summary.items() if k.startswith("final_model/")}
            (ctx.final_dir / "meta.json").write_text(json.dumps({**meta, "summary": fin, "variant": "final"}, indent=2, default=str))
    ctx.clear_checkpoint()
    if ctx._stop:
        raise KeyboardInterrupt
    return summary
