"""Portfolio export: turn one or more training sessions into a self-contained, GitHub-ready repository.

The repo contains the experiment configs, the original results (metrics, plots, README write-up), a trimmed copy of
the training code (no dashboard, no database) and `train.py`, so anyone can clone it and re-run every experiment:

    pip install -r requirements.txt
    python train.py                 # re-train everything with the original seeds
    python train.py --list          # experiments in this repo
    python train.py -e <name> --seeds 0 1 2 --device cpu

Nothing is published by building an export. Pushing to GitHub is a separate, explicit action (`push_export`).
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from . import config, db, report, storage

EXPORTS_DIR = config.DATA_DIR / "exports"
PKG = Path(__file__).resolve().parent
GH_FILE_LIMIT = 95 * 1024 * 1024          # GitHub rejects files > 100 MB
SNAPSHOT_LIMIT = 50 * 1024 * 1024
VIDEO_LIMIT = 8 * 1024 * 1024

# training code copied into the export (relative to the mltrain package)
PKG_FILES = ["__init__.py", "preprocessing.py", "report.py", "standalone.py", "plugins", "training", "builtin", "generative",
             "mlops/predictors.py", "mlops/features.py"]
CONFIG_PY = '''"""Paths (standalone version: everything lives inside this repository)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MLTRAIN_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("MLTRAIN_DATA", ROOT / "data"))
RUNS_DIR = ROOT / "runs"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"            # downloaded datasets / pretrained weights (git-ignored)
USER_PLUGINS_DIR = Path(os.environ.get("MLTRAIN_PLUGINS", ROOT / "plugins"))
'''

REQ_BASE = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib"]
REQ_BY_FRAMEWORK = {
    "torch": ["torch", "torchvision"],
    "sb3": ["torch", "stable-baselines3", "sb3-contrib", "gymnasium", "opencv-python-headless", "imageio", "imageio-ffmpeg"],
    "xgboost": ["xgboost"], "lightgbm": ["lightgbm"],
    "hf": ["torch", "transformers", "datasets", "sacrebleu", "rouge-score", "accelerate", "sentencepiece"],
}
REQ_BY_PROBLEM = [("vision.", ["torch", "torchvision", "pillow"]), ("hf.", ["datasets"]),
                  ("gym.LunarLander", ["gymnasium[box2d]"]), ("gym.BipedalWalker", ["gymnasium[box2d]"]), ("gym.CarRacing", ["gymnasium[box2d]"]),
                  ("gym.", ["gymnasium[classic-control]"]), ("atari", ["gymnasium[atari]", "ale-py"])]


# ============================================================================= helpers
def slug(s: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n].strip("-") or "experiment"


def _session(c, sid: str) -> dict:
    s = db.one(c, "SELECT * FROM sessions WHERE id=?", (sid,))
    if not s:
        raise ValueError(f"session {sid} not found")
    s["runs"] = db.rows(c, "SELECT * FROM runs WHERE session_id=? ORDER BY seed", (sid,))
    return s


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


class Export:
    def __init__(self, title: str, out: Path):
        self.title = title
        self.out = out
        self.warnings: list[str] = []
        self.frameworks: set[str] = set()
        self.problem_ids: set[str] = set()
        self.plugin_files: set[Path] = set()
        self.snapshots: dict[str, dict] = {}

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ============================================================================= build
def build_export(session_ids: list[str], title: str, description: str = "", author: str = "", email: str = "",
                 license: str = "MIT", include_models: str = "none", include_videos: bool = True,
                 git_init: bool = True, overrides: dict | None = None) -> dict:
    """overrides: {session_id: {"title": ..., "description": ...}} — portfolio-friendly names without renaming the sessions."""
    """Build data/exports/<slug>/ from the given sessions; returns the export's manifest."""
    from .plugins.registry import get_registry
    if not session_ids:
        raise ValueError("Choose at least one session")
    if include_models not in ("none", "best", "both"):
        raise ValueError("include_models must be none, best or both")
    name = slug(title or "ml-experiments", 60)
    out = EXPORTS_DIR / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ex = Export(title or name, out)
    reg = get_registry()
    t0 = time.time()

    with db.get_conn() as c:
        sessions = [_session(c, sid) for sid in dict.fromkeys(session_ids)]
    sessions.sort(key=lambda s: s["created_at"])
    for s in sessions:
        o = (overrides or {}).get(s["id"]) or {}
        if (o.get("title") or "").strip():
            s["name"] = o["title"].strip()
        if o.get("description") is not None:
            s["notes"] = o["description"].strip()
    names: dict[str, str] = {}
    for s in sessions:
        base = slug(s["name"])
        n, k = base, 2
        while n in names.values():
            n, k = f"{base}-{k}", k + 1
        names[s["id"]] = n

    experiments = []
    for s in sessions:
        done = [r for r in s["runs"] if r["status"] == "completed"]
        if not done:
            ex.warn(f"'{s['name']}' has no completed seeds — exported as config only (no results).")
        experiments.append(_export_session(ex, reg, s, names, done, include_models, include_videos))

    _write_package(ex)
    _write_requirements(ex)
    if ex.snapshots:
        (out / "data" / "snapshots").mkdir(parents=True, exist_ok=True)
        (out / "data" / "snapshots" / "index.json").write_text(json.dumps(ex.snapshots, indent=2))
    (out / "experiments" / "index.json").write_text(json.dumps([e["name"] for e in experiments], indent=2))
    (out / "train.py").write_text(TRAIN_PY)
    (out / "predict.py").write_text(PREDICT_PY)
    (out / ".gitignore").write_text(GITIGNORE)
    if license and license != "none":
        (out / "LICENSE").write_text(_license(license, author or "the author"))
    figs = _comparison_figure(ex, experiments)
    (out / "README.md").write_text(_readme(ex, experiments, description, figs, include_models, license))
    _scrub(ex)
    manifest = {"name": name, "title": ex.title, "created_at": time.time(), "sessions": [s["id"] for s in sessions],
                "experiments": [e["name"] for e in experiments], "warnings": ex.warnings, "include_models": include_models,
                "seconds": round(time.time() - t0, 1)}
    if git_init and shutil.which("git"):
        manifest["git"] = _git_init(out, author, email)
    manifest["files"], manifest["bytes"] = _tree_stats(out)
    (EXPORTS_DIR / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _export_session(ex: Export, reg, s: dict, names: dict, done: list[dict], include_models: str, include_videos: bool) -> dict:
    name = names[s["id"]]
    out = ex.out
    ex.frameworks.add(s["framework"])
    ex.problem_ids.add(s["problem_id"])
    model = reg.plugins["model"].get(s["model_id"])
    if model is None:
        raise ValueError(f"model plugin '{s['model_id']}' is not loaded; cannot export '{s['name']}'")
    if model.source not in ("builtin", "upload", "live") and not str(model.source).startswith("examples"):
        _add_plugin_file(ex, model.source, "models")

    problem_id, problem_params = s["problem_id"], dict(s["problem_params"])
    data_note = None
    if s["problem_kind"] == "dataset":
        problem_id, problem_params, data_note = _portable_dataset(ex, reg, s, name)
    else:
        env = reg.plugins["env"].get(s["problem_id"])
        if env is not None and env.source not in ("builtin",):
            _add_plugin_file(ex, env.source, "envs")
        if getattr(env, "export_note", None):          # e.g. an external simulator the repo can't include
            data_note = env.export_note
            ex.warn(f"'{s['name']}': {env.export_note}")

    options = {k: v for k, v in (s["options"] or {}).items() if k not in ("live_preview", "show_gui")}
    init = None
    if s.get("init_from"):
        src = s["init_from"].get("session_id")
        if src in names and not s["init_from"].get("version_id"):
            init = {"experiment": names[src], "variant": s["init_from"].get("variant", "best"),
                    "strategy": "match_seed" if s["init_from"].get("strategy") in ("match_seed", None) else "best"}
        elif s["mode"] == "evaluate":
            ex.warn(f"'{s['name']}' evaluates a model from a session that isn't in this export; include that session too, "
                    "or it will evaluate untrained weights.")
        else:
            ex.warn(f"'{s['name']}' was fine-tuned from a model outside this export; the export trains it from its initial weights.")
    cfg = {"name": name, "title": s["name"], "description": s.get("notes") or "", "tags": s.get("tags") or [],
           "mode": s["mode"], "model_id": s["model_id"], "model_name": model.name, "framework": s["framework"],
           "model_params": s["model_params"], "problem_kind": s["problem_kind"], "problem_id": problem_id,
           "original_problem_id": s["problem_id"] if problem_id != s["problem_id"] else None,
           "problem_params": problem_params, "task": s["task"], "modality": s["modality"], "seeds": s["seeds"],
           "options": options, "init_from": init, "data_note": data_note,
           "trained_on": {"date": _dt.datetime.fromtimestamp(s["created_at"]).strftime("%Y-%m-%d"), "device": s["device"],
                          "versions": (s.get("env_info") or {}).get("versions", {})},
           "parent": names.get(s.get("parent_id")) if s.get("parent_id") else None}
    (out / "experiments").mkdir(exist_ok=True)
    (out / "experiments" / f"{name}.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ---- original results
    res = out / "results" / name
    with db.get_conn() as c:
        for r in done:
            d = res / f"seed_{r['seed']}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "metrics.csv", "w") as f:
                f.write("step,key,value\n")
                for m in c.execute("SELECT step, key, value FROM metrics WHERE run_id=? ORDER BY id", (r["id"],)):
                    f.write(f"{m['step']},{m['key']},{m['value']!r}\n")
            (d / "summary.json").write_text(json.dumps(r["summary"] or {}, indent=2))
            arts = db.rows(c, "SELECT name, kind, data FROM artifacts WHERE run_id=?", (r["id"],))
            if arts:
                (d / "artifacts").mkdir(exist_ok=True)
                for a in arts:
                    if a["name"] == "dataset_info":
                        continue
                    (d / "artifacts" / f"{a['name']}.json").write_text(json.dumps({"kind": a["kind"], "data": a["data"]}))
        media = db.rows(c, "SELECT run_id, step, name, kind, path FROM media WHERE run_id IN (%s) AND kind='video' ORDER BY step DESC"
                        % ",".join("?" * len(done)), tuple(r["id"] for r in done)) if done else []

    series, summaries, artifacts = report.read_runs(res) if done else ({}, {}, {})
    stats = report.summary_stats(summaries)
    figs = {}
    if series:
        p = report.plot_curves(series, res / "learning_curves.png", s["name"])
        if p:
            figs["curves"] = p
    best_seed = _best_seed(s["task"], summaries)
    if best_seed is not None:
        cm = artifacts.get(best_seed, {}).get("confusion_matrix_test")
        if cm:
            p = report.plot_confusion(cm, res / "confusion_matrix.png", f"Confusion matrix (test, seed {best_seed})")
            if p:
                figs["confusion"] = p
    if include_videos and media:
        best_run = next((r["id"] for r in done if r["seed"] == best_seed), done[0]["id"])
        vid = next((m for m in media if m["run_id"] == best_run), media[0])
        src = storage.run_dir(vid["run_id"]) / vid["path"]
        if src.exists() and src.stat().st_size <= VIDEO_LIMIT:
            shutil.copy2(src, res / f"episode{src.suffix}")
            figs["video"] = res / f"episode{src.suffix}"

    # ---- trained weights
    if include_models != "none":
        for r in done:
            for variant in (("best", "final") if include_models == "both" else ("best",)):
                src = storage.model_dir(r["id"], variant)
                if not (src / "meta.json").exists():
                    continue
                dst = out / "models" / name / f"seed_{r['seed']}" / ("model" if variant == "best" else "model_final")
                big = [p for p in src.rglob("*") if p.is_file() and p.stat().st_size > GH_FILE_LIMIT]
                if big:
                    ex.warn(f"Skipped the {variant} model of '{s['name']}' seed {r['seed']}: {big[0].name} is over GitHub's 100 MB file limit.")
                    continue
                shutil.copytree(src, dst)
                _clean_meta(dst / "meta.json", name)
    return {"name": name, "cfg": cfg, "session": s, "stats": stats, "figs": figs, "n_seeds": len(done),
            "best_seed": best_seed, "headline": report.headline(s["task"], stats)}


def _best_seed(task: str, summaries: dict[int, dict]) -> int | None:
    if not summaries:
        return None
    key, hib = report.HEADLINE.get(task, (None, True))
    scored = {sd: v[key] for sd, v in summaries.items() if key and isinstance(v.get(key), (int, float))}
    if not scored:
        return min(summaries)
    return (max if hib else min)(scored, key=scored.get)


def _clean_meta(p: Path, exp: str) -> None:
    """Saved-model meta.json without dashboard ids."""
    m = json.loads(p.read_text())
    for k in ("run_id", "session_id", "session_name"):
        m.pop(k, None)
    m["experiment"] = exp
    p.write_text(json.dumps(m, indent=2, default=str))


def _add_plugin_file(ex: Export, source: str, kind_dir: str) -> None:
    src = Path(source)
    if not src.is_absolute():
        src = config.ROOT / src
    if src.exists() and src.suffix == ".py":
        dst = ex.out / "plugins" / kind_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ex.plugin_files.add(src)


# ----------------------------------------------------------------------------- datasets
def _portable_dataset(ex: Export, reg, s: dict, name: str) -> tuple[str, dict, str | None]:
    pid, params = s["problem_id"], dict(s["problem_params"])
    plugin = reg.plugins["dataset"].get(pid)
    if plugin is None:
        ex.warn(f"Dataset '{pid}' of '{s['name']}' is no longer available; its config is exported as-is.")
        return pid, params, None
    mod = type(plugin).__module__
    if pid.startswith("csv.path."):
        src = Path(params.get("path", "")).expanduser()
        if src.exists() and src.stat().st_size <= SNAPSHOT_LIMIT:
            dst = ex.out / "data" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            params["path"] = f"data/{src.name}"
            return pid, params, f"CSV file included as `data/{src.name}`."
        ex.warn(f"The CSV of '{s['name']}' ({src.name}) is missing or larger than 50 MB and was not included; "
                "put it at the path in the experiment config.")
        params["path"] = f"data/{src.name}"
        return pid, params, f"Needs `data/{src.name}` (not included)."
    if pid.startswith("upload."):
        meta = config.UPLOADS_DIR / "datasets" / f"{pid.split('.', 1)[1]}.json"
        csv = meta.with_suffix(".csv")
        if csv.exists() and csv.stat().st_size <= SNAPSHOT_LIMIT:
            d = ex.out / "data" / "uploads" / "datasets"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta, d / meta.name)
            shutil.copy2(csv, d / csv.name)
            return pid, params, f"Uploaded CSV included under `data/uploads/datasets/{csv.name}`."
        ex.warn(f"Uploaded CSV of '{s['name']}' is larger than 50 MB and was not included.")
        return pid, params, "Uploaded CSV not included (too large)."
    if plugin.source not in ("builtin",) and not pid.startswith("live."):
        _add_plugin_file(ex, plugin.source, "datasets")
        return pid, params, None
    if plugin.source == "builtin" and mod.startswith("mltrain.builtin."):
        if pid == "vision.image_folder":
            ex.warn(f"'{s['name']}' trains on a local image folder; it is not included — set `path` in its config.")
        return pid, params, None
    # live sources, deployment feedback, demo data … -> frozen snapshot with the exact splits
    return _snapshot(ex, plugin, s, name)


def _snapshot(ex: Export, plugin, s: dict, name: str) -> tuple[str, dict, str | None]:
    import numpy as np
    import pandas as pd
    seeds = [r["seed"] for r in s["runs"]] or s["seeds"]
    files, sigs, entry = {}, {}, None
    load_params = dict(s["problem_params"])
    if "until" in load_params and not load_params["until"]:      # live source: data window as of the original run
        load_params["until"] = min((r["started_at"] or r["created_at"]) for r in s["runs"]) if s["runs"] else s["created_at"]
    d = ex.out / "data" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        b = plugin.load(load_params, seed)
        if b.modality == "tabular":
            if not b.raw:
                raise ValueError(f"Dataset '{plugin.id}' has no raw tabular data to snapshot")
            target = "target" if "target" not in b.raw["train"][0].columns else "__target"
            parts = []
            for split in ("train", "val", "test"):
                df, y = b.raw[split]
                df = df.copy()
                df[target] = y
                df["__split"] = split
                if b.weights and split == "train" and b.weights.get("train") is not None:
                    df["__weight"] = np.asarray(b.weights["train"], dtype=np.float32)
                elif b.weights:
                    df["__weight"] = 1.0
                parts.append(df)
            full = pd.concat(parts, ignore_index=True)
            sig = pd.util.hash_pandas_object(full, index=False).sum()
            if sig not in sigs:
                fn = f"{name}.csv" if not sigs else f"{name}_seed{seed}.csv"
                full.to_csv(d / fn, index=False)
                sigs[sig] = fn
            files[str(seed)] = sigs[sig]
            entry = {"modality": "tabular", "target": target, "class_names": b.info.get("class_names")}
        elif b.modality == "text":
            rows = [{**r, "__split": sp} for sp in ("train", "val", "test") for r in (getattr(b, sp) or [])]
            blob = "\n".join(json.dumps(r, default=str) for r in rows)
            sig = hash(blob)
            if sig not in sigs:
                fn = f"{name}.jsonl" if not sigs else f"{name}_seed{seed}.jsonl"
                (d / fn).write_text(blob + "\n")
                sigs[sig] = fn
            files[str(seed)] = sigs[sig]
            entry = {"modality": "text", "info": {k: v for k, v in b.info.items() if isinstance(v, (int, float, str, list)) and k != "feature_names"}}
        else:
            ex.warn(f"'{s['name']}' uses a {b.modality} dataset that can't be snapshotted; its config is exported as-is.")
            return s["problem_id"], s["problem_params"], None
    size = sum((d / f).stat().st_size for f in set(files.values()))
    if size > SNAPSHOT_LIMIT:
        for f in set(files.values()):
            (d / f).unlink()
        ex.warn(f"The data snapshot of '{s['name']}' would be {size / 1e6:.0f} MB (> 50 MB) and was not included.")
        return s["problem_id"], s["problem_params"], "Data snapshot too large to include."
    sid = f"snapshot.{name}"
    if len(set(files.values())) == 1:
        files = {"*": next(iter(files.values()))}
    ex.snapshots[sid] = {**entry, "task": s["task"], "files": files, "name": f"{plugin.name} (snapshot)",
                         "description": f"Frozen copy of '{plugin.name}' with the exact train/val/test split the original run used"}
    rows = sum(1 for _ in open(d / next(iter(files.values())))) - 1 if entry["modality"] == "tabular" else None
    note = (f"Frozen snapshot of **{plugin.name}** (`{plugin.id}`) as used by the original run"
            + (f" — {rows:,} rows" if rows else "") + ", with its exact train/val/test split"
            + (" and recency weights" if entry.get("modality") == "tabular" and "__weight" in pd.read_csv(d / next(iter(files.values())), nrows=1).columns else "")
            + f": `data/snapshots/{next(iter(files.values()))}`."
            + (" Taken at export time over the original time window — labels that arrived after the original run are included, "
               "so results can differ slightly from the original." if s["problem_id"].startswith("live.") else ""))
    params = {"standardize": s["problem_params"].get("standardize", True)} if entry["modality"] == "tabular" else {}
    return sid, params, note


# ----------------------------------------------------------------------------- code, requirements
def _write_package(ex: Export) -> None:
    dst = ex.out / "mltrain"
    for rel in PKG_FILES:
        src = PKG / rel
        if src.is_dir():
            shutil.copytree(src, dst / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / rel)
    (dst / "config.py").write_text(CONFIG_PY)
    (dst / "mlops" / "__init__.py").write_text('"""Model loading for predict.py."""\n')
    (dst / "__init__.py").write_text('"""Training code exported from an ML training dashboard (standalone; no server or database needed)."""\n')


def _write_requirements(ex: Export) -> None:
    reqs = list(REQ_BASE)
    for fw in ex.frameworks:
        reqs += REQ_BY_FRAMEWORK.get(fw, [])
    for prefix, rs in REQ_BY_PROBLEM:
        if any(p.startswith(prefix) for p in ex.problem_ids):
            reqs += rs
    extras: dict[str, set] = {}
    for r in reqs:                                   # gymnasium + gymnasium[box2d] + ... -> gymnasium[box2d,...]
        base, _, ex_ = r.partition("[")
        extras.setdefault(base, set()).update(x for x in ex_.rstrip("]").split(",") if x)
    lines = []
    for base, xs in extras.items():
        r = f"{base}[{','.join(sorted(xs))}]" if xs else base
        dist = base
        v = _version(dist)
        if v is None and base == "opencv-python-headless":
            v = _version("opencv-python")
        if dist in ("torch", "torchvision") and v:
            v = v.split("+")[0]
        lines.append(f"{r}=={v}" if v else r)
    (ex.out / "requirements.txt").write_text(
        "# Versions used for the original runs. PyTorch: for a specific CUDA build see https://pytorch.org/get-started/locally/\n"
        + "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- README
def _comparison_figure(ex: Export, exps: list[dict]) -> dict:
    out = {}
    by_key: dict[str, list] = {}
    for e in exps:
        h = e["headline"]
        if h:
            by_key.setdefault(h["key"], []).append({"label": e["cfg"]["title"][:42], **h})
    for key, rows in by_key.items():
        if len(rows) > 1:
            p = report.plot_comparison(rows, ex.out / "results" / f"comparison_{slug(key)}.png")
            if p:
                out[key] = p
    return out


def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root)).replace(os.sep, "/")


def _readme(ex: Export, exps: list[dict], description: str, comparison: dict, include_models: str, license: str) -> str:
    root = ex.out
    L = [f"# {ex.title}", ""]
    if description.strip():
        L += [description.strip(), ""]
    tasks = sorted({e["cfg"]["task"] for e in exps})
    total_seeds = sum(e["n_seeds"] for e in exps)
    L += [f"{len(exps)} experiment{'s' if len(exps) != 1 else ''} · {total_seeds} training runs (seeds) · "
          + ", ".join(tasks) + ". Every number below is mean ± standard deviation over independent random seeds.", ""]

    L += ["## Results", "", "| Experiment | Model | Data / environment | Headline metric | Seeds |", "|---|---|---|---:|---:|"]
    for e in exps:
        c, h = e["cfg"], e["headline"]
        hv = f"`{h['key']}` {report.fmt_pm({**h, 'min': 0, 'max': 0}, h['key'])}" if h else "—"
        L.append(f"| [{c['title']}](#{_anchor(c['title'])}) | {c['model_name']} | `{c['original_problem_id'] or c['problem_id']}` | {hv} | {e['n_seeds']} |")
    L.append("")
    for key, p in comparison.items():
        L += [f"![Comparison of {key}]({_rel(p, root)})", ""]

    L += ["## Reproduce", "", "```bash", "git clone <this repo> && cd <repo>", "python -m venv .venv && source .venv/bin/activate",
          "pip install -r requirements.txt", "", "python train.py --list                 # experiments in this repo",
          "python train.py                        # re-run all of them with the original seeds",
          f"python train.py -e {exps[0]['name']} --seeds 0 1 2 --device cpu", "```", "",
          "Each run writes `runs/<experiment>/seed_<n>/` (metrics.csv, summary.json, plots, saved model) and prints your results "
          "next to the original ones. GPU is used automatically when available; results can differ slightly across hardware "
          "and library versions (GPU kernels aren't bit-for-bit deterministic).", ""]
    if include_models != "none":
        L += ["Trained weights are in `models/<experiment>/seed_<n>/` — "
              "`model/` is the best checkpoint by validation score" + (", `model_final/` the last-epoch weights" if include_models == "both" else "")
              + ". Run predictions with `python predict.py models/<experiment>/seed_0/model input.json` (a JSON list of records).", ""]

    L += ["## Experiments", ""]
    for e in exps:
        c, st = e["cfg"], e["stats"]
        L += [f"### {c['title']}", ""]
        if c["description"]:
            L += [c["description"], ""]
        lines = [f"- **Model:** {c['model_name']} (`{c['model_id']}`, {c['framework']})",
                 f"- **{'Environment' if c['problem_kind'] == 'env' else 'Dataset'}:** `{c['original_problem_id'] or c['problem_id']}`"
                 + (f" — {c['data_note']}" if c["data_note"] else ""),
                 f"- **Task:** {c['task']} · {c['modality']}" + (" · evaluation only (no training)" if c["mode"] == "evaluate" else ""),
                 f"- **Seeds:** {', '.join(map(str, c['seeds']))} · trained {c['trained_on']['date']} on {c['trained_on']['device']}"]
        if c["init_from"]:
            lines.append(f"- **Starts from:** the {c['init_from']['variant']} weights of [{c['init_from']['experiment']}](#{_anchor(_title_of(exps, c['init_from']['experiment']))})")
        if c["parent"]:
            lines.append(f"- **Iteration of:** [{_title_of(exps, c['parent'])}](#{_anchor(_title_of(exps, c['parent']))})")
        lines.append(f"- **Config:** [`experiments/{c['name']}.json`](experiments/{c['name']}.json) · re-run: `python train.py -e {c['name']}`")
        L += lines + [""]
        if st:
            L += [report.results_table(st, _key_metrics(st)), ""]
            bvf = report.best_vs_final_table(st)
            if bvf:
                L += ["**Best checkpoint vs. final weights** (a large gap means the model overfit after its best epoch):", "", bvf, ""]
        figs = e["figs"]
        if "curves" in figs:
            L += [f"![Learning curves]({_rel(figs['curves'], root)})", ""]
        if "confusion" in figs:
            L += [f"<img src=\"{_rel(figs['confusion'], root)}\" alt=\"Confusion matrix\" width=\"420\">", ""]
        if "video" in figs:
            L += [f"🎬 [Evaluation episode of the trained agent]({_rel(figs['video'], root)})", ""]
        L += ["<details><summary>Hyperparameters</summary>", "", "| Parameter | Value |", "|---|---|"]
        L += [f"| `{k}` | `{json.dumps(v)}` |" for k, v in c["model_params"].items()]
        if c["problem_params"]:
            L += [f"| `data.{k}` | `{json.dumps(v)}` |" for k, v in c["problem_params"].items()]
        L += ["", "</details>", ""]

    versions = {}
    for e in exps:
        versions.update(e["cfg"]["trained_on"].get("versions") or {})
    L += ["## Repository layout", "", "```",
          "experiments/   one JSON config per experiment (model, data, hyperparameters, seeds)",
          "results/       original metrics per seed (metrics.csv, summary.json), plots",
          "mltrain/       the training code (models, datasets, environments, training loops)",
          "train.py       re-run experiments  ·  predict.py  run a saved model"]
    if (root / "plugins").exists():
        L.append("plugins/       custom model / dataset / environment code used by the experiments")
    if (root / "data").exists():
        L.append("data/          datasets that can't be downloaded (snapshots, CSVs)")
    if include_models != "none":
        L.append("models/        trained weights")
    L += ["```", ""]
    if versions:
        L += ["<details><summary>Library versions used</summary>", "", ", ".join(f"{k} {v}" for k, v in sorted(versions.items())), "", "</details>", ""]
    if license and license != "none":
        L += ["## License", "", f"{license} — see [LICENSE](LICENSE).", ""]
    return "\n".join(L)


def _key_metrics(st: dict) -> list[str]:
    keys = [k for k in st if not k.startswith("final_model/")]
    main = [k for k in keys if k.startswith(("test/", "final/"))]
    extra = [k for k in ("best_epoch", "epochs_trained", "timesteps_trained", "n_params", "train_time_s", "fit_time_s") if k in st]
    return (main or [k for k in keys if k.startswith("val/")] or keys)[:14] + extra


def _anchor(title: str) -> str:
    return re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")


def _title_of(exps: list[dict], name: str) -> str:
    return next((e["cfg"]["title"] for e in exps if e["name"] == name), name)


def _license(kind: str, author: str) -> str:
    year = _dt.date.today().year
    if kind == "MIT":
        return MIT.format(year=year, author=author)
    return APACHE_NOTE.format(year=year, author=author)


# ----------------------------------------------------------------------------- safety, git, packaging
def _scrub(ex: Export) -> None:
    """Never leak local paths or dashboard secrets into a repo meant to be public."""
    home = str(Path.home())
    cfg = config.load_config() if config.CONFIG_PATH.exists() else {}
    secrets = [v for k, v in cfg.items() if isinstance(v, str) and len(v) >= 8 and k in ("password", "secret_key", "api_token")]
    with db.get_conn() as c:
        for r in c.execute("SELECT config FROM data_sources") if _has_table(c, "data_sources") else []:
            # database passwords inside connection URLs (anywhere in the source config)
            secrets += re.findall(r"[a-z0-9+]+://[^:/@\s\"]*:([^@\s\"]+)@", r["config"] or "")
    for p in ex.out.rglob("*"):
        if not p.is_file() or p.suffix not in (".md", ".json", ".py", ".txt", ".csv", ".jsonl", ".yaml", ".cfg"):
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        new = text
        for sec in secrets:
            if sec in new:
                new = new.replace(sec, "***")
                ex.warn(f"Removed a secret value from {_rel(p, ex.out)}.")
        new = new.replace(home, "~")
        if new != text:
            p.write_text(new)


def _has_table(c, name: str) -> bool:
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _git(out: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=out, capture_output=True, text=True, timeout=120, env={**os.environ, **(env or {})})


def _git_init(out: Path, author: str, email: str) -> dict:
    who = {}
    if author:
        who.update(GIT_AUTHOR_NAME=author, GIT_COMMITTER_NAME=author)
    if email:
        who.update(GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_EMAIL=email)
    _git(out, "init", "-q", "-b", "main")
    _git(out, "add", "-A")
    r = _git(out, "commit", "-q", "-m", "Experiments exported from my ML training dashboard", env=who)
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or r.stdout).strip()[-500:]}
    return {"ok": True, "commit": _git(out, "rev-parse", "--short", "HEAD").stdout.strip()}


def _tree_stats(out: Path) -> tuple[int, int]:
    files = [p for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts]
    return len(files), sum(p.stat().st_size for p in files)


def list_exports() -> list[dict]:
    out = []
    for p in sorted(EXPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if (EXPORTS_DIR / m["name"]).exists():
            m["remote"] = _git(EXPORTS_DIR / m["name"], "remote", "get-url", "origin").stdout.strip() or None \
                if (EXPORTS_DIR / m["name"] / ".git").exists() else None
            out.append(m)
    return out


def export_dir(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", name) or not (EXPORTS_DIR / name).is_dir():
        raise FileNotFoundError(name)
    return EXPORTS_DIR / name


def zip_export(name: str) -> Path:
    d = export_dir(name)
    z = EXPORTS_DIR / f"{name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                zf.write(p, Path(name) / p.relative_to(d))
    return z


def delete_export(name: str) -> None:
    d = export_dir(name)
    shutil.rmtree(d)
    for suffix in (".json", ".zip"):
        (EXPORTS_DIR / f"{name}{suffix}").unlink(missing_ok=True)


def push_export(name: str, remote: str, create_with_gh: bool = False, private: bool = False) -> dict:
    """Publish an export to GitHub. Only ever called from an explicit user action in the dashboard."""
    d = export_dir(name)
    if not (d / ".git").exists():
        raise ValueError("This export has no git repository; rebuild it with git enabled")
    if create_with_gh:
        if not shutil.which("gh"):
            raise ValueError("The GitHub CLI (gh) is not installed. Create an empty repository on github.com and paste its URL instead.")
        r = subprocess.run(["gh", "repo", "create", remote or name, "--private" if private else "--public", "--source", ".",
                            "--remote", "origin", "--push"], cwd=d, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise ValueError((r.stderr or r.stdout).strip()[-800:])
        url = _git(d, "remote", "get-url", "origin").stdout.strip()
        return {"ok": True, "remote": url, "output": (r.stdout + r.stderr).strip()[-800:]}
    if not re.match(r"^(https://github\.com/[\w.-]+/[\w.-]+?(\.git)?|git@github\.com:[\w.-]+/[\w.-]+?(\.git)?)$", remote.strip()):
        raise ValueError("Paste the URL of an empty GitHub repository, e.g. https://github.com/you/my-ml-projects.git")
    remote = remote.strip()
    if _git(d, "remote", "get-url", "origin").returncode == 0:
        _git(d, "remote", "set-url", "origin", remote)
    else:
        _git(d, "remote", "add", "origin", remote)
    r = _git(d, "push", "-u", "origin", "main", env={"GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()[-800:]
        if "could not read Username" in msg or "Authentication failed" in msg or "Permission denied" in msg:
            msg += ("\n\nGit has no GitHub credentials on this machine. Either use an SSH URL (git@github.com:you/repo.git) with an SSH key "
                    "added to GitHub, install the GitHub CLI and run `gh auth login`, or download the zip and push it yourself.")
        raise ValueError(msg)
    return {"ok": True, "remote": remote, "output": (r.stdout + r.stderr).strip()[-800:]}


def git_identity() -> dict:
    g = lambda k: subprocess.run(["git", "config", "--get", k], capture_output=True, text=True).stdout.strip() if shutil.which("git") else ""
    return {"name": g("user.name"), "email": g("user.email"), "git": bool(shutil.which("git")), "gh": bool(shutil.which("gh"))}


# ============================================================================= templates
GITIGNORE = """runs/
data/cache/
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
"""

MIT = """MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

APACHE_NOTE = """Copyright {year} {author}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

TRAIN_PY = r'''#!/usr/bin/env python
"""Re-run the experiments in this repository — no dashboard or database needed.

    python train.py --list
    python train.py                                   # every experiment, original seeds
    python train.py -e <experiment> [-e <other>] --seeds 0 1 2 --device cpu
    python train.py -e <experiment> --set epochs=5 --set lr=0.001   # override hyperparameters
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MLTRAIN_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mltrain import report  # noqa: E402
from mltrain.standalone import run_seed  # noqa: E402


def load(name):
    return json.loads((ROOT / "experiments" / f"{name}.json").read_text())


def order(names):
    """Experiments that start from another experiment's weights run after it."""
    out, seen = [], set()
    def visit(n):
        if n in seen:
            return
        seen.add(n)
        dep = (load(n).get("init_from") or {}).get("experiment")
        if dep and dep in names:
            visit(dep)
        out.append(n)
    for n in names:
        visit(n)
    return out


def init_model(exp, seed):
    """Weights to start from: the source experiment's model trained here (runs/), else the shipped one (models/)."""
    init = exp.get("init_from")
    if not init:
        return None
    sub = "model_final" if init.get("variant") == "final" else "model"
    for base in (ROOT / "runs" / init["experiment"], ROOT / "models" / init["experiment"]):
        cands = sorted(base.glob(f"seed_*/{sub}/meta.json"))
        match = [p for p in cands if p.parent.parent.name == f"seed_{seed}"]
        if init.get("strategy") == "match_seed" and match:
            return str(match[0].parent)
        if cands:   # strategy "best": the seed with the best headline metric
            key, hib = report.HEADLINE.get(exp.get("task"), (None, True))
            score = lambda p: json.loads(p.read_text()).get("summary", {}).get(key)
            scored = [p for p in cands if isinstance(score(p), (int, float))]
            best = (max if hib else min)(scored, key=score) if scored else cands[0]
            return str(best.parent)
    print(f"  ! no trained '{init['experiment']}' model found; train it first (python train.py -e {init['experiment']}) — starting from scratch")
    return None


def coerce(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiment", action="append", help="experiment name (repeatable); default: all")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = the original run's device (GPU if it used one and one is available)")
    ap.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE", help="override a model hyperparameter")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    names = json.loads((ROOT / "experiments" / "index.json").read_text())
    if a.list:
        for n in names:
            e = load(n)
            print(f"{n:<40} {e['model_name']} on {e['original_problem_id'] or e['problem_id']} · seeds {e['seeds']}")
        return
    chosen = a.experiment or names
    unknown = [n for n in chosen if n not in names]
    if unknown:
        sys.exit(f"unknown experiment(s): {unknown}. Use --list.")
    overrides = dict(kv.split("=", 1) for kv in a.set)

    for name in order(chosen):
        exp = load(name)
        exp["model_params"].update({k: coerce(v) for k, v in overrides.items()})
        seeds = a.seeds or exp["seeds"]
        print(f"\n=== {exp['title']}  ({exp['model_name']} · {exp['original_problem_id'] or exp['problem_id']} · seeds {seeds})")
        out = ROOT / "runs" / name
        for seed in seeds:
            print(f"- seed {seed}")
            exp["init_model"] = init_model(exp, seed)
            device = a.device
            if device == "auto" and exp["trained_on"].get("device") == "cpu":
                device = "cpu"
            try:
                run_seed(exp, seed, out, device, a.quiet)
            except KeyboardInterrupt:
                sys.exit("\nstopped")
        series, summaries, artifacts = report.read_runs(out)
        mine = report.summary_stats(summaries)
        orig = report.summary_stats(report.read_runs(ROOT / "results" / name)[1]) if (ROOT / "results" / name).exists() else {}
        report.plot_curves(series, out / "learning_curves.png", exp["title"])
        keys = [k for k in mine if k.startswith(("test/", "final/")) and not k.startswith("final_model/")] or list(mine)[:10]
        print(f"\n  {'metric':<28}{'this run':>24}{'original':>24}")
        for k in keys:
            print(f"  {k:<28}{report.fmt_pm(mine[k], k):>24}{(report.fmt_pm(orig[k], k) if k in orig else '—'):>24}")
        print(f"  outputs: {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
'''

PREDICT_PY = r'''#!/usr/bin/env python
"""Run a saved model on new inputs.

    python predict.py models/<experiment>/seed_0/model input.json
input.json: a JSON list of records — tabular: [{"feature": value, ...}], text: ["some text", ...],
image: [base64 PNG/JPEG strings], RL: [observation vectors]. Raw (un-preprocessed) values are fine:
the preprocessing fitted during training is saved with the model.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MLTRAIN_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))

from mltrain.mlops.predictors import load_predictor  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
model = load_predictor(Path(sys.argv[1]), "cpu")
for out in model.predict(json.loads(Path(sys.argv[2]).read_text())):
    print(json.dumps(out, default=str))
'''
