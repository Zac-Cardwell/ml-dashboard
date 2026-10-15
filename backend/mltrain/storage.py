"""Saved models, checkpoints and media on disk (data/runs/<run_id>/...)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config


def run_dir(run_id: str) -> Path:
    return config.RUNS_DIR / run_id


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


VARIANT_DIRS = {"best": "model", "final": "model_final"}


def model_dir(run_id: str, variant: str = "best") -> Path:
    """'best' = model/ (the primary saved model: best checkpoint, or the final one if only final was kept); 'final' = model_final/."""
    return run_dir(run_id) / VARIANT_DIRS.get(variant or "best", "model")


def model_meta(run_id: str, variant: str = "best") -> dict | None:
    meta = model_dir(run_id, variant) / "meta.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_storage(run_id: str) -> dict:
    d = run_dir(run_id)
    return {"has_model": (d / "model" / "meta.json").exists(), "model_bytes": dir_size(d / "model"),
            "has_final": (d / "model_final" / "meta.json").exists(), "final_bytes": dir_size(d / "model_final"),
            "has_checkpoint": (d / "checkpoint").exists() and any((d / "checkpoint").iterdir()),
            "checkpoint_bytes": dir_size(d / "checkpoint"), "media_bytes": dir_size(d / "media"),
            "total_bytes": dir_size(d)}


def delete_model(run_id: str, checkpoints: bool = True, variant: str | None = None) -> int:
    """Delete saved models (both variants unless `variant` is given, plus optionally the resumable checkpoint); returns bytes freed."""
    d = run_dir(run_id)
    freed = 0
    if variant:
        sub = d / VARIANT_DIRS[variant]
        freed = dir_size(sub)
        shutil.rmtree(sub, ignore_errors=True)
        return freed
    for sub in ("model", "model_final") + (("checkpoint",) if checkpoints else ()):
        freed += dir_size(d / sub)
        shutil.rmtree(d / sub, ignore_errors=True)
    for legacy in ("best_model.zip", "final_model.zip", "final.pt", "best.pt", "model.joblib"):
        if (d / legacy).exists():
            freed += (d / legacy).stat().st_size
            (d / legacy).unlink()
    return freed
