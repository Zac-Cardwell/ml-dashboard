"""Model registry: immutable, versioned copies of trained models with stages and reference profiles.

A version is a copy of a run's saved model under data/registry/<name>/v<N>/, so deleting the training
session never breaks a deployment. Each version carries meta.json (incl. the fitted preprocessor),
profile.json (training reference distributions for drift) and card.json (a model card).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from fastapi import HTTPException

from .. import analysis, config, db, storage

REGISTRY_DIR = config.DATA_DIR / "registry"
STAGES = ("none", "staging", "production", "archived")


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
    if not s:
        raise HTTPException(400, "Name must contain letters or digits")
    return s


def best_run(c, session_id: str) -> dict:
    s = db.one(c, "SELECT task FROM sessions WHERE id=?", (session_id,))
    runs = [r for r in db.rows(c, "SELECT * FROM runs WHERE session_id=?", (session_id,)) if storage.model_meta(r["id"])]
    if not runs:
        raise HTTPException(400, "This session has no saved models to register")
    key, hib = analysis.HEADLINE.get(s["task"], (None, True))
    scored = [r for r in runs if isinstance((r["summary"] or {}).get(key), (int, float))]
    return (max if hib else min)(scored, key=lambda r: r["summary"][key]) if scored else runs[0]


def register_version(model_name: str, session_id: str, run_id: str | None = None, description: str = "",
                     created_by: str = "user", variant: str = "best") -> dict:
    """Copy a run's saved model into the registry, profile it, and create the next version."""
    from .predictors import load_predictor
    from .profile import build_profile
    name = slug(model_name)
    with db.get_conn() as c:
        sess = db.one(c, "SELECT * FROM sessions WHERE id=?", (session_id,))
        if not sess:
            raise HTTPException(404, "session not found")
        run = db.one(c, "SELECT * FROM runs WHERE id=? AND session_id=?", (run_id, session_id)) if run_id else best_run(c, session_id)
        if not run or not storage.model_meta(run["id"], variant):
            raise HTTPException(400, "That run has no saved final-epoch model" if variant == "final" else "That run has no saved model")
        existing = db.one(c, "SELECT * FROM registered_models WHERE name=?", (name,))
        if existing and existing["model_id"] != sess["model_id"]:
            raise HTTPException(400, f"Registered model '{name}' holds {existing['model_id']} models; use a new name")
        version = (c.execute("SELECT MAX(version) FROM model_versions WHERE model_name=?", (name,)).fetchone()[0] or 0) + 1
    dest = REGISTRY_DIR / name / f"v{version}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(storage.model_dir(run["id"], variant), dest)
    meta = json.loads((dest / "meta.json").read_text())
    try:
        profile = build_profile(dest, meta, load_predictor(dest, "cpu"))
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(400, f"Could not profile the model for monitoring: {e}")
    (dest / "profile.json").write_text(json.dumps(profile))
    metrics = dict(meta.get("summary") or {}) if variant == "final" else \
        {k: v for k, v in (run["summary"] or {}).items() if isinstance(v, (int, float)) and not k.startswith("final_model/")}
    card = {"name": name, "version": version, "model": meta["model_id"], "framework": meta["framework"], "task": meta["task"],
            "trained_on": {"problem": meta["problem_id"], "params": meta["problem_params"]}, "hyperparameters": meta["model_params"],
            "metrics": metrics, "source_session": session_id, "source_run": run["id"], "seed": run["seed"], "weights": meta.get("variant", variant),
            "versions": (sess.get("env_info") or {}).get("versions"), "description": description,
            "input_schema": profile.get("schema"), "example_input": profile.get("example")}
    (dest / "card.json").write_text(json.dumps(card, indent=2, default=str))
    vid = db.new_id()
    with db.get_conn() as c:
        if not existing:
            c.execute("INSERT INTO registered_models(name,description,task,modality,model_id,created_at) VALUES (?,?,?,?,?,?)",
                      (name, description, sess["task"], sess["modality"], sess["model_id"], db.now()))
        c.execute("INSERT INTO model_versions(id,model_name,version,stage,source_session_id,source_run_id,model_id,framework,task,modality,"
                  "path,metrics,description,created_by,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (vid, name, version, "none", session_id, run["id"], sess["model_id"], sess["framework"], sess["task"],
                   sess["modality"], str(dest), json.dumps(metrics), description, created_by, db.now()))
        return db.one(c, "SELECT * FROM model_versions WHERE id=?", (vid,))


def get_version(c, vid: str) -> dict:
    v = db.one(c, "SELECT * FROM model_versions WHERE id=?", (vid,))
    if not v:
        raise HTTPException(404, "model version not found")
    return v


def load_profile(version: dict) -> dict:
    return json.loads((Path(version["path"]) / "profile.json").read_text())


def set_stage(c, vid: str, stage: str) -> None:
    if stage not in STAGES:
        raise HTTPException(400, f"stage must be one of {STAGES}")
    c.execute("UPDATE model_versions SET stage=? WHERE id=?", (stage, vid))
