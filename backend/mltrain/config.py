"""Paths and persistent server configuration (password, secret key, queue limits)."""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

ROOT = Path(os.environ.get("MLTRAIN_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("MLTRAIN_DATA", ROOT / "data"))
DB_PATH = DATA_DIR / "mltrain.db"
RUNS_DIR = DATA_DIR / "runs"            # per-run logs, checkpoints, artifacts
UPLOADS_DIR = DATA_DIR / "uploads"      # uploaded datasets (csv etc.)
CACHE_DIR = DATA_DIR / "cache"          # torchvision / HF downloads
USER_PLUGINS_DIR = Path(os.environ.get("MLTRAIN_PLUGINS", ROOT / "user_plugins"))
CONFIG_PATH = DATA_DIR / "config.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"

DEFAULTS = {
    "max_gpu_jobs": 1,
    "max_cpu_jobs": 2,
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, RUNS_DIR, UPLOADS_DIR / "datasets", CACHE_DIR,
              USER_PLUGINS_DIR / "models", USER_PLUGINS_DIR / "datasets", USER_PLUGINS_DIR / "envs"):
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load config.json, creating it (with a random password) on first start."""
    ensure_dirs()
    cfg: dict = {}
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
    changed = False
    if "secret_key" not in cfg:
        cfg["secret_key"] = secrets.token_hex(32)
        changed = True
    if "password" not in cfg:
        cfg["password"] = os.environ.get("MLTRAIN_PASSWORD") or secrets.token_urlsafe(9)
        cfg["password_generated"] = True
        changed = True
    if "api_token" not in cfg:
        cfg["api_token"] = secrets.token_urlsafe(24)
        changed = True
    for k, v in DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
            changed = True
    if changed:
        save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    os.chmod(CONFIG_PATH, 0o600)
