"""Removing (and restoring) datasets and RL environments from the dashboard.

    uploaded CSV                 -> its files are deleted
    plugin file in user_plugins/ -> the file moves to user_plugins/.trash/ (restorable) when it defines only this dataset
    built-in / linked project    -> hidden (data/hidden_datasets.json); your project's files are never touched
    live source                  -> removed on Data -> Live sources (it has its own settings and history)

Downloaded files (torchvision, Hugging Face, sklearn, ... in data/cache) and quality reports can be deleted too;
they're downloaded again if the dataset is used later.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from . import config, db
from .plugins.registry import get_registry, hidden_plugins, set_hidden_plugins

TRASH = lambda: config.USER_PLUGINS_DIR / ".trash"


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _data_paths(plugin) -> list[Path]:
    try:
        paths = [Path(q) for q in (plugin.data_paths() if hasattr(plugin, "data_paths") else [])]
    except Exception:
        return []
    cache = config.CACHE_DIR.resolve()
    return [q for q in paths if q.exists() and q.resolve().is_relative_to(cache)]      # only ever inside data/cache


def _reports(pid: str) -> list[Path]:
    d = config.DATA_DIR / "reports"
    return sorted(d.glob(f"{pid}-*.json")) if d.exists() else []


def _plugin_file(plugin) -> Path | None:
    src = plugin.source
    if src in ("builtin", "upload", "live") or not src.endswith(".py"):
        return None
    p = Path(src) if Path(src).is_absolute() else config.ROOT / src
    plugins_root = config.USER_PLUGINS_DIR.resolve()
    return p if p.exists() and p.resolve().is_relative_to(plugins_root) else None


def plan(pid: str, kind: str = "dataset") -> dict:
    """What removing this dataset / environment would do (shown before confirming)."""
    if kind not in ("dataset", "env"):
        raise ValueError("only datasets and environments can be removed")
    reg = get_registry()
    plugin = reg.plugins[kind].get(pid)
    if plugin is None:
        raise KeyError(pid)
    with db.get_conn() as c:
        sessions = c.execute("SELECT COUNT(*) FROM sessions WHERE problem_id=?", (pid,)).fetchone()[0]
    data = _data_paths(plugin)
    out = {"id": pid, "kind_of_plugin": kind, "name": plugin.name, "sessions": sessions, "data_bytes": sum(_size(q) for q in data),
           "data_paths": [str(q.relative_to(config.DATA_DIR)) for q in data], "reports": len(_reports(pid))}
    if kind == "dataset" and pid.startswith("live."):
        out.update(kind="live", action="Remove it on Data → Live sources (it has its own settings and history).")
    elif plugin.source == "upload":
        out.update(kind="upload", action="Deletes the uploaded CSV.")
    else:
        f = _plugin_file(plugin)
        src = str(f.relative_to(config.ROOT)) if f and f.is_relative_to(config.ROOT) else str(f) if f else None
        others = [x for x in reg.files.get(plugin.source, []) if x != pid] if f else []
        if f and not others:
            out.update(kind="file", file=src, action=f"Moves {src} to user_plugins/.trash/ (restorable).")
        else:
            out.update(kind="hide", action="Hides it from the dashboard (restorable)" +
                       (f"; {src} also defines {', '.join(others)}, so the file stays." if others else
                        "; it's built into the dashboard's code." if plugin.source == "builtin" else "."))
    return out


def remove(pid: str, delete_data: bool = True, kind: str = "dataset") -> dict:
    p = plan(pid, kind)
    reg = get_registry()
    plugin = reg.plugins[kind][pid]
    if p["kind"] == "live":
        raise ValueError(p["action"])
    freed = 0
    if delete_data:
        for q in _data_paths(plugin) + _reports(pid):
            freed += _size(q)
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
    if p["kind"] == "upload":
        slug = pid.split(".", 1)[1]
        for ext in (".csv", ".json"):
            f = config.UPLOADS_DIR / "datasets" / f"{slug}{ext}"
            freed += _size(f)
            f.unlink(missing_ok=True)
    elif p["kind"] == "file":
        f = _plugin_file(plugin)
        TRASH().mkdir(parents=True, exist_ok=True)
        rel = f.resolve().relative_to(config.USER_PLUGINS_DIR.resolve())
        dest = TRASH() / f"{time.strftime('%Y%m%d-%H%M%S')}__{'__'.join(rel.parts)}"
        shutil.move(str(f), dest)
    else:
        set_hidden_plugins(kind, hidden_plugins(kind) + [pid])
    get_registry(reload=True)
    return {"ok": True, "kind": p["kind"], "freed_bytes": freed}


def removed(kind: str = "dataset") -> list[dict]:
    """Hidden plugins and plugin files in the trash (of this kind), for restoring."""
    from .plugins.registry import KIND_DIR
    reg = get_registry()
    hidden = reg.hidden.get(kind, {})
    out = [{"kind": "hidden", "id": pid, "name": pl.name, "restore": pid} for pid, pl in hidden.items()]
    out += [{"kind": "hidden", "id": pid, "name": pid, "restore": pid} for pid in hidden_plugins(kind) if pid not in hidden
            and pid not in reg.plugins[kind]]
    if TRASH().exists():
        for f in sorted(TRASH().glob(f"*__{KIND_DIR[kind]}__*.py"), reverse=True):
            stamp, _, rest = f.name.partition("__")
            out.append({"kind": "trash", "id": f.name, "name": rest.replace("__", "/"), "deleted": stamp, "restore": f.name})
    return out


def restore(key: str, kind: str = "dataset") -> dict:
    if key in hidden_plugins(kind):
        set_hidden_plugins(kind, [x for x in hidden_plugins(kind) if x != key])
    else:
        f = TRASH() / key
        if "/" in key or not f.is_file():
            raise KeyError(key)
        stamp, _, rest = key.partition("__")
        dest = config.USER_PLUGINS_DIR.joinpath(*rest.split("__"))
        if dest.exists():
            raise ValueError(f"{dest.name} already exists; remove it first")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), dest)
    reg = get_registry(reload=True)
    return {"ok": True, "errors": reg.errors}
