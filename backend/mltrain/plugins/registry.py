"""Discovers built-in plugins and user plugin files in user_plugins/{models,datasets,envs}/.

A user file may define:
  * subclasses of ModelPlugin / DatasetPlugin / EnvPlugin (full control), or
  * a plain torch `nn.Module` subclass  -> auto-wrapped as a TorchModel, its constructor
    arguments with defaults become form parameters, or
  * a plain `gymnasium.Env` subclass     -> auto-wrapped as an EnvPlugin.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
import threading
import traceback
from pathlib import Path

from .. import config
from .base import DatasetPlugin, EnvPlugin, ModelPlugin, Param, Plugin

_BUILTIN_MODULES = [
    "mltrain.builtin.datasets",
    "mltrain.builtin.envs",
    "mltrain.builtin.models_sklearn",
    "mltrain.builtin.models_torch",
    "mltrain.builtin.models_sb3",
    "mltrain.builtin.models_gbm",
    "mltrain.mlops.demo",
    "mltrain.mlops.datasets",
    "mltrain.builtin.models_hf",
    "mltrain.builtin.snapshots",
    "mltrain.generative.diffusion",
    "mltrain.generative.vae",
    "mltrain.generative.gan",
    "mltrain.generative.lm",
    "mltrain.generative.latent",
    "mltrain.generative.datasets",
    "mltrain.builtin.models_transfer",
]
LINKED_FILE = "linked.json"     # user_plugins/linked.json: project folders imported in place (see linked_folders())

KIND_BASE = {"model": ModelPlugin, "dataset": DatasetPlugin, "env": EnvPlugin}
KIND_DIR = {"model": "models", "dataset": "datasets", "env": "envs"}

_builtin_classes: list[type[Plugin]] = []


def register(cls: type[Plugin]) -> type[Plugin]:
    """Decorator for built-in plugin classes."""
    _builtin_classes.append(cls)
    return cls


class Registry:
    def __init__(self) -> None:
        self.plugins: dict[str, dict[str, Plugin]] = {k: {} for k in KIND_BASE}
        self.errors: list[dict] = []
        self.hashes: dict[str, str] = {}   # plugin source file -> sha256 (reproducibility)
        self.notes: list[dict] = []        # discovery notes per file (e.g. classes skipped as building blocks)
        self.files: dict[str, list[str]] = {}   # source file -> plugin ids it produced
        self.hidden: dict[str, dict[str, Plugin]] = {}    # kind -> plugins removed in the dashboard
        self._lock = threading.Lock()

    # ---- loading -------------------------------------------------------
    def load(self) -> "Registry":
        with self._lock:
            self.plugins = {k: {} for k in KIND_BASE}
            self.errors, self.notes, self.files = [], [], {}
            try:
                from ..generative import components
                components.reset()
            except ImportError:
                pass
            for mod in _BUILTIN_MODULES:
                try:
                    importlib.import_module(mod)
                except ModuleNotFoundError as e:
                    if not (e.name or "").startswith("mltrain."):   # dashboard-only modules are absent in standalone exports
                        self.errors.append({"file": mod, "error": traceback.format_exc(limit=3)})
                except Exception:
                    self.errors.append({"file": mod, "error": traceback.format_exc(limit=3)})
            for cls in _builtin_classes:
                self._add(cls(), "builtin")
            # dynamic builtins (e.g. one dataset per uploaded CSV)
            from ..builtin.datasets import uploaded_csv_datasets
            for p in uploaded_csv_datasets():
                self._add(p, "upload")
            try:
                from ..data.live import live_datasets
            except ModuleNotFoundError:
                live_datasets = list
            for p in live_datasets():
                self._add(p, "live")
            for kind, sub in KIND_DIR.items():
                for path in sorted((config.USER_PLUGINS_DIR / sub).glob("*.py")):
                    self._load_file(path, kind)
                # optional extra datasets / models / envs: kept locally, not in git
                for path in sorted((config.USER_PLUGINS_DIR / sub / "optional").glob("*.py")):
                    self._load_file(path, kind)
            for link in linked_folders():
                folder = Path(link["path"])
                if not folder.is_dir():
                    self.errors.append({"file": str(folder), "error": "Linked project folder not found"})
                    continue
                if str(folder) not in sys.path:
                    sys.path.insert(0, str(folder))       # the project's own sibling imports keep working
                dont_write = sys.dont_write_bytecode
                sys.dont_write_bytecode = True            # never write __pycache__ into someone else's project
                try:
                    for f in link.get("files", []):
                        self._load_file(folder / f["file"], f.get("kind", "model"), prefix=f"linked_{_slug(str(folder))}")
                finally:
                    sys.dont_write_bytecode = dont_write
            try:
                from ..generative import losses as gen_losses
                gen_losses.load()
                for link in linked_folders():
                    for f in link.get("files", []):
                        if f.get("kind") == "losses":
                            gen_losses.load_files([Path(link["path"]) / f["file"]])
                self.errors += [{"file": e["file"], "error": e["error"]} for e in gen_losses.errors()]
            except Exception:
                self.errors.append({"file": "losses", "error": traceback.format_exc(limit=3)})
            # datasets / environments removed in the dashboard (built-ins can't be deleted, so they're hidden)
            self.hidden = {}
            for kind in ("dataset", "env"):
                self.hidden[kind] = {pid: self.plugins[kind].pop(pid) for pid in hidden_plugins(kind) if pid in self.plugins[kind]}
        return self

    def _add(self, plugin: Plugin, source: str) -> None:
        plugin.source = source
        bucket = self.plugins[plugin.kind]
        if plugin.id in bucket and source != "builtin":
            self.errors.append({"file": source, "error": f"Duplicate {plugin.kind} id '{plugin.id}' (overrides earlier)"})
        bucket[plugin.id] = plugin

    def _load_file(self, path: Path, kind: str, prefix: str = "user_plugins") -> None:
        if kind == "losses":
            return
        mod_name = f"{prefix}.{KIND_DIR[kind]}.{path.stem}"
        try:
            self.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            found = 0
            source = str(path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path)
            modules = []
            for cname, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != mod_name or cname.startswith("_"):   # _private helpers aren't plugins
                    continue
                if kind == "model" and _is_torch_module(obj) and not issubclass(obj, ModelPlugin):
                    modules.append(obj)                                  # classified together (blocks, GAN pairs)
                    continue
                plugin = self._plugin_from_class(obj, kind, path)
                if plugin is not None:
                    self._add(plugin, source)
                    self.files.setdefault(source, []).append(plugin.id)
                    found += 1
            if modules:
                from ..generative.discovery import plugins_for_file
                plugins, notes = plugins_for_file(modules, path)
                self.notes += [{"file": source, "note": n} for n in notes]
                for plugin in plugins:
                    self._add(plugin, source)
                    self.files.setdefault(source, []).append(plugin.id)
                    found += 1
            if not found:
                self.errors.append({"file": str(path), "error": f"No {kind} plugin class found in file"})
        except Exception:
            self.errors.append({"file": str(path), "error": traceback.format_exc(limit=5)})

    def _plugin_from_class(self, obj: type, kind: str, path: Path) -> Plugin | None:
        base = KIND_BASE[kind]
        if issubclass(obj, base) and obj.id and not inspect.isabstract(obj):
            return obj()
        if kind == "model":
            try:
                import torch.nn as nn
                if issubclass(obj, nn.Module):
                    from ..builtin.models_torch import wrap_module
                    return wrap_module(obj, path)
            except ImportError:
                pass
        if kind == "env":
            try:
                import gymnasium as gym
                if issubclass(obj, gym.Env):
                    return wrap_env(obj, path)
            except ImportError:
                pass
        return None

    # ---- queries -------------------------------------------------------
    def get(self, kind: str, pid: str) -> Plugin:
        try:
            return self.plugins[kind][pid]
        except KeyError:
            raise KeyError(f"Unknown {kind} '{pid}'") from None

    def list(self, kind: str) -> list[Plugin]:
        return list(self.plugins[kind].values())

    def problem(self, kind: str, pid: str) -> DatasetPlugin | EnvPlugin:
        return self.get("dataset" if kind == "dataset" else "env", pid)


def wrap_env(env_cls: type, path: Path) -> EnvPlugin:
    """Turn a plain gymnasium.Env subclass into an EnvPlugin; ctor kwargs become params."""
    params = _params_from_signature(env_cls.__init__, group="Environment")

    class Wrapped(EnvPlugin):
        id = f"custom.{path.stem}.{env_cls.__name__}"
        name = env_cls.__name__
        description = (inspect.getdoc(env_cls) or f"Custom environment from {path.name}").split("\n")[0]

        def params(self):
            return params

        def make(self, p, seed, render_mode=None):
            kw = {k: v for k, v in p.items() if k in {q.name for q in params}}
            if "render_mode" in inspect.signature(env_cls.__init__).parameters:
                kw["render_mode"] = render_mode
            env = env_cls(**kw)
            if render_mode and getattr(env, "render_mode", None) is None:
                env.render_mode = render_mode
            return env

    return Wrapped()


_RESERVED = {"self", "render_mode", "input_dim", "output_dim", "num_classes", "input_shape", "in_channels", "args", "kwargs"}


def _is_torch_module(obj) -> bool:
    try:
        import torch.nn as nn
    except ImportError:
        return False
    return isinstance(obj, type) and issubclass(obj, nn.Module)


def _hidden_path(kind: str):
    return config.DATA_DIR / f"hidden_{KIND_DIR[kind]}.json"      # hidden_datasets.json / hidden_envs.json


def hidden_plugins(kind: str) -> list[str]:
    import json
    p = _hidden_path(kind)
    try:
        return json.loads(p.read_text()) if p.exists() else []
    except (OSError, ValueError):
        return []


def set_hidden_plugins(kind: str, ids: list[str]) -> None:
    import json
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _hidden_path(kind).write_text(json.dumps(sorted(set(ids)), indent=2))


def _slug(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def linked_folders() -> list[dict]:
    """[{"path": folder, "files": [{"file": "UNet.py", "kind": "model"|"dataset"|"env"|"losses"}]}]"""
    import json
    p = config.USER_PLUGINS_DIR / LINKED_FILE
    try:
        return json.loads(p.read_text()) if p.exists() else []
    except (OSError, json.JSONDecodeError):
        return []


def save_linked_folders(links: list[dict]) -> None:
    import json
    p = config.USER_PLUGINS_DIR / LINKED_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(links, indent=2))


def _params_from_signature(fn, group: str = "Model") -> list[Param]:
    out = []
    for name, prm in inspect.signature(fn).parameters.items():
        if name in _RESERVED or prm.default is inspect.Parameter.empty:
            continue
        d = prm.default
        if isinstance(d, bool):
            t = "bool"
        elif isinstance(d, int):
            t = "int"
        elif isinstance(d, float):
            t = "float"
        elif isinstance(d, (list, tuple)) and all(isinstance(x, int) for x in d):
            t, d = "int_list", list(d)
        else:
            t, d = "str", (d if d is None else str(d))
        out.append(Param(name, t, d, group=group))
    return out


_registry: Registry | None = None


def get_registry(reload: bool = False) -> Registry:
    global _registry
    if _registry is None or reload:
        _registry = Registry().load()
    return _registry
