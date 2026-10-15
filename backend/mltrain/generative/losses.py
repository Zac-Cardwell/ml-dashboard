"""Loss functions selectable in generative model forms.

Add your own by dropping a .py file into `user_plugins/losses/` (or a linked project). Every public top-level
function with the signature `fn(pred, target)` that returns a scalar tensor becomes a choice named after the function:

    def charbonnier(pred, target, eps=1e-3):
        return ((pred - target) ** 2 + eps ** 2).sqrt().mean()

Diffusion uses it between the network output and its target (noise / x0 / v); VAEs between the reconstruction and
the input (the KL term is added separately). Language models use cross-entropy unless a custom loss is chosen
(called with logits (N, V) and integer targets (N,)).
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import config

BUILTIN = {
    "mse": lambda p, t: F.mse_loss(p, t),
    "l1": lambda p, t: F.l1_loss(p, t),
    "huber": lambda p, t: F.smooth_l1_loss(p, t, beta=0.1),
    "charbonnier": lambda p, t: ((p - t) ** 2 + 1e-6).sqrt().mean(),
    # images/audio live in [-1, 1]; BCE is computed on the [0, 1] rescaling
    "bce": lambda p, t: F.binary_cross_entropy(((p + 1) / 2).clamp(1e-6, 1 - 1e-6), ((t + 1) / 2).clamp(0, 1)),
}
DESCRIPTIONS = {"mse": "mean squared error", "l1": "mean absolute error", "huber": "smooth L1 (beta 0.1)",
                "charbonnier": "differentiable L1", "bce": "binary cross-entropy on [0,1]-scaled values"}

_custom: dict[str, dict] = {}
_errors: list[dict] = []


def load_files(paths: list[Path]) -> None:
    """Register every public `fn(pred, target, ...)` function from these files."""
    for path in paths:
        name = f"user_losses.{Path(path).stem}"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            for fname, fn in inspect.getmembers(mod, inspect.isfunction):
                if fn.__module__ != name or fname.startswith("_"):
                    continue
                if len([p for p in inspect.signature(fn).parameters.values() if p.default is inspect.Parameter.empty]) < 2:
                    continue
                _custom[fname] = {"fn": fn, "file": str(path), "doc": (inspect.getdoc(fn) or "").split("\n")[0],
                                  "source": inspect.getsource(fn)}
        except Exception:
            _errors.append({"file": str(path), "error": traceback.format_exc(limit=3)})


def load(extra_dirs: list[Path] | None = None) -> None:
    _custom.clear()
    _errors.clear()
    for d in [config.USER_PLUGINS_DIR / "losses", *(extra_dirs or [])]:
        if d.is_dir():
            load_files(sorted(d.glob("*.py")))


def names() -> list[str]:
    return list(BUILTIN) + sorted(_custom)


def get(name: str):
    if name in BUILTIN:
        return BUILTIN[name]
    if name in _custom:
        return _custom[name]["fn"]
    if not _custom:
        load()
        if name in _custom:
            return _custom[name]["fn"]
    raise ValueError(f"Unknown loss '{name}'. Available: {names()}")


def catalog() -> list[dict]:
    out = [{"name": k, "kind": "builtin", "doc": DESCRIPTIONS.get(k, "")} for k in BUILTIN]
    out += [{"name": k, "kind": "custom", "doc": v["doc"], "file": v["file"], "source": v["source"]} for k, v in sorted(_custom.items())]
    return out


def errors() -> list[dict]:
    return list(_errors)


def check(name: str, shape=(2, 1, 8, 8)) -> None:
    """Raise a readable error if a loss can't be called like the trainer calls it."""
    p, t = torch.zeros(shape, requires_grad=True), torch.zeros(shape)
    v = get(name)(p, t)
    if not torch.is_tensor(v) or v.dim() != 0:
        raise ValueError(f"Loss '{name}' must return a scalar tensor")
