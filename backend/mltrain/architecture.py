"""Architecture summaries for model plugins: layer tree with parameter counts and output shapes.

The model is built on PyTorch's "meta" device (shapes only, no memory), so even very large networks summarize
instantly; a dummy batch is pushed through with forward hooks recording each module's output shape.
"""
from __future__ import annotations

import inspect
import math
import traceback
from pathlib import Path

DEFAULT_SHAPES = {"image": [3, 32, 32], "audio": [1, 8000], "video": [1, 16, 32, 32], "tabular": [16], "text": [128]}


def _shape(o):
    import torch
    if torch.is_tensor(o):
        return list(o.shape)
    if isinstance(o, (list, tuple)) and o and all(torch.is_tensor(x) for x in o):
        return [list(x.shape) for x in o]
    if hasattr(o, "logits"):
        return list(o.logits.shape)
    return None


def _tree(module, name, shapes, depth, max_depth):
    own = sum(p.numel() for p in module.parameters(recurse=False))
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    extra = module.extra_repr()
    node = {"name": name, "type": type(module).__name__, "params": total, "own_params": own, "trainable": trainable,
            "out_shape": shapes.get(id(module)), "extra": extra[:160] if extra else ""}
    kids = list(module.named_children())
    if kids and depth < max_depth:
        node["children"] = [_tree(m, n, shapes, depth + 1, max_depth) for n, m in kids]
    elif kids:
        node["collapsed"] = len(kids)
    return node


def _run(plugin, params: dict, data_shape, modality: str, device: str, n_classes: int):
    import torch
    from .generative.core import GenerativeModel
    from .generative.lm import LanguageModel
    info = {"data_shape": list(data_shape), "modality": modality, "n_classes": n_classes, "conditional": False}
    batch = 2
    with torch.device(device):
        if isinstance(plugin, LanguageModel):
            info.update(vocab_size=params.get("_vocab_size", 256), block_size=params["block_size"], data_shape=[params["block_size"]])
            plugin._tok = None
            net = plugin.build(params, info)
            x = torch.randint(0, info["vocab_size"], (batch, min(params["block_size"], 64) + 1))
            calls = [lambda: net.losses(x)]
        elif isinstance(plugin, GenerativeModel):
            net = plugin.build(params, info)
            x = torch.randn(batch, *data_shape)
            groups = list(net.optimizer_groups()) if hasattr(net, "optimizer_groups") else [None]
            calls = [(lambda g=g: net.losses(x, None, group=g)) if g else (lambda: net.losses(x)) for g in groups]
        else:
            shape = list(data_shape)
            pinfo = {"input_shape": shape, "input_dim": int(math.prod(shape)), "output_dim": max(n_classes, 1),
                     "n_classes": n_classes, "n_features": shape[0]}
            net = plugin.build(params, pinfo) if hasattr(plugin, "build") else None
            if net is None:
                raise ValueError("This model type has no PyTorch network to summarize")
            x = torch.randn(batch, *shape)
            calls = [lambda: net(x)]
    shapes, hooks = {}, []

    def record(mod, inp, out):          # must return None: a hook's return value replaces the module output
        shapes.setdefault(id(mod), _shape(out))

    for m in net.modules():
        hooks.append(m.register_forward_hook(record))
    try:
        net.train()
        for c in calls:
            with torch.no_grad() if device == "meta" else torch.enable_grad():
                c()
    finally:
        for h in hooks:
            h.remove()
    return net, shapes


def summarize(plugin, params: dict, data_shape: list | None = None, modality: str | None = None, max_depth: int = 4,
              n_classes: int = 10) -> dict:
    from .plugins.base import coerce_params
    mods = getattr(plugin, "modalities", ("image",))
    modality = modality or ("image" if "image" in mods else mods[0])
    data_shape = data_shape or DEFAULT_SHAPES.get(modality, [3, 32, 32])
    p = coerce_params(plugin.params(), params)
    errors, net, shapes, device = [], None, {}, None
    for dev in ("meta", "cpu"):
        try:
            net, shapes = _run(plugin, p, data_shape, modality, dev, n_classes)
            device = dev
            break
        except Exception as e:
            errors.append(f"{dev}: {type(e).__name__}: {e}")
            if net is not None and dev == "meta":
                n = sum(q.numel() for q in net.parameters())
                if n > 200_000_000:
                    break
    if net is None:
        return {"ok": False, "error": "\n".join(errors), "trace": traceback.format_exc(limit=4)}
    return {"ok": True, "device": device, "data_shape": data_shape, "modality": modality,
            "total_params": sum(q.numel() for q in net.parameters()),
            "trainable_params": sum(q.numel() for q in net.parameters() if q.requires_grad),
            "size_mb": sum(q.numel() * q.element_size() for q in net.parameters()) / 1e6,
            "tree": _tree(net, "model", shapes, 0, max_depth), "warnings": errors if device != "meta" else []}


def details(plugin, registry) -> dict:
    """Static facts about a model plugin: role, source file, class source, params with help, discovery notes."""
    from .plugins.introspect import own_doc
    cls = getattr(plugin, "network_class", None) or getattr(plugin, "module_class", None)
    src, file = None, None
    if cls is not None:
        try:
            src, file = inspect.getsource(cls), inspect.getsourcefile(cls)
        except (OSError, TypeError):
            pass
    extra = []
    for c in getattr(plugin, "extra_classes", ()):
        try:
            extra.append({"name": c.__name__, "source": inspect.getsource(c)})
        except (OSError, TypeError):
            pass
    source_file = plugin.source if plugin.source not in ("builtin", "upload", "live") else None
    notes = [n["note"] for n in registry.notes if source_file and Path(n["file"]).name == Path(str(source_file)).name]
    return {"id": plugin.id, "name": plugin.name, "description": plugin.description, "source": plugin.source,
            "class_name": cls.__name__ if cls else None, "class_doc": own_doc(cls) if cls else "",
            "class_source": src, "class_file": file, "extra_classes": extra, "notes": notes,
            "framework": getattr(plugin, "framework", None), "modalities": list(getattr(plugin, "modalities", ())),
            "params": [q.to_dict() for q in plugin.params()]}
