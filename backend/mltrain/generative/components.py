"""Interchangeable model components.

Methods made of several networks declare *slots* (GAN: generator + discriminator; VAE: encoder + decoder;
latent diffusion: autoencoder + denoiser; ...). Each slot accepts any component with the right *interface*:

    denoiser     forward(x, t[, y]) -> tensor shaped like x
    decoder      forward(z) -> sample in [-1, 1]            (GAN generators, VAE decoders)
    encoder      forward(x) -> features (any size)          (VAE encoders, GAN discriminators, classifiers' bodies;
                                                            a trainable linear head adapts the size)
    autoencoder  encode(x) -> latent, decode(latent) -> x   (VAEs; Stable Diffusion's VAE / TAESD)

and a slot can be filled from four sources:

    builtin               the method's own configurable network
    class:<file>.<Class>  your nn.Module (uploaded or from a linked folder), optionally initialized from a .pt/.pth file
    session:<id>:<part>   a part of a model trained in this dashboard (best seed), e.g. a VAE's decoder as a GAN generator
    hub:<name>            a pretrained network that is downloaded once (Stable Diffusion VAEs, a pretrained DDPM UNet,
                          ImageNet encoders)

Every non-built-in component can be frozen (weights fixed, kept in eval mode; frozen downloads aren't re-saved).
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..plugins.base import Param

INTERFACES = {
    "denoiser": "forward(x, t[, y]) -> same shape as x",
    "decoder": "forward(z) -> sample (generator / VAE decoder)",
    "encoder": "forward(x) -> features (a linear head adapts the size)",
    "autoencoder": "encode(x) -> latent, decode(latent) -> x",
}

# ============================================================================= user classes by interface
_CLASSES: dict[str, dict] = {}       # "file.Class" -> {"cls", "interfaces", "path", "params"}


def reset() -> None:
    _CLASSES.clear()


def register_class(cls, interfaces: list[str], path) -> None:
    from ..plugins.introspect import extract
    key = f"{Path(path).stem}.{cls.__name__}"
    _CLASSES[key] = {"cls": cls, "interfaces": interfaces, "path": str(path), "params": extract(cls, "Network")}


def classes_for(interface: str) -> dict[str, dict]:
    return {k: v for k, v in _CLASSES.items() if interface in v["interfaces"]}


# ============================================================================= pretrained downloads
HUB = {
    "taesd": {"interface": "autoencoder", "label": "TAESD — tiny Stable Diffusion VAE (10 MB, fast; 8x downsampling, 4 channels)",
              "modality": "image"},
    "sd-vae-ft-mse": {"interface": "autoencoder", "label": "Stable Diffusion VAE ft-MSE (335 MB; 8x downsampling, 4 channels)",
                      "modality": "image"},
    "ddpm-cifar10": {"interface": "denoiser", "label": "DDPM UNet pretrained on CIFAR-10 (Ho et al.; 36M params, 3x32x32)",
                     "modality": "image"},
    "resnet18": {"interface": "encoder", "label": "ResNet-18, ImageNet (512 features)", "modality": "image"},
    "resnet50": {"interface": "encoder", "label": "ResNet-50, ImageNet (2048 features)", "modality": "image"},
    "efficientnet_b0": {"interface": "encoder", "label": "EfficientNet-B0, ImageNet (1280 features)", "modality": "image"},
}


def _hf_cache():
    from .. import config
    return str(config.CACHE_DIR / "hf_models")


class HubAutoencoder(nn.Module):
    """diffusers AutoencoderTiny / AutoencoderKL in [-1, 1] space; grayscale data is repeated to RGB and averaged back."""

    def __init__(self, name: str, channels: int):
        super().__init__()
        from diffusers import AutoencoderKL, AutoencoderTiny
        if name == "taesd":
            self.ae, self.kind, self.scale = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir=_hf_cache()), "tiny", 1.0
        else:
            self.ae, self.kind, self.scale = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=_hf_cache()), "kl", 0.18215
        self.channels = channels

    def encode(self, x):
        if self.channels == 1:
            x = x.repeat(1, 3, *([1] * (x.dim() - 2)))
        out = self.ae.encode(x)
        return out.latents if self.kind == "tiny" else out.latent_dist.mean

    def decode(self, z):
        x = self.ae.decode(z).sample
        return x.mean(1, keepdim=True) if self.channels == 1 else x


class HubDenoiser(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        from diffusers import UNet2DModel
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32", cache_dir=_hf_cache())

    def forward(self, x, t, y=None):
        return self.unet(x, t).sample


class TorchvisionEncoder(nn.Module):
    """ImageNet backbone as a feature extractor; takes [-1, 1] images (grayscale repeated to RGB)."""

    def __init__(self, name: str):
        super().__init__()
        import torchvision
        from .. import config
        torch.hub.set_dir(str(config.CACHE_DIR / "torch_hub"))
        weights = {"resnet18": "ResNet18_Weights", "resnet50": "ResNet50_Weights", "efficientnet_b0": "EfficientNet_B0_Weights"}[name]
        m = getattr(torchvision.models, name)(weights=getattr(torchvision.models, weights).IMAGENET1K_V1)
        if hasattr(m, "fc"):
            m.fc = nn.Identity()
        else:
            m.classifier = nn.Identity()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if min(x.shape[-2:]) < 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return self.m(((x + 1) / 2 - self.mean) / self.std)


def load_hub(name: str, info: dict) -> nn.Module:
    if name not in HUB:
        raise ValueError(f"Unknown pretrained component '{name}'")
    shape = info["data_shape"]
    if HUB[name]["modality"] != info.get("modality", "image") or len(shape) != 3:
        raise ValueError(f"'{name}' works on images only")
    if name in ("taesd", "sd-vae-ft-mse"):
        if shape[0] not in (1, 3) or shape[1] % 8 or shape[2] % 8:
            raise ValueError(f"'{name}' needs 1- or 3-channel images whose size is a multiple of 8 (data is "
                             f"{'x'.join(map(str, shape))}; set image_size, e.g. 32)")
        m = HubAutoencoder(name, shape[0])
    elif name == "ddpm-cifar10":
        if list(shape) != [3, 32, 32]:
            raise ValueError(f"The pretrained CIFAR-10 DDPM expects 3x32x32 data (this is {'x'.join(map(str, shape))})")
        m = HubDenoiser(name)
    else:
        m = TorchvisionEncoder(name)
    m._external = True          # re-downloaded when the model is loaded; not stored in model.pt
    return m


# ============================================================================= adapters
class MethodModule(nn.Module):
    """Expose one method of a module as forward() (e.g. a VAE class's decode as a decoder)."""

    def __init__(self, module: nn.Module, method: str, first: bool = False):
        super().__init__()
        self.module, self.method, self.first = module, method, first

    def forward(self, *a, **kw):
        out = getattr(self.module, self.method)(*a, **kw)
        return out[0] if self.first and isinstance(out, (tuple, list)) else out


class WithHead(nn.Module):
    """encoder(x) -> features -> Linear -> out_features. The feature size is measured with a dummy batch."""

    def __init__(self, encoder: nn.Module, data_shape, out_features: int, hidden: int = 0):
        super().__init__()
        self.encoder = encoder
        dev = next((p.device for p in encoder.parameters()), torch.device("cpu"))
        with torch.no_grad():
            was = encoder.training
            encoder.eval()
            f = encoder(torch.zeros(2, *data_shape, device=dev))
            encoder.train(was)
        f = f.logits if hasattr(f, "logits") else f
        self.n_features = int(f.flatten(1).shape[1])
        self.head = nn.Linear(self.n_features, out_features) if not hidden else nn.Sequential(
            nn.Linear(self.n_features, hidden), nn.SiLU(), nn.Linear(hidden, out_features))

    def forward(self, x):
        f = self.encoder(x)
        f = f.logits if hasattr(f, "logits") else f
        return self.head(f.flatten(1))


def freeze(m: nn.Module) -> nn.Module:
    """Fix weights and keep the module in eval mode even when the parent calls .train()."""
    import types
    m.requires_grad_(False)
    m.eval()
    for sub in m.modules():
        sub._frozen = True
        sub.train = types.MethodType(lambda self, mode=True: nn.Module.train(self, False), sub)
    return m


def load_state_file(m: nn.Module, path: str) -> list[str]:
    """Initialize from a .pt/.pth file (state dict, or a checkpoint dict containing one). Returns notes."""
    sd = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "model_state_dict", "ema"):
        if isinstance(sd, dict) and isinstance(sd.get(k), dict):
            sd = sd[k]
            break
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    own = m.state_dict()
    for pre in ("", "module.", "net.", "model."):
        cand = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        ok = {k: v for k, v in cand.items() if k in own and own[k].shape == v.shape}
        if len(ok) >= 0.5 * len(own):
            m.load_state_dict(ok, strict=False)
            return [f"loaded {len(ok)}/{len(own)} tensors from {Path(path).name}"]
    raise ValueError(f"{Path(path).name} doesn't match this network's parameters "
                     f"(e.g. {next(iter(sd), '?')} vs {next(iter(own), '?')})")


# ============================================================================= slots
@dataclass
class Slot:
    name: str                      # also the form field name
    interface: str
    label: str
    builtin: Callable | None = None              # (params, info) -> nn.Module
    builtin_params: list[Param] = field(default_factory=list)
    builtin_label: str = "built-in"
    freeze_default: bool = False
    hub: bool = True                             # offer pretrained downloads for this interface
    help: str = ""
    trainable: bool = True                       # offer per-slot optimizer settings
    beta1: bool = False                          # also a per-slot Adam beta1 (GANs)

    # ------------------------------------------------------------------ form
    def options(self) -> tuple[list[str], dict, dict]:
        keys, labels, groups = [], {}, {}
        if self.builtin:
            keys.append("builtin")
            labels["builtin"] = self.builtin_label
            groups["builtin"] = "Built-in"
        for k, v in sorted(classes_for(self.interface).items()):
            o = f"class:{k}"
            keys.append(o)
            labels[o] = f"{v['cls'].__name__}  ({Path(v['path']).name})"
            groups[o] = "Your networks"
        for o, lab in trained_options(self.interface):
            keys.append(o)
            labels[o] = lab
            groups[o] = "Trained in this dashboard"
        if self.hub:
            for name, h in HUB.items():
                if h["interface"] == self.interface:
                    keys.append(f"hub:{name}")
                    labels[f"hub:{name}"] = h["label"]
                    groups[f"hub:{name}"] = "Pretrained (downloaded once)"
        return keys, labels, groups

    def params(self, group: str) -> list[Param]:
        keys, labels, groups = self.options()
        default = "builtin" if self.builtin else (keys[0] if keys else "")
        out = [Param(self.name, "choice", default, self.help or f"Where the {self.label} comes from — {INTERFACES[self.interface]}",
                     choices=keys or [""], labels=labels, option_groups=groups, group=group)]
        from dataclasses import replace
        out += [replace(q, group=group, show_if={"param": self.name, "in": ["builtin"]}) for q in self.builtin_params]
        cls_keys = [k for k in keys if k.startswith("class:")]
        for o in cls_keys:
            for q in _CLASSES[o[6:]]["params"]:
                out.append(replace(q, name=f"{self.name}.{o[6:]}.{q.name}", group=group, show_if={"param": self.name, "in": [o]}))
        if cls_keys:
            out.append(Param(f"{self.name}_weights", "str", "", "Optional: initialize your network from a .pt/.pth state dict",
                             group=group, show_if={"param": self.name, "in": cls_keys}))
        if self.builtin:
            out.append(Param(f"{self.name}_freeze", "bool", self.freeze_default, f"Freeze the {self.label} (no training)",
                             group=group, show_if={"param": self.name, "not_in": ["builtin"]}))
        else:
            out.append(Param(f"{self.name}_freeze", "bool", self.freeze_default, f"Freeze the {self.label} (no training)", group=group))
        if self.trainable:
            out.append(Param(f"{self.name}_lr", "float", None, f"Learning rate for the {self.label} (empty = the training lr)",
                             0.0, log=True, group=group))
            out.append(Param(f"{self.name}_weight_decay", "float", None, f"Weight decay for the {self.label} (empty = training value)",
                             0.0, group=group))
            if self.beta1:
                out.append(Param(f"{self.name}_beta1", "float", None, f"Adam beta1 for the {self.label} (empty = training value)",
                                 0.0, 0.999, group=group))
        return out

    # ------------------------------------------------------------------ build
    def build(self, p: dict, info: dict, inject: dict | None = None) -> tuple[nn.Module, dict]:
        """-> (module, meta). meta: source, note, plus facts from trained components (z_dim, data_shape, ...)."""
        from ..plugins.introspect import assemble
        from .core import inject_kwargs
        choice = p.get(self.name) or "builtin"
        meta: dict = {"source": choice}
        if choice == "builtin":
            if not self.builtin:
                raise ValueError(f"Choose a {self.label}")
            m = self.builtin(p, info)
        elif choice.startswith("class:"):
            key = choice[6:]
            if key not in _CLASSES:
                raise ValueError(f"{self.label}: network class '{key}' is no longer loaded (file removed or unlinked?)")
            cls = _CLASSES[key]["cls"]
            inj = inject_kwargs(inspect.signature(cls.__init__).parameters, info)
            inj.update(inject or {})
            m = cls(**assemble(cls, p, inj, prefix=f"{self.name}.{key}."))
            if p.get(f"{self.name}_weights"):
                meta["notes"] = load_state_file(m, p[f"{self.name}_weights"])
        elif choice.startswith("session:"):
            _, sid, part = choice.split(":", 2)
            m, src = load_session_component(sid, part)
            meta.update(src)
        elif choice.startswith("hub:"):
            m = load_hub(choice[4:], info)
        else:
            raise ValueError(f"Unknown {self.label} source '{choice}'")
        if choice != "builtin" and p.get(f"{self.name}_freeze", self.freeze_default):
            freeze(m)
            meta["frozen"] = True
        return m, meta


def slot_params(slots: list[Slot]) -> list[Param]:
    out = []
    for s in slots:
        out += s.params(f"{s.label[:1].upper()}{s.label[1:]}")
    return out


# ============================================================================= trained components
def trained_options(interface: str) -> list[tuple[str, str]]:
    """Saved generative sessions offering a part with this interface: [(option key, label)]."""
    try:
        from .. import db, storage
        from ..plugins import registry as R
        reg = R._registry
        if reg is None:             # called while plugins are still loading
            return []
        with db.get_conn() as c:
            rows = db.rows(c, "SELECT id, name, model_id, modality FROM sessions WHERE framework='generative' AND mode='train' "
                              "ORDER BY created_at DESC LIMIT 200")
            runs = db.rows(c, "SELECT session_id, id FROM runs WHERE status='completed'")
    except Exception:          # standalone export (no database) or registry still loading
        return []
    by = {}
    for r in runs:
        by.setdefault(r["session_id"], []).append(r["id"])
    out = []
    for s in rows:
        plugin = reg.plugins["model"].get(s["model_id"])
        parts = getattr(plugin, "COMPONENTS", {}) if plugin else {}
        rid = next((r for r in by.get(s["id"], []) if (storage.model_dir(r) / "generation.json").exists()), None)
        if not rid:
            continue
        try:
            shape = json.loads((storage.model_dir(rid) / "generation.json").read_text()).get("data_shape")
        except (OSError, json.JSONDecodeError):
            shape = None
        for part, iface in parts.items():
            if iface == interface:
                out.append((f"session:{s['id']}:{part}", f"{s['name']} — {part}" + (f" ({'x'.join(map(str, shape))})" if shape else "")))
    return out


def load_session_component(sid: str, part: str) -> tuple[nn.Module, dict]:
    """Rebuild a trained session's model (its best seed), load its weights and return one part."""
    from .. import analysis, db, storage
    from ..plugins.base import coerce_params
    from ..plugins.registry import get_registry
    with db.get_conn() as c:
        s = db.one(c, "SELECT * FROM sessions WHERE id=?", (sid,))
        runs = db.rows(c, "SELECT id, seed, summary FROM runs WHERE session_id=? AND status='completed'", (sid,)) if s else []
    if not s:
        raise ValueError(f"Session {sid} (component source) no longer exists")
    runs = [r for r in runs if (storage.model_dir(r["id"]) / "model.pt").exists()]
    if not runs:
        raise ValueError(f"Session '{s['name']}' has no saved model to take the {part} from")
    h = analysis.headline(s["task"], analysis.summary_stats(runs))
    best = runs[0]
    if h:
        scored = [r for r in runs if isinstance((r["summary"] or {}).get(h["key"]), (int, float))]
        if scored:
            best = (max if h["higher_is_better"] else min)(scored, key=lambda r: r["summary"][h["key"]])
    plugin = get_registry().get("model", s["model_id"])
    mdir = storage.model_dir(best["id"])
    info = json.loads((mdir / "generation.json").read_text())
    p = coerce_params(plugin.params(), s["model_params"])
    proc = plugin.build(p, info)
    plugin.load_weights(proc, mdir / "model.pt", torch.device("cpu"))
    parts = proc.components() if hasattr(proc, "components") else {}
    if part not in parts:
        raise ValueError(f"'{s['name']}' has no '{part}' component (has: {list(parts)})")
    m = parts[part]
    return m, {"session_id": sid, "session_name": s["name"], "seed": best["seed"], "data_shape": info.get("data_shape"),
               "z_dim": getattr(m, "in_features", None) or _in_features(m), "source_params": p}


def _in_features(m) -> int | None:
    fc = getattr(m, "fc", None)
    return getattr(fc, "in_features", None)


def copy_sharing_frozen(model: nn.Module) -> nn.Module:
    """deepcopy that shares (doesn't duplicate) frozen components, e.g. a 335 MB pretrained VAE."""
    import copy
    memo = {}
    for m in model.modules():
        if getattr(m, "_frozen", False):
            memo[id(m)] = m
            memo.update({id(q): q for q in m.parameters(recurse=False)})
            memo.update({id(b): b for b in m.buffers(recurse=False)})
    return copy.deepcopy(model, memo)


def external_prefixes(net: nn.Module) -> list[str]:
    return [n + "." for n, m in net.named_modules() if getattr(m, "_external", False) and getattr(m, "_frozen", False) and n]


def saveable_state(net: nn.Module) -> dict:
    """state_dict without frozen downloaded components (they're re-downloaded on load)."""
    pre = external_prefixes(net)
    return {k: v for k, v in net.state_dict().items() if not any(k.startswith(p) for p in pre)}


def check_shape(meta: dict, expected, what: str) -> None:
    got = meta.get("data_shape")
    if got and list(got) != list(expected):
        raise ValueError(f"The {what} from '{meta.get('session_name')}' was trained on {'x'.join(map(str, got))} data, "
                         f"but this needs {'x'.join(map(str, expected))} (match image_size / dataset)")
