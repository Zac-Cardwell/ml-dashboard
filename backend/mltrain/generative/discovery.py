"""Decide what each nn.Module class in a model file is, and wrap it in the right dashboard plugin(s).

    forward(x, t, ...)                       -> diffusion denoiser        ("Diffusion · <Class>")
    encode(x) + decode(z)                     -> VAE                       ("VAE · <Class>")
    *Encoder* / *Decoder* (unpaired)          -> components only (selectable in VAE / GAN / classifier slots)
    *Generator* + *Discriminator*/*Critic*    -> GAN (one per file)        ("GAN · G + D")
    losses(...) + sample(...)                 -> custom generative model   ("Custom · <Class>")
    __init__ takes vocab_size                 -> language model            ("Language model · <Class>")
    anything else                             -> classifier / regressor (the existing supervised wrapper)

Building blocks — classes another class in the same file constructs (ResidualBlock, Downsample, …) — are skipped.
Override with a class attribute `DASHBOARD_ROLE = "denoiser" | "vae" | "generator" | "discriminator" | "custom" | "lm" |
"supervised" | None` (None hides the class).
"""
from __future__ import annotations

import inspect
import re

T_NAMES = ("t", "time", "times", "timestep", "timesteps", "noise_level", "sigma", "sigmas", "t_emb")


def _defined(cls, name):
    return any(name in vars(k) for k in cls.__mro__ if k.__name__ not in ("Module", "object"))


def role(cls) -> str | None:
    if "DASHBOARD_ROLE" in vars(cls):
        return cls.DASHBOARD_ROLE
    if _defined(cls, "losses") and _defined(cls, "sample"):
        return "custom"
    if _defined(cls, "encode") and _defined(cls, "decode"):
        return "vae"
    try:
        fwd = [p for p in inspect.signature(cls.forward).parameters if p != "self"]
    except (TypeError, ValueError):
        fwd = []
    if len(fwd) >= 2 and fwd[1] in T_NAMES:
        return "denoiser"
    init = inspect.signature(cls.__init__).parameters
    if any(n in init for n in ("vocab_size", "n_vocab", "num_tokens")):
        return "lm"
    n = cls.__name__.lower()
    if "generator" in n:
        return "generator"
    if "discriminator" in n or "critic" in n:
        return "discriminator"
    if "decoder" in n:
        return "decoder"
    if "encoder" in n:
        return "encoder"
    return "supervised"


# which component slots each role can fill (see generative/components.py)
ROLE_INTERFACES = {"denoiser": ["denoiser"], "vae": ["autoencoder"], "generator": ["decoder"], "decoder": ["decoder"],
                   "discriminator": ["encoder"], "encoder": ["encoder"], "supervised": ["encoder"]}


def building_blocks(classes: list[type]) -> set[str]:
    """Names of classes instantiated inside another class of the same file."""
    srcs = {}
    for c in classes:
        try:
            srcs[c.__name__] = inspect.getsource(c)
        except (OSError, TypeError):
            srcs[c.__name__] = ""
    used = set()
    for c in classes:
        for other, src in srcs.items():
            if other != c.__name__ and re.search(rf"\b{re.escape(c.__name__)}\s*\(", src):
                used.add(c.__name__)
    return used


def plugins_for_file(classes: list[type], path) -> tuple[list, list[str]]:
    """-> (plugins, notes). `classes` are the nn.Module subclasses defined in the file."""
    from ..plugins.introspect import extract
    from .diffusion import diffusion_for
    from .gan import gan_for
    from .lm import lm_for
    from .vae import vae_for
    from .custom import custom_for
    from ..builtin.models_torch import wrap_module

    blocks = building_blocks(classes)
    out, notes = [], []
    roles = {}
    for c in classes:
        r = role(c)
        if r is None:
            continue
        if c.__name__ in blocks and "DASHBOARD_ROLE" not in vars(c):
            notes.append(f"{c.__name__}: building block (used inside another class) — not listed as a model")
            continue
        roles[c] = r
    from .components import register_class
    for c, r in roles.items():
        if ROLE_INTERFACES.get(r):
            register_class(c, ROLE_INTERFACES[r], path)
        if r in ("encoder", "decoder"):
            notes.append(f"{c.__name__}: available as {'an encoder' if r == 'encoder' else 'a decoder'} component "
                         f"(VAE / GAN / classifier slots)")
    gens = [c for c, r in roles.items() if r == "generator"]
    discs = [c for c, r in roles.items() if r == "discriminator"]
    if gens and discs:
        skip = {"z_dim", "latent_dim", "nz"}
        out.append(gan_for(gens[0], discs[0], path, extract(gens[0], "Generator", skip), extract(discs[0], "Discriminator")))
    for c, r in roles.items():
        if r == "denoiser":
            out.append(diffusion_for(c, path, extract(c, "Network")))
        elif r == "vae":
            out.append(vae_for(c, path, extract(c, "Network")))
        elif r == "custom":
            out.append(custom_for(c, path, extract(c, "Network")))
        elif r == "lm":
            out.append(lm_for(c, path, extract(c, "Network")))
        elif r == "supervised":
            out.append(wrap_module(c, path))
        elif r in ("generator", "discriminator") and not (gens and discs):
            notes.append(f"{c.__name__}: looks like a GAN {r} but the file has no matching "
                         f"{'Discriminator' if r == 'generator' else 'Generator'} class")
    return out, notes
