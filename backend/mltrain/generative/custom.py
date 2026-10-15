"""Fully custom generative models: your nn.Module implements the training contract itself.

    class MyModel(nn.Module):
        MODALITIES = ("image",)                      # optional: image | audio | video
        def __init__(self, in_channels, latent=64): ...      # in_channels etc. are filled from the dataset
        def losses(self, x, y=None):                 # x: a batch in [-1, 1]
            ...
            return {"loss": total, "recon": r, "adv": a}      # "loss" is optimized, the rest are charted
        def sample(self, n, y=None, generator=None):  # -> tensor shaped like x, in [-1, 1]
            ...
        # optional: optimizer_groups() / update_groups(step) for several optimizers (GAN-style) — declare
        #           OPTIMIZER_GROUPS = ("g", "d") to get an lr field per group,
        #           sample_trajectory(n, generator=None, frames=10), reconstruct(x), set_progress(fraction)

The dashboard supplies data loading, devices, AMP, gradient accumulation, EMA, schedules, checkpoints (best +
final, resumable), sample snapshots, FID/KID and multi-seed runs.
"""
from __future__ import annotations

import inspect

from ..plugins.base import Param
from ..plugins.introspect import assemble, own_doc
from .core import GenerativeModel, inject_kwargs


def custom_for(module_cls, path, net_params):
    sig = inspect.signature(module_cls.__init__).parameters

    class _C(GenerativeModel):
        id = f"gen.custom.{path.stem}.{module_cls.__name__}"
        name = f"Custom · {module_cls.__name__}"
        description = (own_doc(module_cls) or f"Your generative model from {path.name}").split("\n")[0]
        modalities = tuple(getattr(module_cls, "MODALITIES", ("image", "audio", "video")))
        network_class = module_cls

        def network_params(self):
            # optional OPTIMIZER_GROUPS = ("g", "d") on your class: one learning-rate field per optimizer group
            groups = getattr(module_cls, "OPTIMIZER_GROUPS", ())
            return net_params + [Param(f"lr_{g}", "float", None, f"Learning rate for optimizer group '{g}' (empty = training lr)",
                                       0.0, log=True, group="Optimizers") for g in groups]

        def build(self, p, info):
            return module_cls(**assemble(module_cls, p, inject_kwargs(sig, info)))

    _C.__name__ = f"Custom_{module_cls.__name__}"
    return _C()
