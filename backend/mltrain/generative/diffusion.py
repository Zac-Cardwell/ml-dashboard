"""Denoising diffusion (DDPM training; DDPM or DDIM sampling) around any denoiser network `net(x_t, t[, y])`.

Schedules follow Ho et al. 2020 (linear) and Nichol & Dhariwal 2021 (cosine) — numerically identical to the
DDPM research project in /mnt/Linux_Data_2/Projects/Machine_Learning/Research Projects/ddpm (noise.py), so its
checkpoints can be evaluated or fine-tuned here.
"""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn

from ..plugins.base import Param
from ..plugins.registry import register
from . import losses as L
from ..plugins.introspect import assemble, own_doc
from .core import GenerativeModel, inject_kwargs
from .components import Slot, check_shape, slot_params
from .nets import UNet


def beta_schedule(T: int, kind: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02, s: float = 0.008):
    if kind == "linear":
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        return betas, torch.cumprod(1 - betas, 0)
    if kind == "cosine":
        t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        abar = f / f[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        return betas, abar[1:]
    raise ValueError(f"unknown schedule {kind!r}")


class DiffusionProcess(nn.Module):
    def __init__(self, net: nn.Module, data_shape, T=1000, schedule="linear", beta_start=1e-4, beta_end=0.02, cosine_s=0.008,
                 prediction="eps", loss="mse", sampler="ddpm", sample_steps=100, eta=0.0, clip_x0=False, variance="beta",
                 conditional=False):
        super().__init__()
        self.net, self.shape, self.T = net, tuple(data_shape), T
        betas, abar = beta_schedule(T, schedule, beta_start, beta_end, cosine_s)
        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("abar", abar.float(), persistent=False)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        self.register_buffer("post_var", (betas * (1 - abar_prev) / (1 - abar)).float(), persistent=False)
        self.prediction, self.sampler, self.sample_steps, self.eta = prediction, sampler, sample_steps, eta
        self.clip_x0, self.variance = clip_x0, variance
        self.loss_fn = L.get(loss)
        self.conditional = conditional
        self._takes_y = "y" in inspect.signature(net.forward).parameters

    def _bc(self, v, x):
        return v.view(-1, *([1] * (x.dim() - 1)))

    def model_out(self, x, t, y=None):
        return self.net(x, t, y) if (self.conditional and self._takes_y and y is not None) else self.net(x, t)

    def losses(self, x0, y=None, generator=None):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        a = self._bc(self.abar[t], x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        out = self.model_out(xt, t, y)
        target = {"eps": noise, "x0": x0, "v": a.sqrt() * noise - (1 - a).sqrt() * x0}[self.prediction]
        loss = self.loss_fn(out.float(), target.float())
        return {"loss": loss}

    def _eps_x0(self, out, xt, a):
        if self.prediction == "eps":
            eps = out
            x0 = (xt - (1 - a).sqrt() * eps) / a.sqrt()
        elif self.prediction == "x0":
            x0 = out
            eps = (xt - a.sqrt() * x0) / (1 - a).sqrt()
        else:
            x0 = a.sqrt() * xt - (1 - a).sqrt() * out
            eps = a.sqrt() * out + (1 - a).sqrt() * xt
        if self.clip_x0:
            x0 = x0.clamp(-1, 1)
            eps = (xt - a.sqrt() * x0) / (1 - a).sqrt()
        return eps, x0

    @torch.no_grad()
    def _ddpm(self, x, y, generator, record=None):
        for t in reversed(range(self.T)):
            tb = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            a, beta = self._bc(self.abar[tb], x), self._bc(self.betas[tb], x)
            eps, x0 = self._eps_x0(self.model_out(x, tb, y), x, a)
            mean = (x - beta / (1 - a).sqrt() * eps) / (1 - beta).sqrt()
            if t > 0:
                var = beta if self.variance == "beta" else self._bc(self.post_var[tb], x)
                x = mean + var.sqrt() * torch.randn(x.shape, device=x.device, generator=generator)
            else:
                x = mean
            if record is not None and (t % max(1, self.T // record[0]) == 0 or t == 0):
                record[1].append(x.clamp(-1, 1).cpu())
        return x

    @torch.no_grad()
    def _ddim(self, x, y, generator, steps, record=None):
        ts = torch.linspace(self.T - 1, 0, steps, device=x.device).round().long().unique_consecutive()
        for i, t in enumerate(ts):
            tb = t.repeat(x.shape[0])
            a = self._bc(self.abar[tb], x)
            a_prev = self._bc(self.abar[ts[i + 1]].repeat(x.shape[0]), x) if i + 1 < len(ts) else torch.ones_like(a)
            eps, x0 = self._eps_x0(self.model_out(x, tb, y), x, a)
            sigma = self.eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).clamp(min=0).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if self.eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn(x.shape, device=x.device, generator=generator)
            if record is not None and (i % max(1, len(ts) // record[0]) == 0 or i == len(ts) - 1):
                record[1].append(x.clamp(-1, 1).cpu())
        return x

    @torch.no_grad()
    def sample(self, n, y=None, generator=None, fast=False, steps=None, sampler=None):
        dev = self.abar.device
        x = torch.randn((n, *self.shape), device=dev, generator=generator)
        kind = sampler or ("ddim" if fast else self.sampler)
        if kind == "ddim":
            x = self._ddim(x, y, generator, steps or self.sample_steps)
        else:
            x = self._ddpm(x, y, generator)
        return x.clamp(-1, 1)

    def components(self):
        return {"denoiser": self.net}

    def slot_parameters(self):
        return {"denoiser": list(self.net.parameters())}

    @torch.no_grad()
    def sample_trajectory(self, n, y=None, generator=None, frames=10):
        x = torch.randn((n, *self.shape), device=self.abar.device, generator=generator)
        rec = (frames, [x.clamp(-1, 1).cpu()])
        if self.sampler == "ddim":
            self._ddim(x, y, generator, self.sample_steps, rec)
        else:
            self._ddpm(x, y, generator, rec)
        return rec[1]


DIFFUSION_PARAMS = [
    Param("timesteps", "int", 1000, "Diffusion steps T", 10, 10000, group="Diffusion"),
    Param("schedule", "choice", "linear", "Noise schedule", choices=["linear", "cosine"], group="Diffusion"),
    Param("beta_start", "float", 1e-4, "Linear schedule: first beta", 0.0, 1.0, group="Diffusion"),
    Param("beta_end", "float", 0.02, "Linear schedule: last beta", 0.0, 1.0, group="Diffusion"),
    Param("cosine_s", "float", 0.008, "Cosine schedule offset s", 0.0, 1.0, group="Diffusion"),
    Param("prediction", "choice", "eps", "What the network predicts: noise (eps), the clean sample (x0) or velocity (v)",
          choices=["eps", "x0", "v"], group="Diffusion"),
    Param("loss", "str", "mse", "Loss between prediction and target (built-in or your own from user_plugins/losses)",
          suggestions=["mse", "l1", "huber", "charbonnier"], group="Diffusion"),
    Param("sampler", "choice", "ddpm", "ddpm = all T steps (as in the paper); ddim = fewer, deterministic steps",
          choices=["ddpm", "ddim"], group="Sampling"),
    Param("sample_steps", "int", 100, "DDIM steps (also used for the FID evaluation samples)", 1, 10000, group="Sampling"),
    Param("ddim_eta", "float", 0.0, "DDIM stochasticity (0 = deterministic, 1 = DDPM-like)", 0.0, 1.0, group="Sampling"),
    Param("clip_x0", "bool", False, "Clamp the predicted clean sample to [-1, 1] while sampling", group="Sampling"),
    Param("variance", "choice", "beta", "DDPM sampling variance: beta_t (paper, Algorithm 2) or the posterior variance",
          choices=["beta", "posterior"], group="Sampling"),
    Param("class_conditional", "bool", False, "Condition on dataset labels (networks with a `y` argument)", group="Diffusion"),
]


class DiffusionModel(GenerativeModel):
    """Base: subclasses provide `make_net(p, info)`."""
    COMPONENTS = {"denoiser": "denoiser"}

    def method_params(self):
        return DIFFUSION_PARAMS

    def make_net(self, p, info) -> nn.Module:
        raise NotImplementedError

    def build(self, p, info):
        return DiffusionProcess(self.make_net(p, info), info["data_shape"], p["timesteps"], p["schedule"], p["beta_start"],
                                p["beta_end"], p["cosine_s"], p["prediction"], p["loss"], p["sampler"], p["sample_steps"],
                                p["ddim_eta"], p["clip_x0"], p["variance"], info.get("conditional", False))


UNET_PARAMS = [Param("base_channels", "int", 64, "Channels at full resolution", 8, 512, group="Network"),
               Param("channel_mult", "int_list", [1, 2, 2], "Channel multiplier per resolution level", group="Network"),
               Param("num_res_blocks", "int", 2, "Residual blocks per level", 1, 8, group="Network"),
               Param("attention_levels", "int_list", [2], "Levels with self-attention (images only; 0 = full res)", group="Network"),
               Param("dropout", "float", 0.1, min=0.0, max=0.9, group="Network")]


def builtin_denoiser(p, info):
    """UNet for spatial data (or spatial latents); MLP for vector latents."""
    from .nets import MLPDenoiser
    shape = info["data_shape"]
    ncls = info.get("n_classes", 0) if info.get("conditional") else 0
    if len(shape) == 1:
        return MLPDenoiser(shape[0], max(256, p["base_channels"] * 4), 4, ncls)
    return UNet(shape[0], len(shape) - 1, p["base_channels"], tuple(p["channel_mult"]), p["num_res_blocks"],
                tuple(p["attention_levels"]), p["dropout"], ncls)


def denoiser_slot(hub=True):
    return Slot("denoiser", "denoiser", "denoiser", builtin=builtin_denoiser, builtin_params=UNET_PARAMS,
                builtin_label="UNet (built-in, configurable)", hub=hub)


@register
class BuiltinDiffusion(DiffusionModel):
    id = "gen.diffusion"
    name = "Diffusion · UNet (built-in)"
    description = "DDPM / DDIM; the denoiser can be the built-in UNet, your network, a trained one or a pretrained DDPM"

    def network_params(self):
        return slot_params([denoiser_slot()])

    def make_net(self, p, info):
        m, meta = denoiser_slot().build(p, info)
        check_shape(meta, info["data_shape"], "denoiser")
        if meta.get("notes"):
            print(*meta["notes"], flush=True)
        return m


def diffusion_for(module_cls, path, net_params, source_note=""):
    """A diffusion plugin around a user denoiser class with forward(x, t[, y])."""
    sig = inspect.signature(module_cls.__init__).parameters

    class _D(DiffusionModel):
        id = f"gen.diffusion.{path.stem}.{module_cls.__name__}"
        name = f"Diffusion · {module_cls.__name__}"
        description = (own_doc(module_cls) or f"Your denoiser from {path.name}").split("\n")[0] + " — trained as a DDPM"
        network_class = module_cls

        def network_params(self):
            return net_params

        def make_net(self, p, info):
            return module_cls(**assemble(module_cls, p, inject_kwargs(sig, info)))

    _D.__name__ = f"Diffusion_{module_cls.__name__}"
    return _D()
