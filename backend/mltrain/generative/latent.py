"""Latent diffusion: a frozen autoencoder maps data to a compact latent, a denoiser learns diffusion there,
and samples are decoded back (Rombach et al. 2022, the Stable Diffusion recipe).

The autoencoder can be a pretrained Stable Diffusion VAE (TAESD / sd-vae-ft-mse), a VAE trained in this dashboard,
or your class with encode()/decode(). The denoiser is the built-in UNet (spatial latents) or MLP (vector latents),
your (x, t) network, or a denoiser from another latent-diffusion run with the same latent shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..plugins.registry import register
from .components import Slot, check_shape, freeze, slot_params
from .core import GenerativeModel
from .diffusion import DIFFUSION_PARAMS, DiffusionProcess, UNET_PARAMS, builtin_denoiser


class AE(nn.Module):
    """Uniform encode(x) -> latent tensor / decode(latent) -> x around any autoencoder-like module."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def encode(self, x):
        z = self.m.encode(x)
        return z[0] if isinstance(z, (tuple, list)) else z

    def decode(self, z):
        return self.m.decode(z)


class LatentDiffusionProcess(nn.Module):
    def __init__(self, ae: AE, diffusion: DiffusionProcess, scale: float | None):
        super().__init__()
        self.ae, self.diff = ae, diffusion
        self.register_buffer("z_scale", torch.tensor(float(scale or 0.0)))      # 0 = estimate from the first batch

    @torch.no_grad()
    def to_latent(self, x):
        z = self.ae.encode(x)
        if not self.z_scale.is_meta and float(self.z_scale) == 0.0:
            self.z_scale.fill_(1.0 / max(float(z.float().std()), 1e-6))
            print(f"latent scale estimated from the first batch: {float(self.z_scale):.4f}", flush=True)
        return z * self.z_scale

    @torch.no_grad()
    def from_latent(self, z):
        return self.ae.decode(z / self.z_scale).clamp(-1, 1)

    def losses(self, x, y=None, generator=None):
        return self.diff.losses(self.to_latent(x), y, generator=generator)

    @torch.no_grad()
    def sample(self, n, y=None, generator=None, fast=False, steps=None, sampler=None):
        return self.from_latent(self.diff.sample(n, y, generator, fast=fast, steps=steps, sampler=sampler))

    @torch.no_grad()
    def sample_trajectory(self, n, y=None, generator=None, frames=10):
        dev = self.z_scale.device
        return [self.from_latent(z.to(dev)).cpu() for z in self.diff.sample_trajectory(n, y, generator, frames)]

    @torch.no_grad()
    def reconstruct(self, x):
        return self.from_latent(self.to_latent(x))

    def components(self):
        return {"denoiser": self.diff.net, "autoencoder": self.ae}

    def slot_parameters(self):
        return {"denoiser": list(self.diff.net.parameters())}


AE_SLOT = Slot("autoencoder", "autoencoder", "autoencoder", builtin=None, freeze_default=True,
               help="Maps images to latents and back; always frozen. Pretrained Stable Diffusion VAEs need image sizes divisible by 8")
DEN_SLOT = Slot("denoiser", "denoiser", "denoiser", builtin=builtin_denoiser, builtin_params=UNET_PARAMS, hub=False,
                builtin_label="UNet / MLP (built-in, sized to the latent)")


@register
class LatentDiffusion(GenerativeModel):
    id = "gen.latent_diffusion"
    name = "Latent diffusion (autoencoder + denoiser)"
    description = "Diffusion in the latent space of a frozen autoencoder — pretrained SD VAE, a VAE trained here, or yours"
    COMPONENTS = {"denoiser": "denoiser", "autoencoder": "autoencoder"}
    modalities = ("image", "audio", "video")

    def method_params(self):
        return DIFFUSION_PARAMS

    def network_params(self):
        ps = slot_params([AE_SLOT, DEN_SLOT])
        for q in ps:
            if q.name == "autoencoder":
                q.default = "hub:taesd" if "hub:taesd" in (q.choices or []) else (q.choices or [""])[0]
            if q.name == "autoencoder_freeze":
                q.help = "The autoencoder is always frozen in latent diffusion"
        return ps

    def build(self, p, info):
        shape = info["data_shape"]
        p["autoencoder_freeze"] = True
        ae, ameta = AE_SLOT.build(p, info)
        check_shape(ameta, shape, "autoencoder")
        ae = AE(ae)
        freeze(ae)
        dev = next((q.device for q in ae.parameters()), torch.device("cpu"))
        with torch.no_grad():
            z = ae.encode(torch.zeros(2, *shape, device=dev))
        latent = list(z.shape[1:])
        print(f"latent shape {latent} (data {shape})", flush=True)
        linfo = {**info, "data_shape": latent}
        den, dmeta = DEN_SLOT.build(p, linfo)
        check_shape(dmeta, latent, "denoiser")
        diff = DiffusionProcess(den, latent, p["timesteps"], p["schedule"], p["beta_start"], p["beta_end"], p["cosine_s"],
                                p["prediction"], p["loss"], p["sampler"], p["sample_steps"], p["ddim_eta"], p["clip_x0"],
                                p["variance"], info.get("conditional", False))
        scale = getattr(ae.m, "scale", None)
        return LatentDiffusionProcess(ae, diff, scale)
