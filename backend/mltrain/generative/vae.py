"""Variational autoencoders: built-in conv encoder/decoder (1-D/2-D/3-D) or your own network with encode()/decode()."""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn

from ..plugins.base import Param
from ..plugins.registry import register
from . import losses as L
from ..plugins.introspect import assemble, own_doc
from .components import MethodModule, Slot, WithHead, check_shape, slot_params
from .core import GenerativeModel, inject_kwargs
from .nets import ConvDecoder, ConvEncoder


class VAEProcess(nn.Module):
    """ELBO = reconstruction (mean over values) + beta * KL / values-per-sample, with optional KL warm-up."""

    def __init__(self, encode, decode, latent_dim: int, data_shape, loss="mse", beta=1.0, kl_warmup=0.1, modules=()):
        super().__init__()
        self.parts = nn.ModuleList(modules)
        self._encode, self._decode = encode, decode
        self.latent_dim, self.shape = latent_dim, tuple(data_shape)
        self.loss_fn, self.beta, self.kl_warmup = L.get(loss), beta, kl_warmup
        self.values = float(math.prod(self.shape))
        self.progress = 1.0

    def set_progress(self, f):
        self.progress = f

    def encode(self, x):
        h = self._encode(x)
        if isinstance(h, (tuple, list)):
            mu, logvar = h[0], h[1]
        else:
            mu, logvar = h.chunk(2, -1)
        return mu, logvar.clamp(-30, 20)

    def losses(self, x, y=None, generator=None):
        mu, logvar = self.encode(x)
        z = mu + (0.5 * logvar).exp() * torch.randn(mu.shape, device=mu.device, generator=generator)
        rec = self._decode(z)
        recon = self.loss_fn(rec.float(), x.float())
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).flatten(1).sum(1)).mean()
        w = min(1.0, self.progress / self.kl_warmup) if self.kl_warmup and self.training else 1.0
        loss = recon + self.beta * w * kl / self.values
        return {"loss": loss, "recon": recon, "kl": kl}

    @torch.no_grad()
    def sample(self, n, y=None, generator=None):
        dev = next(self.parameters()).device
        z = torch.randn((n, self.latent_dim), device=dev, generator=generator)
        return self._decode(z).clamp(-1, 1)

    @torch.no_grad()
    def reconstruct(self, x):
        return self._decode(self.encode(x)[0]).clamp(-1, 1)

    def decode(self, z):
        return self._decode(z)

    def slot_parameters(self):
        return {"encoder": list(self.parts[0].parameters()), "decoder": list(self.parts[1].parameters())} if len(self.parts) == 2 else {}

    def components(self):
        """Reusable parts: encoder (x -> 2*latent features), decoder (z -> x), the whole VAE as an autoencoder."""
        if len(self.parts) == 2:
            enc, dec = self.parts[0], self.parts[1]
        else:
            enc, dec = MethodModule(self.parts[0], "encode", first=True), MethodModule(self.parts[0], "decode")
        dec.in_features = self.latent_dim
        return {"encoder": enc, "decoder": dec, "autoencoder": self}


VAE_PARAMS = [
    Param("latent_dim", "int", 32, "Size of the latent vector z", 1, 4096, group="VAE"),
    Param("loss", "str", "mse", "Reconstruction loss (built-in or your own)", suggestions=["mse", "l1", "bce", "huber"], group="VAE"),
    Param("beta", "float", 1.0, "KL weight (beta-VAE; >1 = more disentangled, blurrier)", 0.0, group="VAE"),
    Param("kl_warmup", "float", 0.1, "Fraction of training over which the KL weight ramps from 0 to beta", 0.0, 1.0, group="VAE"),
]


def _enc_builtin(p, info):
    return ConvEncoder(info["data_shape"], 2 * p["latent_dim"], p["base_channels"], p["max_channels"])


def _dec_builtin(p, info):
    return ConvDecoder(info["data_shape"], p["latent_dim"], p["base_channels"], p["max_channels"])


_NET = [Param("base_channels", "int", 32, "Channels after the first conv", 4, 512),
        Param("max_channels", "int", 256, "Channel cap for deep layers", 8, 2048)]
ENC_SLOT = Slot("encoder", "encoder", "encoder", builtin=_enc_builtin, builtin_params=_NET,
                builtin_label="strided conv (built-in)",
                help="Maps x to features; a trainable linear head turns them into the mean and log-variance of z")
DEC_SLOT = Slot("decoder", "decoder", "decoder", builtin=_dec_builtin, builtin_params=_NET,
                builtin_label="upsample + conv (built-in)", hub=False,
                help="Maps z back to data; a decoder from a trained model fixes the latent size to its input size")


@register
class BuiltinVAE(GenerativeModel):
    id = "gen.vae"
    name = "VAE · conv (built-in)"
    description = "Variational autoencoder; encoder and decoder are interchangeable (built-in, yours, trained, ImageNet encoders)"
    COMPONENTS = {"encoder": "encoder", "decoder": "decoder", "autoencoder": "autoencoder"}

    def method_params(self):
        return VAE_PARAMS

    def network_params(self):
        from dataclasses import replace
        ps = slot_params([ENC_SLOT, DEC_SLOT])
        seen, out = set(), []
        for q in ps:   # base/max_channels are shared by the built-in encoder and decoder: show once
            if q.name in ("base_channels", "max_channels"):
                if q.name in seen:
                    continue
                seen.add(q.name)
                q = replace(q, group="Network", show_if=None)
            out.append(q)
        return out

    def build(self, p, info):
        shape = info["data_shape"]
        dec, dmeta = DEC_SLOT.build(p, info, {"latent_dim": p["latent_dim"], "z_dim": p["latent_dim"]})
        if p.get("decoder", "builtin") != "builtin":
            check_shape(dmeta, shape, "decoder")
            z = dmeta.get("z_dim") or _decoder_in(dec)
            if z and z != p["latent_dim"]:
                print(f"latent_dim set to {z} to match the decoder", flush=True)
                p["latent_dim"] = z
        enc, emeta = ENC_SLOT.build(p, info)
        if p.get("encoder", "builtin") != "builtin":
            check_shape(emeta, shape, "encoder")
            enc = WithHead(enc, shape, 2 * p["latent_dim"])
        return VAEProcess(enc.forward, dec.forward, p["latent_dim"], shape, p["loss"], p["beta"], p["kl_warmup"], (enc, dec))


def _decoder_in(m):
    fc = getattr(m, "fc", None)
    if isinstance(fc, nn.Linear):
        return fc.in_features
    first = next((x for x in m.modules() if isinstance(x, nn.Linear)), None)
    return first.in_features if first is not None else None


def vae_for(module_cls, path, net_params):
    """A VAE plugin around a user module with encode(x) -> (mu, logvar) and decode(z) -> x."""
    sig = inspect.signature(module_cls.__init__).parameters
    has_latent = any(n in sig for n in ("latent_dim", "z_dim", "latent_size"))

    class _V(GenerativeModel):
        COMPONENTS = {"encoder": "encoder", "decoder": "decoder", "autoencoder": "autoencoder"}
        id = f"gen.vae.{path.stem}.{module_cls.__name__}"
        name = f"VAE · {module_cls.__name__}"
        description = (own_doc(module_cls) or f"Your VAE from {path.name}").split("\n")[0]
        network_class = module_cls

        def method_params(self):
            ps = [q for q in VAE_PARAMS if not (q.name == "latent_dim" and has_latent)]
            return ps

        def network_params(self):
            return net_params

        def build(self, p, info):
            kw = assemble(module_cls, p, inject_kwargs(sig, info))
            m = module_cls(**kw)
            latent = next((kw[n] for n in ("latent_dim", "z_dim", "latent_size") if n in kw), None) or p.get("latent_dim")
            latent = getattr(m, "latent_dim", latent)
            return VAEProcess(m.encode, m.decode, latent, info["data_shape"], p["loss"], p["beta"], p["kl_warmup"], (m,))

    _V.__name__ = f"VAE_{module_cls.__name__}"
    return _V()
