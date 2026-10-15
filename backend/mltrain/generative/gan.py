"""GANs: built-in DCGAN-style generator/discriminator (1-D/2-D/3-D) or your own Generator + Discriminator classes."""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..plugins.base import Param
from ..plugins.registry import register
from . import losses as L
from ..plugins.introspect import assemble, own_doc
from .components import Slot, WithHead, check_shape, slot_params
from .core import GenerativeModel, inject_kwargs
from .nets import ConvDecoder, ConvEncoder


ADVERSARIAL = {"non_saturating": ("bce", "bce"), "hinge": ("hinge", "hinge"), "wgan_gp": ("wasserstein", "wasserstein"),
               "least_squares": ("least_squares", "least_squares")}
BUILTIN_ADV = ["bce", "hinge", "wasserstein", "least_squares"]


def d_objective(kind: str, sr, sf):
    """Discriminator loss from scores on real (sr) and fake (sf) samples."""
    if kind == "bce":
        return F.softplus(-sr).mean() + F.softplus(sf).mean()
    if kind == "hinge":
        return F.relu(1 - sr).mean() + F.relu(1 + sf).mean()
    if kind == "wasserstein":
        return sf.mean() - sr.mean()
    if kind == "least_squares":
        return 0.5 * ((sr - 1) ** 2).mean() + 0.5 * (sf ** 2).mean()
    fn = L.get(kind)                                   # your loss: fn(scores, targets), real = 1, fake = 0
    return fn(sr, torch.ones_like(sr)) + fn(sf, torch.zeros_like(sf))


def g_objective(kind: str, sf):
    """Generator loss from the discriminator's scores on generated samples."""
    if kind == "bce":
        return F.softplus(-sf).mean()
    if kind in ("hinge", "wasserstein"):
        return -sf.mean()
    if kind == "least_squares":
        return 0.5 * ((sf - 1) ** 2).mean()
    return L.get(kind)(sf, torch.ones_like(sf))


class GANProcess(nn.Module):
    """Alternating updates: discriminator ("d") every step, generator ("g") every `n_critic` steps.
    The two networks can use different losses (and, via their slots, different learning rates / betas)."""

    def __init__(self, G: nn.Module, D: nn.Module, z_dim: int, adversarial="non_saturating", n_critic=1, gp_weight=10.0,
                 r1_weight=0.0, d_loss: str | None = None, g_loss: str | None = None):
        super().__init__()
        self.G, self.D, self.z_dim = G, D, z_dim
        self.adv, self.n_critic, self.r1_weight = adversarial, max(1, n_critic), r1_weight
        if adversarial == "custom":
            self.d_kind, self.g_kind = d_loss or "bce", g_loss or "bce"
            self.gp_weight = gp_weight if self.d_kind == "wasserstein" else 0.0
        else:
            self.d_kind, self.g_kind = ADVERSARIAL[adversarial]
            self.gp_weight = gp_weight if adversarial == "wgan_gp" else 0.0
        self.no_amp = self.gp_weight > 0 or r1_weight > 0      # gradient penalties need fp32 double-backward

    def optimizer_groups(self):
        return {"d": list(self.D.parameters()), "g": list(self.G.parameters())}

    def slot_parameters(self):
        return {"discriminator": list(self.D.parameters()), "generator": list(self.G.parameters())}

    def update_groups(self, step):
        return ["d", "g"] if step % self.n_critic == 0 else ["d"]

    def _z(self, n, device, generator=None):
        return torch.randn((n, self.z_dim), device=device, generator=generator)

    def _d(self, x):
        return self.D(x).reshape(len(x), -1).mean(1)

    def losses(self, x, y=None, group="d", generator=None):
        if not self.training:      # validation: report both losses without updating anything
            with torch.enable_grad():
                d = self._d_loss(x, generator)
            g = self._g_loss(len(x), x.device, generator)
            return {"loss": d["loss"].detach() + g["loss"].detach(), "d_loss": d["loss"].detach(), "g_loss": g["loss"].detach(),
                    "d_real": d["d_real"], "d_fake": d["d_fake"]}
        return self._d_loss(x, generator) if group == "d" else self._g_loss(len(x), x.device, generator)

    def _d_loss(self, x, generator=None):
        fake = self.G(self._z(len(x), x.device, generator)).detach()
        if self.r1_weight:
            x = x.detach().requires_grad_(True)
        dr, df = self._d(x), self._d(fake)
        loss = d_objective(self.d_kind, dr, df)
        if self.gp_weight:
            a = torch.rand((len(x),) + (1,) * (x.dim() - 1), device=x.device)
            mix = (a * x.detach() + (1 - a) * fake).requires_grad_(True)
            gr = torch.autograd.grad(self._d(mix).sum(), mix, create_graph=True)[0]
            loss = loss + self.gp_weight * ((gr.flatten(1).norm(dim=1) - 1) ** 2).mean()
        if self.r1_weight:
            gr = torch.autograd.grad(dr.sum(), x, create_graph=True)[0]
            loss = loss + self.r1_weight / 2 * gr.pow(2).flatten(1).sum(1).mean()
        return {"loss": loss, "d_real": dr.mean().detach(), "d_fake": df.mean().detach()}

    def _g_loss(self, n, device, generator=None):
        return {"loss": g_objective(self.g_kind, self._d(self.G(self._z(n, device, generator))))}

    @torch.no_grad()
    def sample(self, n, y=None, generator=None):
        return self.G(self._z(n, next(self.parameters()).device, generator)).clamp(-1, 1)

    def components(self):
        self.G.in_features = self.z_dim
        return {"generator": self.G, "discriminator": self.D}


GAN_PARAMS = [
    Param("z_dim", "int", 128, "Latent noise size", 1, 4096, group="GAN"),
    Param("adversarial", "choice", "non_saturating", "Adversarial objective; 'custom' picks the discriminator and generator "
          "losses separately", choices=["non_saturating", "hinge", "wgan_gp", "least_squares", "custom"], group="GAN"),
    Param("discriminator_loss", "str", "hinge", "Discriminator loss: bce, hinge, wasserstein, least_squares, or your "
          "loss fn(scores, targets) with real = 1, fake = 0", suggestions=BUILTIN_ADV, group="GAN",
          show_if={"param": "adversarial", "in": ["custom"]}),
    Param("generator_loss", "str", "bce", "Generator loss: bce (non-saturating), hinge, wasserstein, least_squares, or your "
          "loss fn(scores, targets) with target = 1", suggestions=BUILTIN_ADV, group="GAN",
          show_if={"param": "adversarial", "in": ["custom"]}),
    Param("n_critic", "int", 1, "Discriminator steps per generator step (5 is typical for WGAN-GP)", 1, 20, group="GAN"),
    Param("gp_weight", "float", 10.0, "Gradient-penalty weight (WGAN-GP, or custom with a wasserstein discriminator loss)", 0.0,
          group="GAN", show_if={"param": "adversarial", "in": ["wgan_gp", "custom"]}),
    Param("r1_weight", "float", 0.0, "R1 penalty on real data (0 = off; stabilizes non-saturating GANs)", 0.0, group="GAN"),
]


class _GANBase(GenerativeModel):
    COMPONENTS = {"generator": "decoder", "discriminator": "encoder"}
    default_monitor = "none"     # GAN losses don't track sample quality: keep the final weights (or set fid_every)

    def method_params(self):
        return GAN_PARAMS + self._optim_params() + [Param("fid_every", "int", 0, "Compute FID on 500 samples every N epochs and keep the best-FID "
                                                         "weights (0 = keep final weights)", 0, group="GAN")]

    def params(self):
        from dataclasses import replace
        overrides = {"beta1": 0.5, "ema_decay": 0.0, "grad_clip": 0.0}     # GAN-friendly defaults (copies, not the shared Params)
        return [replace(q, default=overrides[q.name]) if q.name in overrides else q for q in super().params()]

    def _optim_params(self):
        """Per-network optimizer settings for GANs whose networks aren't slots (your Generator/Discriminator classes)."""
        if any(q.name == "generator_lr" for q in self.network_params()):
            return []
        out = []
        for net in ("generator", "discriminator"):
            out += [Param(f"{net}_lr", "float", None, f"{net} learning rate (empty = the training lr)", 0.0, log=True, group="GAN"),
                    Param(f"{net}_beta1", "float", None, f"{net} Adam beta1 (empty = training value)", 0.0, 0.999, group="GAN")]
        return out

    def build(self, p, info):
        if p.get("fid_every"):
            p["monitor_metric"] = "val/fid"
        G, D = self.make_nets(p, info)
        if p["adversarial"] == "custom":
            for k in ("discriminator_loss", "generator_loss"):
                if p[k] not in BUILTIN_ADV:
                    L.get(p[k])            # fail early with the list of available losses
        return GANProcess(G, D, p["z_dim"], p["adversarial"], p["n_critic"], p["gp_weight"], p["r1_weight"],
                          p.get("discriminator_loss"), p.get("generator_loss"))


def _g_builtin(p, info):
    return ConvDecoder(info["data_shape"], p["z_dim"], p["g_channels"], max(256, p["g_channels"] * 8))


def _d_builtin(p, info):
    return ConvEncoder(info["data_shape"], 1, p["d_channels"], max(256, p["d_channels"] * 8), spectral_norm=p["spectral_norm"],
                       use_norm=not p["spectral_norm"])


G_SLOT = Slot("generator", "decoder", "generator", builtin=_g_builtin, hub=False, beta1=True, builtin_label="DCGAN-style (built-in)",
              builtin_params=[Param("g_channels", "int", 64, "Generator base channels", 4, 1024)],
              help="z -> sample. A VAE decoder or GAN generator trained here can start (or stay) as the generator; z_dim follows it")
D_SLOT = Slot("discriminator", "encoder", "discriminator", builtin=_d_builtin, beta1=True, builtin_label="spectral-norm conv (built-in)",
              builtin_params=[Param("d_channels", "int", 64, "Discriminator base channels", 4, 1024),
                              Param("spectral_norm", "bool", True, "Spectral normalization in the discriminator")],
              help="x -> realness score. Any encoder works (a linear head is added): e.g. a frozen ImageNet ResNet "
                   "(projected-GAN style) or a trained VAE encoder")


@register
class BuiltinGAN(_GANBase):
    id = "gen.gan"
    name = "GAN · DCGAN (built-in)"
    description = "Generator and discriminator are interchangeable: built-in, yours, trained here, or a pretrained encoder as D"

    def network_params(self):
        return slot_params([G_SLOT, D_SLOT])

    def make_nets(self, p, info):
        shape = info["data_shape"]
        G, gmeta = G_SLOT.build(p, info, {"z_dim": p["z_dim"], "latent_dim": p["z_dim"], "nz": p["z_dim"]})
        if p.get("generator", "builtin") != "builtin":
            check_shape(gmeta, shape, "generator")
            if gmeta.get("z_dim") and gmeta["z_dim"] != p["z_dim"]:
                print(f"z_dim set to {gmeta['z_dim']} to match the generator", flush=True)
                p["z_dim"] = gmeta["z_dim"]
        D, dmeta = D_SLOT.build(p, info)
        if p.get("discriminator", "builtin") != "builtin":
            check_shape(dmeta, shape, "discriminator")
            D = WithHead(D, shape, 1)
        return G, D


def gan_for(gen_cls, disc_cls, path, g_params, d_params):
    """A GAN plugin from a user file's Generator(z) and Discriminator(x) classes (params prefixed G_ / D_)."""
    gsig = inspect.signature(gen_cls.__init__).parameters
    dsig = inspect.signature(disc_cls.__init__).parameters
    for q in g_params:
        q.name, q.group = f"G_{q.name}", "Generator"
    for q in d_params:
        q.name, q.group = f"D_{q.name}", "Discriminator"

    class _G(_GANBase):
        id = f"gen.gan.{path.stem}"
        name = f"GAN · {gen_cls.__name__} + {disc_cls.__name__}"
        description = (own_doc(gen_cls) or f"Your GAN from {path.name}").split("\n")[0]
        network_class = gen_cls
        extra_classes = (disc_cls,)

        def network_params(self):
            return g_params + d_params

        def make_nets(self, p, info):
            gi = inject_kwargs(gsig, info)
            for n in ("z_dim", "latent_dim", "nz"):
                if n in gsig:
                    gi[n] = p["z_dim"]
            return gen_cls(**assemble(gen_cls, p, gi, "G_")), disc_cls(**assemble(disc_cls, p, inject_kwargs(dsig, info), "D_"))

    _G.__name__ = f"GAN_{path.stem}"
    return _G()
