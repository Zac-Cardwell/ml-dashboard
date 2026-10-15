"""Built-in generative networks that work for 1-D (audio), 2-D (images) and 3-D (video) data.

`dims` selects Conv1d/2d/3d. Spatial sizes that aren't powers of two are handled by padding up to the next
multiple the network needs and cropping back at the end, so e.g. 28x28 MNIST works unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(dims: int):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]


def convT(dims: int):
    return {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[dims]


def norm(ch: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


def _pad_to(x: torch.Tensor, multiple: int, spatial_from: int = 2):
    """Pad trailing spatial dims (time is never padded for video: spatial_from skips it) to a multiple."""
    sizes = x.shape[spatial_from:]
    pads = []
    for s in reversed(sizes):
        pads += [0, (-s) % multiple]
    if any(pads):
        x = F.pad(x, pads, mode="replicate" if x.dim() <= 4 else "constant")
    return x, tuple(sizes)


def _crop(x: torch.Tensor, sizes: tuple, spatial_from: int = 2) -> torch.Tensor:
    idx = [slice(None)] * spatial_from + [slice(0, s) for s in sizes]
    return x[tuple(idx)]


# ============================================================================= time embedding
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -(math.log(10000) / max(half - 1, 1)))
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, dims, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        C = conv(dims)
        self.n1, self.c1 = norm(cin), C(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout) if emb_dim else None
        self.n2, self.c2 = norm(cout), C(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = C(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.emb is not None and emb is not None:
            scale, shift = self.emb(F.silu(emb)).view(*emb.shape[:1], -1, *([1] * (h.dim() - 2))).chunk(2, 1)
            h = h * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n = norm(ch)
        self.attn = nn.MultiheadAttention(ch, heads if ch % heads == 0 else 1, batch_first=True)

    def forward(self, x):
        b, c = x.shape[:2]
        s = x.shape[2:]
        h = self.n(x).reshape(b, c, -1).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).reshape(b, c, *s)


# ============================================================================= UNet (diffusion denoiser)
class UNet(nn.Module):
    """Time-conditioned UNet for diffusion, in 1-D, 2-D or 3-D.

    Args:
        in_channels: data channels (injected from the dataset)
        dims: 1 = audio, 2 = images, 3 = video (injected)
        base_channels: channels at the highest resolution
        channel_mult: channel multiplier per resolution level (one downsample between levels)
        num_res_blocks: residual blocks per level
        attention_levels: levels (0 = highest resolution) that get self-attention
        dropout: dropout inside residual blocks
        num_classes: >0 adds class conditioning (labels from the dataset)
    """

    def __init__(self, in_channels: int = 3, dims: int = 2, base_channels: int = 64, channel_mult: tuple = (1, 2, 2),
                 num_res_blocks: int = 2, attention_levels: tuple = (2,), dropout: float = 0.1, num_classes: int = 0):
        super().__init__()
        C, self.dims = conv(dims), dims
        emb = base_channels * 4
        self.levels = len(channel_mult)
        self.time = nn.Sequential(SinusoidalEmbedding(base_channels), nn.Linear(base_channels, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.label = nn.Embedding(num_classes, emb) if num_classes else None
        self.inp = C(in_channels, base_channels, 3, padding=1)
        self.down, chans, ch = nn.ModuleList(), [base_channels], base_channels
        for i, m in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                blk = nn.ModuleList([ResBlock(dims, ch, base_channels * m, emb, dropout)])
                ch = base_channels * m
                if i in attention_levels and dims == 2:
                    blk.append(SelfAttention(ch))
                self.down.append(blk)
                chans.append(ch)
            if i < len(channel_mult) - 1:
                self.down.append(nn.ModuleList([C(ch, ch, 3, stride=self._stride(), padding=1)]))
                chans.append(ch)
        self.mid = nn.ModuleList([ResBlock(dims, ch, ch, emb, dropout)] + ([SelfAttention(ch)] if dims == 2 else []) +
                                 [ResBlock(dims, ch, ch, emb, dropout)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(channel_mult))):
            for j in range(num_res_blocks + 1):
                blk = nn.ModuleList([ResBlock(dims, ch + chans.pop(), base_channels * m, emb, dropout)])
                ch = base_channels * m
                if i in attention_levels and dims == 2:
                    blk.append(SelfAttention(ch))
                if i > 0 and j == num_res_blocks:
                    blk.append(_Upsample(dims, ch))
                self.up.append(blk)
        self.out = nn.Sequential(norm(ch), nn.SiLU(), C(ch, in_channels, 3, padding=1))

    def _stride(self):
        return (1, 2, 2) if self.dims == 3 else 2       # video: downsample space, keep time

    def forward(self, x, t, y=None):
        sf = 3 if self.dims == 3 else 2
        x, sizes = _pad_to(x, 2 ** (self.levels - 1), sf)
        e = self.time(t)
        if self.label is not None and y is not None:
            e = e + self.label(y)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            for layer in blk:
                h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
        for blk in self.up:
            h = torch.cat([h, hs.pop()], 1)
            for layer in blk:
                h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
        return _crop(self.out(h), sizes, sf)


class _Upsample(nn.Module):
    def __init__(self, dims, ch):
        super().__init__()
        self.dims = dims
        self.c = conv(dims)(ch, ch, 3, padding=1)

    def forward(self, x):
        sf = (1, 2, 2) if self.dims == 3 else 2
        return self.c(F.interpolate(x, scale_factor=sf, mode="nearest"))


class MLPDenoiser(nn.Module):
    """Time-conditioned residual MLP denoiser for vector data (e.g. a VAE's latent vectors)."""

    def __init__(self, dim: int, hidden: int = 512, depth: int = 4, num_classes: int = 0):
        super().__init__()
        self.time = nn.Sequential(SinusoidalEmbedding(128), nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.label = nn.Embedding(num_classes, hidden) if num_classes else None
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x, t, y=None):
        e = self.time(t)
        if self.label is not None and y is not None:
            e = e + self.label(y)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h + e)
        return self.out(h)


# ============================================================================= conv encoder / decoder (VAE, GAN)
def _levels(shape, max_levels=5, min_size=4):
    """How many 2x downsamples fit the spatial size (time axis of video excluded)."""
    spatial = shape[2:] if len(shape) == 4 else shape[1:]
    n = 0
    s = min(spatial)
    while n < max_levels and s // 2 >= min_size:
        s //= 2
        n += 1
    return max(n, 1)


class ConvEncoder(nn.Module):
    """Strided conv stack -> flat vector of size `out_features` (VAE encoder: 2 x latent; GAN discriminator: 1)."""

    def __init__(self, data_shape: tuple, out_features: int, base_channels: int = 32, max_channels: int = 256,
                 spectral_norm: bool = False, use_norm: bool = True):
        super().__init__()
        dims = len(data_shape) - 1
        C, n = conv(dims), _levels(data_shape)
        sn = nn.utils.spectral_norm if spectral_norm else (lambda m: m)
        stride = (1, 2, 2) if dims == 3 else 2
        layers, ch = [], data_shape[0]
        for i in range(n):
            out = min(base_channels * 2 ** i, max_channels)
            layers += [sn(C(ch, out, 4, stride=stride, padding=1))]
            if use_norm and i > 0:
                layers += [norm(out)]
            layers += [nn.LeakyReLU(0.2)]
            ch = out
        self.body = nn.Sequential(*layers)
        self.multiple = 2 ** n
        self.data_shape = tuple(data_shape)
        with torch.no_grad():
            flat = self.body(torch.zeros(1, *self._padded_shape())).numel()
        self.head = sn(nn.Linear(flat, out_features))

    def _padded_shape(self):
        c, *s = self.data_shape
        if len(s) == 3:
            return (c, s[0], *[x + (-x) % self.multiple for x in s[1:]])
        return (c, *[x + (-x) % self.multiple for x in s])

    def forward(self, x):
        x, _ = _pad_to(x, self.multiple, 3 if x.dim() == 5 else 2)
        return self.head(self.body(x).flatten(1))


class ConvDecoder(nn.Module):
    """Vector -> data (VAE decoder / GAN generator): nearest-neighbour upsampling + 3x3 convs (no transposed-conv
    checkerboard artifacts), residual refinement at each scale. Output in [-1, 1] (tanh)."""

    def __init__(self, data_shape: tuple, in_features: int, base_channels: int = 32, max_channels: int = 256,
                 out_activation: str = "tanh"):
        super().__init__()
        dims = len(data_shape) - 1
        n = _levels(data_shape)
        self.data_shape, self.dims = tuple(data_shape), dims
        c, *s = data_shape
        if dims == 3:
            self.start = (s[0], *[math.ceil(x / 2 ** n) for x in s[1:]])
        else:
            self.start = tuple(math.ceil(x / 2 ** n) for x in s)
        self.ch0 = min(base_channels * 2 ** (n - 1), max_channels)
        self.fc = nn.Linear(in_features, self.ch0 * math.prod(self.start))
        C = conv(dims)
        self.ups = nn.ModuleList()
        ch = self.ch0
        for i in reversed(range(n)):
            out = min(base_channels * 2 ** max(i - 1, 0), max_channels)
            self.ups.append(nn.ModuleList([C(ch, out, 3, padding=1), ResBlock(dims, out, out, 0)]))
            ch = out
        self.out = nn.Sequential(norm(ch), nn.SiLU(), C(ch, c, 3, padding=1))
        self.act = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "none": lambda v: v}[out_activation]

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], self.ch0, *self.start)
        sf = (1, 2, 2) if self.dims == 3 else 2
        for c, res in self.ups:
            h = res(c(F.interpolate(h, scale_factor=sf, mode="nearest")))
        return self.act(_crop(self.out(h), self.data_shape[1:], 2))


# ============================================================================= GPT (text)
class GPT(nn.Module):
    """Small decoder-only transformer language model (nanoGPT-style).

    Args:
        vocab_size: tokens in the vocabulary (injected from the tokenizer)
        block_size: context length in tokens
        n_layer: transformer blocks
        n_head: attention heads
        n_embd: embedding width
        dropout: dropout probability
    """

    def __init__(self, vocab_size: int = 256, block_size: int = 256, n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(n_embd, n_head, 4 * n_embd, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        L = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))[None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=idx.device), 1)
        return self.head(self.ln(self.blocks(h, mask=mask, is_causal=True)))
