"""Monitoring features extracted from model inputs, per modality.

Tabular inputs are monitored column by column. For images, text and RL observations we monitor
summary statistics that are cheap to compute and drift when the input distribution changes.
"""
from __future__ import annotations

import base64
import io
import re
import string

import numpy as np

_WORD = re.compile(r"[A-Za-z']+")


def decode_image(x) -> np.ndarray:
    """base64 (optionally a data: URL) or nested list -> HxW or HxWxC uint8/float array."""
    if isinstance(x, dict):
        x = x.get("image") or x.get("b64") or x.get("data")
    if isinstance(x, str):
        from PIL import Image
        if x.startswith("data:"):
            x = x.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(x)))
        return np.asarray(img.convert("RGB") if img.mode not in ("L", "RGB") else img)
    return np.asarray(x)


def image_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    gray = a if a.ndim == 2 else a.mean(-1)
    out = {"brightness": float(gray.mean()), "contrast": float(gray.std()),
           "edge_density": float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()),
           "dark_fraction": float((gray < 0.1).mean()), "bright_fraction": float((gray > 0.9).mean())}
    if a.ndim == 3 and a.shape[-1] == 3:
        out.update(red=float(a[..., 0].mean()), green=float(a[..., 1].mean()), blue=float(a[..., 2].mean()))
    return out


def text_stats(text: str, vocab: set | None = None) -> dict:
    text = text or ""
    words = _WORD.findall(text.lower())
    n = max(len(text), 1)
    out = {"n_chars": float(len(text)), "n_words": float(len(words)),
           "frac_upper": sum(ch.isupper() for ch in text) / n, "frac_digit": sum(ch.isdigit() for ch in text) / n,
           "frac_punct": sum(ch in string.punctuation for ch in text) / n,
           "frac_non_ascii": sum(ord(ch) > 127 for ch in text) / n}
    if vocab is not None:
        out["oov_rate"] = sum(w not in vocab for w in words) / max(len(words), 1)
    return out


def text_of(x) -> str:
    return x.get("text") or x.get("source") or "" if isinstance(x, dict) else str(x)


def obs_features(obs, max_dims: int = 32) -> dict:
    a = np.asarray(obs, dtype=np.float32).ravel()
    if a.size > max_dims:   # image observations: summarize
        return {"obs_mean": float(a.mean()), "obs_std": float(a.std())}
    return {f"obs_{i}": float(v) for i, v in enumerate(a)}
