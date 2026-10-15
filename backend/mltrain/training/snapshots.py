"""Builders for 'learning snapshots': small JSON payloads that show what a model does at a point in training.

The dashboard renders each kind with a step slider so you can watch the model learn.
"""
from __future__ import annotations

import base64
from typing import Callable

import numpy as np


def tabular_snapshot(task: str, predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray,
                     info: dict, max_points: int = 400) -> tuple[str, dict] | None:
    """Decision boundary for 2-feature classification, fitted curve for 1-feature regression; else None."""
    if X is None or len(X) == 0:
        return None
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(max_points, len(X)), replace=False)
    if task == "classification" and X.shape[1] == 2:
        return "decision_boundary", decision_boundary(predict, X, y, info.get("class_names"), idx)
    if task == "regression" and X.shape[1] == 1:
        lo, hi = float(X[:, 0].min()), float(X[:, 0].max())
        grid = np.linspace(lo, hi, 200, dtype=np.float32)[:, None]
        return "curve_fit", {"x": grid[:, 0].round(4).tolist(), "pred": np.asarray(predict(grid)).ravel().round(5).tolist(),
                             "points": {"x": X[idx, 0].round(4).tolist(), "y": np.asarray(y)[idx].round(5).tolist()}}
    return None


def decision_boundary(predict, X, y, class_names, idx, n: int = 70) -> dict:
    pad = 0.5
    x0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n, dtype=np.float32)
    x1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n, dtype=np.float32)
    gx, gy = np.meshgrid(x0, x1)
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    z = np.asarray(predict(grid)).reshape(n, n)
    return {"x": x0.round(4).tolist(), "y": x1.round(4).tolist(), "z": z.astype(int).tolist(),
            "class_names": [str(c) for c in (class_names or sorted(set(np.asarray(y).tolist())))],
            "points": {"x": X[idx, 0].round(4).tolist(), "y": X[idx, 1].round(4).tolist(), "label": np.asarray(y)[idx].astype(int).tolist()}}


def png_b64(img: np.ndarray) -> str:
    """HxW or HxWxC float/uint8 image -> base64 PNG (min-max scaled)."""
    import cv2
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255
    a = a.astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if max(a.shape[:2]) < 64:  # upscale tiny images so they're visible
        s = 64 // max(a.shape[:2]) + 1
        a = cv2.resize(a, (a.shape[1] * s, a.shape[0] * s), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", a)
    return base64.b64encode(buf.tobytes()).decode()


def image_predictions(images, labels, probs, class_names) -> dict:
    """images: N x C x H x W array; probs: N x K softmax."""
    probs = np.asarray(probs)
    preds = probs.argmax(1)
    names = [str(c) for c in class_names] if class_names else [str(i) for i in range(probs.shape[1])]
    return {"items": [{"img": png_b64(images[i]), "label": names[int(labels[i])], "pred": names[int(preds[i])],
                       "conf": round(float(probs[i, preds[i]]), 4), "correct": bool(preds[i] == labels[i])}
                      for i in range(len(preds))]}
