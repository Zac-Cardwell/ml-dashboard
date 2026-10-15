"""Model explanations.

Tabular — SHAP values (Shapley values estimated by permutation sampling, Štrumbelj & Kononenko 2014) computed on
the *raw* input features through the model's serving pipeline (preprocessing included). Works for any tabular
model, and attributions are in terms of the columns you know ("annual_income", "home_ownership"), not internal
one-hot / scaled columns. What is explained: P(positive class) for binary classifiers, P(predicted class) for
multiclass, the predicted value for regression.

Images — Grad-CAM (Selvaraju et al. 2017) on the last convolutional layer: which regions drove the predicted class.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .mlops.predictors import load_predictor

_lock = threading.Lock()


@lru_cache(maxsize=8)
def _predictor(path: str):
    return load_predictor(Path(path), "cpu")


@lru_cache(maxsize=4)
def _raw_data(path: str):
    """Raw train / test frames of the problem the model was trained on."""
    from .plugins.registry import get_registry
    meta = json.loads((Path(path) / "meta.json").read_text())
    plugin = get_registry().problem(meta["problem_kind"], meta["problem_id"])
    data = plugin.load(meta["problem_params"], meta["seed"])
    return meta, data


def _target_fn(pred, meta: dict):
    """f(records) -> 1-D output to explain, plus a label for it."""
    task = meta["task"]
    classes = (meta.get("info") or {}).get("class_names")
    if task == "classification":
        if classes and len(classes) == 2:
            def f(recs, target=None):
                return np.array([o["proba"][classes[1]] for o in pred.predict(recs)])
            return f, f"P({classes[1]})"

        def f(recs, target=None):
            outs = pred.predict(recs)
            return np.array([o["proba"][target] if target else o["score"] for o in outs])
        return f, "P(predicted class)"

    def f(recs, target=None):
        return np.array([float(o["prediction"]) for o in pred.predict(recs)])
    return f, "prediction"


def _clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def shapley_one(f, x: dict, background: list[dict], features: list[str], n_perm: int, rng, target=None) -> tuple[np.ndarray, float, float]:
    """Permutation-sampling Shapley values for one record. Returns (phi[features], base value, f(x))."""
    F = len(features)
    rows = []
    perms = []
    for _ in range(n_perm):
        perm = rng.permutation(F)
        b = background[rng.integers(len(background))]
        z = dict(b)
        rows.append(dict(z))
        for j in perm:
            z[features[j]] = x.get(features[j])
            rows.append(dict(z))
        perms.append(perm)
    out = f(rows, target)
    phi = np.zeros(F)
    for m, perm in enumerate(perms):
        seg = out[m * (F + 1):(m + 1) * (F + 1)]
        phi[perm] += np.diff(seg)
    phi /= n_perm
    fx = float(f([x], target)[0])
    return phi, fx - phi.sum(), fx


def tabular_global(path: str, n_explain: int = 150, n_background: int = 60, n_perm: int = 10) -> dict:
    cache = Path(path) / "explain_tabular.json"
    key = hashlib.md5(f"{n_explain}-{n_background}-{n_perm}".encode()).hexdigest()
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c
    with _lock:
        pred = _predictor(path)
        meta, data = _raw_data(path)
        if meta["modality"] != "tabular" or not data.raw:
            raise ValueError("SHAP explanations are available for tabular models")
        f, target_label = _target_fn(pred, meta)
        rng = np.random.default_rng(0)
        train_df = data.raw["train"][0]
        test_df, y_test = data.raw["test"] if len(data.raw["test"][0]) else data.raw["val"]
        features = [c for c in ((meta.get("info") or {}).get("raw_columns") or list(train_df.columns))]
        bg = train_df.sample(min(n_background, len(train_df)), random_state=0)[features].to_dict("records")
        idx = rng.choice(len(test_df), min(n_explain, len(test_df)), replace=False)
        sample = test_df.iloc[idx][features].reset_index(drop=True)
        recs = [{k: _clean(v) for k, v in r.items()} for r in sample.to_dict("records")]
        preds = pred.predict(recs)
        multiclass = meta["task"] == "classification" and len((meta.get("info") or {}).get("class_names") or []) > 2
        phis, bases, outs = [], [], []
        for x, p in zip(recs, preds):
            phi, base, fx = shapley_one(f, x, bg, features, n_perm, rng, p["prediction"] if multiclass else None)
            phis.append(phi); bases.append(base); outs.append(fx)
        phis = np.asarray(phis)
    order = np.argsort(-np.abs(phis).mean(0))
    classes = (meta.get("info") or {}).get("class_names")
    y_names = [classes[int(v)] if classes and meta["task"] == "classification" else _clean(v) for v in np.asarray(y_test)[idx]]

    def color_value(col: str) -> list:
        s = sample[col]
        if pd.api.types.is_numeric_dtype(s):
            return (s.rank(pct=True)).fillna(-1).round(3).tolist()   # percentile within the sample (for the color scale)
        return [None] * len(s)

    res = {
        "key": key, "method": "Shapley values (permutation sampling) on raw features", "explains": target_label,
        "base_value": float(np.mean(bases)), "features": [features[i] for i in order],
        "importance": [float(np.abs(phis[:, i]).mean()) for i in order],
        "direction": [float(np.corrcoef(phis[:, i], color_value(features[i]))[0, 1]) if
                      pd.api.types.is_numeric_dtype(sample[features[i]]) and np.std(phis[:, i]) > 0 else None for i in order],
        "beeswarm": {features[i]: {"shap": phis[:, i].round(5).tolist(), "color": color_value(features[i]),
                                   "value": [_clean(v) for v in sample[features[i]].tolist()]} for i in order[:12]},
        "examples": [{"record": r, "prediction": p["prediction"], "output": round(o, 5), "label": y}
                     for r, p, o, y in zip(recs, preds, outs, y_names)],
        "n": len(recs), "n_perm": n_perm,
    }
    cache.write_text(json.dumps(res, default=_clean))
    return res


def tabular_record(path: str, record: dict, n_background: int = 100, n_perm: int = 48) -> dict:
    """Waterfall explanation of a single raw record (more permutations => lower sampling noise)."""
    with _lock:
        pred = _predictor(path)
        meta, data = _raw_data(path)
        f, target_label = _target_fn(pred, meta)
        rng = np.random.default_rng(1)
        train_df = data.raw["train"][0]
        features = (meta.get("info") or {}).get("raw_columns") or list(train_df.columns)
        bg = train_df.sample(min(n_background, len(train_df)), random_state=1)[features].to_dict("records")
        x = {k: _clean(record.get(k)) for k in features}
        p = pred.predict([x])[0]
        multiclass = meta["task"] == "classification" and len((meta.get("info") or {}).get("class_names") or []) > 2
        phi, base, fx = shapley_one(f, x, bg, features, n_perm, rng, p["prediction"] if multiclass else None)
    order = np.argsort(-np.abs(phi))
    return {"explains": target_label if not multiclass else f"P({p['prediction']})", "base_value": base, "output": fx,
            "prediction": p["prediction"], "contributions": [{"feature": features[i], "value": x.get(features[i]), "shap": float(phi[i])} for i in order],
            "n_perm": n_perm}


# ---------------------------------------------------------------- Grad-CAM
def _png(a: np.ndarray) -> str:
    import cv2
    ok, buf = cv2.imencode(".png", a)
    return base64.b64encode(buf.tobytes()).decode()


def gradcam(path: str, images: list | None = None, n: int = 12) -> dict:
    import cv2
    import torch
    with _lock:
        pred = _predictor(path)
        meta = pred.meta
        if meta.get("framework") != "torch" or meta.get("modality") != "image":
            raise ValueError("Grad-CAM is available for PyTorch image models")
        model = pred.model
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        if not convs:
            raise ValueError("Model has no convolutional layer")
        # deepest conv layer that still has >= 7x7 spatial resolution (the very last one can be 3x3 on small images)
        sizes = {}
        hooks = [m.register_forward_hook(lambda mod, i, o: sizes.__setitem__(mod, o.shape[-2:])) for m in convs]
        with torch.no_grad():
            model(torch.zeros(1, *pred.info["input_shape"], device=pred.device))
        for hk in hooks:
            hk.remove()
        good = [m for m in convs if min(sizes.get(m, (0, 0))) >= 7]
        conv = good[-1] if good else convs[-1]
        # hook the conv *block* output (after its BatchNorm + activation), as in the Grad-CAM paper / ResNet layer4:
        # the raw conv output sits before BN, which can flip signs and distort the map
        acts_types = (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.LeakyReLU, torch.nn.Tanh, torch.nn.ELU)
        mods = list(model.modules())
        layer = conv
        for m in mods[mods.index(conv) + 1:]:
            if isinstance(m, torch.nn.Conv2d):
                break
            if isinstance(m, acts_types):
                layer = m
                break
        if layer is not conv:
            sizes[layer] = sizes[conv]
        acts, grads = {}, {}
        if isinstance(layer, torch.nn.ReLU) and layer.inplace:
            layer.inplace = False   # in-place activations break backward hooks
        h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        labels = None
        try:
            if images is None:
                _, data = _raw_data(path)
                src = data.test if data.test is not None and len(data.test) else data.val
                idx = np.random.default_rng(0).choice(len(src), min(n, len(src)), replace=False)
                norm = meta["info"].get("image_norm") or {"mean": [0.5], "std": [0.5]}
                images, labels = [], []
                for i in idx:
                    x, y = src[int(i)]
                    x = x.numpy() * np.asarray(norm["std"])[:, None, None] + np.asarray(norm["mean"])[:, None, None]
                    images.append(np.clip(x, 0, 1).transpose(1, 2, 0).squeeze())
                    labels.append(meta["info"]["class_names"][int(y)])
            c, h, w = pred.info["input_shape"]
            out = []
            for k, img in enumerate(images):
                x = pred._tensor([img])
                model.zero_grad()
                logits = model(x)
                probs = torch.softmax(logits, -1)[0]
                cls = int(probs.argmax())
                logits[0, cls].backward()
                weights = grads["g"].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * acts["a"]).sum(1))[0].detach().cpu().numpy()
                cam = cam / (cam.max() + 1e-8)
                base = pred._image(img, c, h, w).transpose(1, 2, 0)
                base = np.repeat(base, 3, -1) if base.shape[-1] == 1 else base
                size = max(160, h)
                base_u8 = cv2.resize((base * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
                heat = cv2.applyColorMap((cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8),
                                         cv2.COLORMAP_INFERNO)
                # blend in proportion to importance: unimportant regions keep the original image
                a = (cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)[..., None] * 0.75).astype(np.float32)
                overlay = (cv2.cvtColor(base_u8, cv2.COLOR_RGB2BGR) * (1 - a) + heat * a).astype(np.uint8)
                names = meta["info"]["class_names"]
                out.append({"image": _png(cv2.cvtColor(base_u8, cv2.COLOR_RGB2BGR)), "overlay": _png(overlay),
                            "prediction": names[cls], "confidence": float(probs[cls]),
                            "label": labels[k] if labels else None,
                            "top3": [{"class": names[int(i)], "p": float(probs[i])} for i in torch.argsort(probs, descending=True)[:3]]})
        finally:
            h1.remove()
            h2.remove()
    hw = tuple(sizes.get(layer, ()))
    return {"method": f"Grad-CAM on the last conv block {tuple(conv.weight.shape)} ({hw[0]}×{hw[1]} resolution)" if hw else "Grad-CAM", "items": out}


def model_path_for_run(run_id: str, variant: str = "best") -> Path:
    from . import storage
    p = storage.model_dir(run_id, variant)
    if not (p / "meta.json").exists():
        raise FileNotFoundError("This run has no saved model (saving was off or it was deleted)")
    return p
