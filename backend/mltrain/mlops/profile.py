"""Reference profiles: what the model saw in training, captured once per registered version.

Drift is always measured against this profile, so it holds per-feature histograms (quantile bins for
numeric features, top categories for categorical ones), a reference sample for KS / Wasserstein tests,
the reference prediction distribution and the offline baseline metrics.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .features import image_stats, obs_features, text_stats

N_BINS = 10
MAX_CATS = 30
SAMPLE = 1000


def fractions(idx: np.ndarray, k: int) -> list[float]:
    c = np.bincount(np.asarray(idx, int), minlength=k)[:k].astype(float)
    return (c / max(c.sum(), 1)).round(6).tolist()


def numeric_profile(values, rng) -> dict:
    v = np.asarray(values, dtype=float)
    missing = float(np.isnan(v).mean()) if len(v) else 0.0
    v = v[~np.isnan(v)]
    if not len(v):
        return {"type": "numeric", "edges": [], "ref_hist": [1.0], "missing": missing}
    # quantile bins + an edge at the training min and just above the max, so values outside the training range get
    # their own (empty in training) bins — otherwise near-constant features (e.g. always 0) could never show drift
    # Edges are stored rounded to 6 decimals, so the reference histogram must be computed with those same rounded edges,
    # and the top edge must stay strictly above the max after rounding (np.nextafter(max) rounds back to max, which put
    # e.g. every 1 of a 0/1 column into a different bin at monitoring time than in the reference -> PSI ~9 with no drift).
    top = float(np.round(v.max(), 6)) + 1e-6
    edges = np.unique(np.round(np.concatenate([np.quantile(v, np.linspace(0, 1, N_BINS + 1)[1:-1]), [v.min(), top]]), 6))
    return {"type": "numeric", "n": int(len(v)), "edges": edges.tolist(), "ref_hist": fractions(np.digitize(v, edges), len(edges) + 1),
            "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "p01": float(np.quantile(v, 0.01)), "p99": float(np.quantile(v, 0.99)), "missing": missing,
            "sample": rng.choice(v, min(SAMPLE, len(v)), replace=False).round(6).tolist()}


def categorical_profile(values) -> dict:
    vals = ["__missing__" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values]
    counts = Counter(vals)
    missing = counts.pop("__missing__", 0) / max(len(vals), 1)
    cats = [c for c, _ in counts.most_common(MAX_CATS)]
    tot = max(sum(counts.values()), 1)
    other = 1 - sum(counts[c] for c in cats) / tot
    return {"type": "categorical", "n": int(sum(counts.values())), "categories": cats, "ref_hist": [round(counts[c] / tot, 6) for c in cats] + [round(other, 6)],
            "missing": missing}


def profile_records(rows: list[dict], rng) -> dict:
    """rows: list of feature dicts -> {feature: profile}."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            out[c] = numeric_profile(col.to_numpy(dtype=float), rng)
        else:
            out[c] = categorical_profile(col.tolist())
    return out


def prediction_profile(preds: list[dict], task: str, classes, rng) -> dict:
    if task == "classification":
        names = classes or sorted({str(p["prediction"]) for p in preds})
        counts = Counter(str(p["prediction"]) for p in preds)
        out = {"type": "class", "classes": names, "ref_hist": [round(counts[n] / max(len(preds), 1), 6) for n in names]}
        score_key = "positive_score" if preds and "positive_score" in preds[0] else "score"
        out["score"] = numeric_profile([p.get(score_key) for p in preds], rng) | {"key": score_key}
        return out
    if task == "rl":
        acts = [p["prediction"] for p in preds]
        if acts and isinstance(acts[0], int):
            names = [str(a) for a in sorted(set(acts))]
            counts = Counter(str(a) for a in acts)
            return {"type": "class", "classes": names, "ref_hist": [round(counts[n] / len(acts), 6) for n in names]}
        return {"type": "numeric", **numeric_profile([p["score"] for p in preds], rng)}
    if task == "seq2seq":
        return {"type": "numeric", "key": "output_length", **numeric_profile([p.get("output_length", 0) for p in preds], rng)}
    return {"type": "numeric", **numeric_profile([p["prediction"] for p in preds], rng)}


def build_profile(version_dir: Path, meta: dict, predictor) -> dict:
    """Load the version's training problem, profile its inputs and the model's predictions on held-out data."""
    from ..plugins.registry import get_registry
    rng = np.random.default_rng(0)
    reg = get_registry()
    plugin = reg.problem(meta["problem_kind"], meta["problem_id"])
    modality, task = meta["modality"], meta["task"]
    prof: dict = {"kind": "rl" if task == "rl" else modality, "task": task, "classes": (meta.get("info") or {}).get("class_names"),
                  "baseline": {k: v for k, v in (meta.get("summary") or {}).items() if "/" in k and isinstance(v, (int, float))}}

    if task == "rl":
        # many short-ish episodes so the reference isn't dominated by a few autocorrelated trajectories
        env = plugin.create(meta["problem_params"], meta["seed"])
        obs, _ = env.reset(seed=meta["seed"])
        feats, preds, rets, ret = [], [], [], 0.0
        for _ in range(10000):
            p = predictor.predict([obs.tolist()])[0]
            feats.append(obs_features(obs))
            preds.append(p)
            obs, r, term, trunc, _ = env.step(p["prediction"])
            ret += float(r)
            if term or trunc:
                rets.append(ret)
                ret = 0.0
                obs, _ = env.reset()
        env.close()
        prof["features"] = profile_records(feats, rng)
        prof["prediction"] = prediction_profile(preds, "rl", None, rng)
        prof["baseline"]["episode_return"] = float(np.mean(rets)) if rets else ret
        prof["n_reference"] = len(feats)
        return prof

    data = plugin.load(meta["problem_params"], meta["seed"])
    ref_labels = None
    if modality == "tabular":
        train_df = data.raw["train"][0]
        ref_df, ref_labels = data.raw["val"] if len(data.raw["val"][0]) else data.raw["test"]
        prof["features"] = profile_records(train_df.to_dict("records"), rng)
        recs = ref_df.head(3000).to_dict("records")
        ref_labels = ref_labels[:3000]
        prof["schema"] = {c: ("numeric" if prof["features"][c]["type"] == "numeric" else "categorical") for c in train_df.columns}
        prof["example"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in recs[0].items()} if recs else {}
    elif modality == "image":
        src = data.val if data.val is not None and len(data.val) else data.test
        idx = rng.choice(len(src), min(800, len(src)), replace=False)
        norm = meta["info"].get("image_norm") or {"mean": [0.5], "std": [0.5]}
        recs, ref_labels = [], []
        for i in idx:
            ref_labels.append(int(src[int(i)][1]))
            x = src[int(i)][0].numpy()
            x = x * np.asarray(norm["std"])[:, None, None] + np.asarray(norm["mean"])[:, None, None]
            recs.append(np.clip(x, 0, 1).transpose(1, 2, 0).squeeze())
        prof["features"] = profile_records([image_stats(r) for r in recs], rng)
    elif modality == "text":
        key = "text" if task == "classification" else "source"
        vocab = Counter(w for r in data.train[:20000] for w in r[key].lower().split())
        prof["vocab"] = [w.strip(".,!?;:\"'()") for w, _ in vocab.most_common(20000)]
        vs = set(prof["vocab"])
        rows = (data.val or data.test)[:400]
        recs = [r[key] for r in rows]
        ref_labels = [r.get("label", r.get("target")) for r in rows]
        prof["features"] = profile_records([text_stats(t, vs) for t in recs], rng)
        predictor.vocab = vs
    else:
        raise ValueError(f"Unsupported modality {modality}")

    preds = []
    for i in range(0, len(recs), 64):
        preds += predictor.predict(recs[i:i + 64])
    prof["prediction"] = prediction_profile(preds, task, prof["classes"], rng)
    prof["n_reference"] = len(recs)
    if ref_labels is not None:
        # baseline computed with exactly the code used for live performance, on held-out data
        from .monitoring import performance
        names = prof["classes"]
        rows = [{"prediction": p["prediction"], "score": p.get("positive_score", p.get("score")),
                 "label": names[int(y)] if task == "classification" and names else (y.item() if hasattr(y, "item") else y)}
                for p, y in zip(preds, ref_labels)]
        live = performance(rows, task, names)
        prof["baseline_live"] = live.get("metrics") or {}
    return prof
