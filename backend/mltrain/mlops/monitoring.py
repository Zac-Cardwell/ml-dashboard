"""Window-level monitoring: drift, data quality, prediction distribution, live performance, operations.

Industry-standard tests:
  * PSI (Population Stability Index) on reference quantile bins — < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant
  * Kolmogorov-Smirnov statistic + normalized Wasserstein distance for numeric features
  * Jensen-Shannon distance for categorical features
  * performance on the subset of predictions whose ground-truth labels have arrived
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

EPS = 1e-4

DEFAULT_CONFIG = {
    "window_minutes": 60,          # each snapshot looks at the last N minutes of traffic
    "interval_minutes": 5,         # how often a snapshot is computed
    "min_samples": 30,             # below this a window is 'insufficient data'
    "psi_warn": 0.1, "psi_alert": 0.25,
    "perf_warn_drop": 0.03, "perf_alert_drop": 0.08,   # absolute drop of the headline metric vs. baseline
    "perf_min_labeled": 100,       # labels needed before performance can raise alerts
    "performance_metric": "auto",  # auto = AUC for binary classification, accuracy otherwise
    "error_rate_alert": 0.05,
    "latency_p95_warn_ms": 250,
    "quality_warn": 0.05,          # extra missing / out-of-range / unseen-category rate vs. training
    "retention_days": 30,
}


def psi(ref: list[float], cur: list[float]) -> float:
    r = np.clip(np.asarray(ref, float), EPS, None)
    c = np.clip(np.asarray(cur, float), EPS, None)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def js_distance(ref: list[float], cur: list[float]) -> float:
    r = np.clip(np.asarray(ref, float), EPS, None)
    c = np.clip(np.asarray(cur, float), EPS, None)
    r, c = r / r.sum(), c / c.sum()
    m = (r + c) / 2
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return math.sqrt(max(0.0, (kl(r, m) + kl(c, m)) / 2))


def edges_of(prof: dict) -> list[float]:
    """A profile's bin edges. Profiles saved before the rounding fix stored a top edge equal to the column max
    (np.nextafter(max) rounded back down), so values at the max landed one bin higher than in the reference histogram
    — huge false PSI on binary / integer columns. Nudging that edge just above the max restores the reference binning."""
    edges = list(prof.get("edges") or [])
    mx = prof.get("max")
    if edges and mx is not None and edges[-1] <= mx:
        edges[-1] = float(mx) + 1e-6
    return edges


def numeric_hist(values: np.ndarray, edges: list[float]) -> list[float]:
    k = len(edges) + 1
    if not len(values):
        return [0.0] * k
    c = np.bincount(np.digitize(values, edges), minlength=k)[:k]
    return (c / c.sum()).round(6).tolist()


def _status(v: float | None, warn: float, alert: float) -> str:
    if v is None:
        return "unknown"
    return "alert" if v >= alert else "warn" if v >= warn else "ok"


def feature_drift(name: str, prof: dict, values: list, cfg: dict) -> dict:
    if prof["type"] == "numeric":
        arr = np.asarray([v if isinstance(v, (int, float)) and v is not None else np.nan for v in values], float)
        missing = float(np.isnan(arr).mean()) if len(arr) else 0.0
        a = arr[~np.isnan(arr)]
        out = {"type": "numeric", "missing_rate": missing, "n": int(len(a))}
        if not len(a) or not prof.get("edges") and "mean" not in prof:
            return out | {"psi": None, "status": "unknown"}
        cur = numeric_hist(a, edges_of(prof))
        out["cur_hist"] = cur
        out["psi"] = psi(prof["ref_hist"], cur)
        out["mean"] = float(a.mean())
        out["mean_shift_sd"] = float((a.mean() - prof["mean"]) / (prof["std"] + 1e-9))
        out["out_of_range_rate"] = float(((a < prof["min"]) | (a > prof["max"])).mean())
        if prof.get("sample") and len(a) >= 5:
            from scipy.stats import ks_2samp, wasserstein_distance
            ks = ks_2samp(prof["sample"], a)
            out["ks"], out["ks_p"] = float(ks.statistic), float(ks.pvalue)
            out["wasserstein_sd"] = float(wasserstein_distance(prof["sample"], a) / (prof["std"] + 1e-9))
    else:
        vals = ["__missing__" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values]
        n = max(len(vals), 1)
        missing = vals.count("__missing__") / n
        present = [v for v in vals if v != "__missing__"]
        counts = Counter(present)
        tot = max(len(present), 1)
        cats = prof["categories"]
        cur = [counts[c] / tot for c in cats] + [sum(v for k, v in counts.items() if k not in cats) / tot]
        out = {"type": "categorical", "missing_rate": missing, "n": len(present), "cur_hist": [round(x, 6) for x in cur],
               "psi": psi(prof["ref_hist"], cur), "js": js_distance(prof["ref_hist"], cur),
               "unseen_rate": cur[-1] if prof["ref_hist"][-1] < EPS else max(0.0, cur[-1] - prof["ref_hist"][-1]),
               "unseen_examples": [k for k, _ in counts.most_common() if k not in cats][:5]}
    out["missing_delta"] = out["missing_rate"] - prof.get("missing", 0.0)
    # sampling noise: with no real drift, PSI·n_eff ≈ χ²(bins-1). Only call a shift drift when PSI exceeds both the
    # business threshold and the 99th percentile of that noise — otherwise small windows (or small test splits) look drifted.
    k = len(out.get("cur_hist") or []) or 2
    n_ref = prof.get("n") or 1e9
    n_eff = 1 / (1 / max(out["n"], 1) + 1 / n_ref)
    from scipy.stats import chi2
    out["psi_noise"] = float(chi2.ppf(0.99, max(k - 1, 1)) / n_eff)
    out["status"] = _status(out.get("psi"), max(cfg["psi_warn"], out["psi_noise"]), max(cfg["psi_alert"], out["psi_noise"]))
    return out


def performance(rows: list[dict], task: str, classes: list[str] | None) -> dict:
    lab = [r for r in rows if r.get("label") is not None and not r.get("error")]
    out: dict = {"n_labeled": len(lab)}
    if not lab:
        return out
    if task == "rl":
        rets = [float(r["label"]) for r in lab]
        out["metrics"] = {"episode_return": float(np.mean(rets))}
        return out
    y = [r["label"] for r in lab]
    p = [r["prediction"] for r in lab]
    if classes:   # labels may arrive as class indices or names
        y = [classes[int(v)] if isinstance(v, (int, float)) and not isinstance(v, bool) and str(v) not in classes
             and 0 <= int(v) < len(classes) else v for v in y]
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        ys, ps = [str(v) for v in y], [str(v) for v in p]
        kw = dict(average="macro", zero_division=0)
        out["metrics"] = {"acc": accuracy_score(ys, ps), "f1": f1_score(ys, ps, **kw),
                          "precision": precision_score(ys, ps, **kw), "recall": recall_score(ys, ps, **kw)}
        if classes and len(classes) == 2 and all(r.get("score") is not None for r in lab) and len(set(ys)) == 2:
            out["metrics"]["auc"] = float(roc_auc_score([int(v == classes[1]) for v in ys], [r["score"] for r in lab]))
            # calibration: predicted P(positive) vs. observed rate, in 10 bins
            sp = np.asarray([r["score"] for r in lab])
            yt = np.asarray([int(v == classes[1]) for v in ys])
            bins = np.clip((sp * 10).astype(int), 0, 9)
            out["calibration"] = [{"bin": b / 10 + 0.05, "pred": float(sp[bins == b].mean()), "obs": float(yt[bins == b].mean()),
                                   "n": int((bins == b).sum())} for b in range(10) if (bins == b).any()]
        out["se"] = _bootstrap_se(ys, ps, [r["score"] for r in lab] if "auc" in out["metrics"] else None,
                                  classes[1] if classes and len(classes) == 2 else None)
        names = classes or sorted(set(ys) | set(ps))
        idx = {n: i for i, n in enumerate(names)}
        cm = np.zeros((len(names), len(names)), int)
        for a, b in zip(ys, ps):
            if a in idx and b in idx:
                cm[idx[a], idx[b]] += 1
        out["confusion"] = {"labels": names, "matrix": cm.tolist()}
        out["label_dist"] = {n: ys.count(n) / len(ys) for n in names}
    elif task == "regression":
        yv, pv = np.asarray(y, float), np.asarray(p, float)
        mse = float(((yv - pv) ** 2).mean())
        out["metrics"] = {"rmse": math.sqrt(mse), "mae": float(np.abs(yv - pv).mean()),
                          "r2": float(1 - mse / (yv.var() + 1e-12))}
    else:  # seq2seq: exact match + chrF on labeled samples
        import sacrebleu
        out["metrics"] = {"bleu": sacrebleu.corpus_bleu([str(x) for x in p], [[str(x) for x in y]]).score,
                          "chrf": sacrebleu.corpus_chrf([str(x) for x in p], [[str(x) for x in y]]).score}
    return out


def _bootstrap_se(ys, ps, scores, pos, n_boot: int = 100) -> dict:
    """Bootstrap standard errors of accuracy (and AUC) for this window — used to ignore noise-level drops."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    ys, ps = np.asarray(ys), np.asarray(ps)
    n = len(ys)
    accs, aucs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        accs.append((ys[i] == ps[i]).mean())
        if scores is not None:
            yb = (ys[i] == pos).astype(int)
            if 0 < yb.sum() < n:
                aucs.append(roc_auc_score(yb, np.asarray(scores)[i]))
    out = {"acc": float(np.std(accs))}
    if aucs:
        out["auc"] = float(np.std(aucs))
    return out


HEADLINE = {"classification": ("acc", "test/acc", True), "regression": ("rmse", "test/rmse", False),
            "seq2seq": ("bleu", "test/bleu", True), "rl": ("episode_return", "episode_return", True)}


def compute_window(rows: list[dict], profile: dict, cfg: dict, t0: float, t1: float) -> dict:
    """rows: prediction log rows (dicts with features, prediction, score, latency_ms, error, label)."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    n = len(rows)
    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    lat = np.asarray([r["latency_ms"] for r in ok if r.get("latency_ms") is not None], float)
    m: dict = {
        "n": n, "window": [t0, t1], "errors": len(errors), "error_rate": len(errors) / n if n else 0.0,
        # per-minute rate over the part of the window that actually has traffic history
        "throughput_per_min": n / max((t1 - max(t0, min((r["ts"] for r in rows), default=t0))) / 60, 1 / 60),
        "latency": {q: float(np.percentile(lat, p)) for q, p in (("p50", 50), ("p95", 95), ("p99", 99))} if len(lat) else {},
    }
    task = profile["task"]
    # ---- input drift + data quality
    feats = {}
    for name, fp in (profile.get("features") or {}).items():
        feats[name] = feature_drift(name, fp, [(r.get("features") or {}).get(name) for r in ok], cfg)
    m["features"] = feats
    psis = {k: v["psi"] for k, v in feats.items() if v.get("psi") is not None}
    drifted = sorted([k for k, v in feats.items() if v.get("status") == "alert"], key=lambda k: -psis[k])
    m["drift"] = {"max_psi": max(psis.values()) if psis else None, "mean_psi": float(np.mean(list(psis.values()))) if psis else None,
                  "n_drifted": len(drifted), "n_warn": sum(v.get("status") == "warn" for v in feats.values()),
                  "share_drifted": len(drifted) / max(len(psis), 1), "top": sorted(psis, key=lambda k: -psis[k])[:5]}
    # data-quality checks apply to raw inputs (tabular); image/text/RL features are derived statistics, covered by drift
    quality = {k: max(v.get("missing_delta", 0), v.get("out_of_range_rate", 0) - 0.02, v.get("unseen_rate", 0))
               for k, v in feats.items()} if profile.get("kind") == "tabular" else {}
    m["quality"] = {"worst": max(quality, key=quality.get) if quality else None, "worst_rate": max(quality.values()) if quality else 0.0,
                    "issues": {k: round(v, 4) for k, v in quality.items() if v >= cfg["quality_warn"]}}
    # ---- prediction drift
    pp = profile.get("prediction") or {}
    if pp.get("type") == "class":
        counts = Counter(str(r["prediction"]) for r in ok)
        cur = [counts[c] / max(len(ok), 1) for c in pp["classes"]]
        m["prediction"] = {"type": "class", "classes": pp["classes"], "cur_hist": cur, "ref_hist": pp["ref_hist"],
                           "psi": psi(pp["ref_hist"], cur) if ok else None}
        if pp.get("score") and pp["score"].get("edges"):
            # the logged score is P(positive) for binary classifiers, max-class confidence otherwise
            sv = np.asarray([r["score"] for r in ok if r.get("score") is not None], float)
            if len(sv):
                cur = numeric_hist(sv, edges_of(pp["score"]))
                m["score"] = {"cur_hist": cur, "ref_hist": pp["score"]["ref_hist"],
                              "edges": pp["score"]["edges"], "mean": float(sv.mean()), "ref_mean": pp["score"].get("mean"),
                              "psi": psi(pp["score"]["ref_hist"], cur)}
    elif pp.get("edges") is not None:
        if pp.get("key") == "output_length":
            sv = np.asarray([len(str(r["prediction"]).split()) for r in ok], float)
        else:
            sv = np.asarray([r["score"] for r in ok if r.get("score") is not None], float)
        if len(sv):
            cur = numeric_hist(sv, edges_of(pp))
            m["prediction"] = {"type": "numeric", "edges": pp["edges"], "cur_hist": cur, "ref_hist": pp["ref_hist"],
                               "psi": psi(pp["ref_hist"], cur), "mean": float(sv.mean()), "ref_mean": pp.get("mean")}
    # ---- live performance (vs. the baseline measured the same way on held-out training-time data)
    perf = performance(ok, task, profile.get("classes"))
    key, base_key, hib = HEADLINE.get(task, (None, None, True))
    live_base = profile.get("baseline_live") or {}
    want = cfg.get("performance_metric", "auto")
    if want == "auto":
        want = "auc" if "auc" in (perf.get("metrics") or {}) and "auc" in live_base else key
    if perf.get("metrics") and want in perf["metrics"]:
        key = want
        base = live_base.get(key, (profile.get("baseline") or {}).get(base_key) if key == HEADLINE.get(task, (None,))[0] else None)
        hib = key not in ("rmse", "mae")
        perf["headline"] = {"key": key, "value": perf["metrics"][key], "baseline": base, "se": perf.get("se", {}).get(key)}
        if base is not None and perf["n_labeled"] >= cfg["perf_min_labeled"]:
            drop = (base - perf["metrics"][key]) if hib else (perf["metrics"][key] - base) / (abs(base) + 1e-9)
            se = perf["headline"]["se"] or 0.0
            perf["headline"]["drop"] = drop
            perf["headline"]["significant"] = drop > 2 * se    # beyond sampling noise of this window
    m["performance"] = perf
    m["health"], m["reasons"] = health(m, cfg)
    return m


def health(m: dict, cfg: dict) -> tuple[str, list[dict]]:
    if m["n"] < cfg["min_samples"]:
        return "no_data", [{"severity": "info", "kind": "traffic", "key": "traffic", "message": f"Only {m['n']} predictions in the window"}]
    r: list[dict] = []
    hl = (m.get("performance") or {}).get("headline") or {}
    drop = hl.get("drop") if hl.get("significant") else None
    if drop is not None and drop >= cfg["perf_alert_drop"]:
        r.append({"severity": "critical", "kind": "performance", "key": "performance", "value": drop,
                  "message": f"Live {m['performance']['headline']['key']} {m['performance']['headline']['value']:.3f} is {drop:.3f} below the training baseline"})
    elif drop is not None and drop >= cfg["perf_warn_drop"]:
        r.append({"severity": "warning", "kind": "performance", "key": "performance", "value": drop,
                  "message": f"Live {m['performance']['headline']['key']} dropped {drop:.3f} vs. baseline"})
    if m["error_rate"] >= cfg["error_rate_alert"]:
        r.append({"severity": "critical", "kind": "errors", "key": "errors", "value": m["error_rate"],
                  "message": f"{m['error_rate']:.1%} of requests failed"})
    pred_psi = (m.get("prediction") or {}).get("psi")
    if pred_psi is not None and pred_psi >= cfg["psi_alert"]:
        r.append({"severity": "critical", "kind": "prediction_drift", "key": "prediction_drift", "value": pred_psi,
                  "message": f"Prediction distribution shifted (PSI {pred_psi:.2f})"})
    elif pred_psi is not None and pred_psi >= cfg["psi_warn"]:
        r.append({"severity": "warning", "kind": "prediction_drift", "key": "prediction_drift", "value": pred_psi,
                  "message": f"Prediction distribution moving (PSI {pred_psi:.2f})"})
    for f, v in m["features"].items():
        if v.get("status") == "alert":
            r.append({"severity": "warning", "kind": "feature_drift", "key": f"drift:{f}", "value": v["psi"],
                      "message": f"Feature '{f}' drifted (PSI {v['psi']:.2f})"})
    for f, rate in (m["quality"].get("issues") or {}).items():
        r.append({"severity": "warning", "kind": "data_quality", "key": f"quality:{f}", "value": rate,
                  "message": f"Data quality issue in '{f}': {rate:.1%} missing / out-of-range / unseen values"})
    p95 = (m.get("latency") or {}).get("p95")
    if p95 is not None and p95 >= cfg["latency_p95_warn_ms"]:
        r.append({"severity": "warning", "kind": "latency", "key": "latency", "value": p95, "message": f"p95 latency {p95:.0f} ms"})
    status = "critical" if any(x["severity"] == "critical" for x in r) else "warning" if r else "healthy"
    return status, r
