"""Interactive unsupervised exploration on a reproducible sample of a Studio dataset.

cluster   KMeans / Gaussian mixture / HDBSCAN / DBSCAN / agglomerative: quality metrics (silhouette, Davies-Bouldin,
          Calinski-Harabasz), an optional k sweep (elbow + silhouette), cluster profiles, agreement with the target
anomaly   Isolation Forest / LOF: score distribution and the most anomalous rows
project   PCA (explained variance + loadings) or t-SNE — t-SNE is for looking only: it has no transform for new rows,
          so it can't be a model feature (use PCA for feature reduction)

Results are exploratory; "apply" turns the chosen configuration into a recipe step (fit on train rows of the full table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine
from .eda import NUMERIC, _f
from .engine import quote_ident as q
from .steps import SPLIT

PLOT_POINTS = 3000


def _load(ds: dict, columns: list[str], sample_rows: int):
    con = engine.connect([ds])
    t = q(ds["slug"])
    types = {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM {t}").fetchall()}
    target = ds.get("target")
    num = columns or [c for c, ty in types.items() if NUMERIC.match(ty) and c not in (target, SPLIT)]
    bad = [c for c in num if c not in types or not NUMERIC.match(types[c])]
    if bad:
        raise ValueError(f"Not numeric (encode or drop them first): {bad}")
    if not num:
        raise ValueError("No numeric columns to work with — encode categoricals first (One-hot step)")
    extra = [target] if target and target in types else []
    df = con.execute(f"SELECT {', '.join(q(c) for c in num + extra)} FROM {t} USING SAMPLE reservoir({int(sample_rows)} ROWS) REPEATABLE (0)").fetchdf()
    n_total = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    con.close()
    X = df[num].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(med), 0, med), X)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd == 0, 1, sd)
    return df, num, Z, X, (df[target] if extra else None), n_total


def _project2d(Z, method="pca", seed=0):
    idx = np.random.default_rng(seed).permutation(len(Z))[:PLOT_POINTS]
    if method == "tsne" and len(Z) > 50:
        from sklearn.manifold import TSNE
        P = TSNE(2, init="pca", random_state=seed, perplexity=min(30, max(5, len(idx) // 50))).fit_transform(Z[idx])
    else:
        from sklearn.decomposition import PCA
        P = PCA(2, random_state=seed).fit(Z).transform(Z[idx]) if Z.shape[1] >= 2 else np.c_[Z[idx, 0], np.zeros(len(idx))]
    return idx, P


def _metrics(Z, labels, seed=0):
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    ok = labels >= 0
    k = len(set(labels[ok]))
    if k < 2 or ok.sum() < 10:
        return {"n_clusters": k}
    sub = np.random.default_rng(seed).permutation(np.flatnonzero(ok))[:10_000]
    return {"n_clusters": k, "silhouette": _f(silhouette_score(Z[sub], labels[sub])),
            "davies_bouldin": _f(davies_bouldin_score(Z[ok], labels[ok])), "calinski_harabasz": _f(calinski_harabasz_score(Z[ok], labels[ok])),
            "noise_share": _f(1 - ok.mean())}


def _fit_cluster(Z, algo, p, seed):
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        m = KMeans(int(p.get("k", 5)), n_init="auto", random_state=seed).fit(Z)
        return m.labels_, {"inertia": _f(m.inertia_)}
    if algo == "gmm":
        from sklearn.mixture import GaussianMixture
        m = GaussianMixture(int(p.get("k", 5)), random_state=seed).fit(Z)
        return m.predict(Z), {"bic": _f(m.bic(Z))}
    if algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        if len(Z) > 20_000:
            raise ValueError("agglomerative clustering is O(n²): use a sample of at most 20,000 rows")
        return AgglomerativeClustering(int(p.get("k", 5)), linkage=p.get("linkage", "ward")).fit_predict(Z), {}
    if algo == "hdbscan":
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=int(p.get("min_cluster_size", 25))).fit_predict(Z), {}
    if algo == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(p.get("eps", 0.5)), min_samples=int(p.get("min_samples", 5))).fit_predict(Z), {}
    raise ValueError(f"unknown algorithm {algo}")


def run(ds: dict, spec: dict) -> dict:
    task = spec.get("task", "cluster")
    seed = int(spec.get("seed", 0))
    sample_rows = int(spec.get("sample_rows") or 20_000)
    df, cols, Z, X, y, n_total = _load(ds, spec.get("columns") or [], sample_rows)
    out: dict = {"task": task, "columns": cols, "sample_rows": len(df), "total_rows": n_total}
    proj = spec.get("projection", "pca")
    if task == "cluster":
        algo = spec.get("algorithm", "kmeans")
        labels, extra = _fit_cluster(Z, algo, spec, seed)
        out["metrics"] = {**_metrics(Z, labels, seed), **extra}
        if spec.get("k_range") and algo in ("kmeans", "gmm"):
            lo, hi = spec["k_range"]
            sweep = []
            for k in range(max(2, int(lo)), min(int(hi), 20) + 1):
                lab, ex = _fit_cluster(Z, algo, {**spec, "k": k}, seed)
                sweep.append({"k": k, **_metrics(Z, lab, seed), **ex})
            out["sweep"] = sweep
        # profiles: mean of each feature per cluster (raw units and z-scores)
        prof = []
        for c in sorted(set(labels)):
            m = labels == c
            z = Z[m].mean(0)
            top = np.argsort(-np.abs(z))[:5]
            prof.append({"cluster": int(c), "size": int(m.sum()), "share": float(m.mean()),
                         "means": {col: _f(v) for col, v in zip(cols, X[m].mean(0))},
                         "distinguishing": [{"column": cols[i], "z": _f(z[i])} for i in top]})
        out["profiles"] = prof
        if y is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            yy = y.astype(str).to_numpy()
            out["metrics"]["ari_vs_target"] = _f(adjusted_rand_score(yy, labels))
            out["metrics"]["nmi_vs_target"] = _f(normalized_mutual_info_score(yy, labels))
            ct = pd.crosstab(labels, yy)
            out["crosstab"] = {"clusters": [int(i) for i in ct.index], "classes": [str(c) for c in ct.columns], "counts": ct.values.tolist()}
        idx, P = _project2d(Z, proj, seed)
        out["points"] = {"x": P[:, 0].round(4).tolist(), "y": P[:, 1].round(4).tolist(), "label": labels[idx].astype(int).tolist()}
    elif task == "anomaly":
        algo = spec.get("algorithm", "isolation_forest")
        cont = float(spec.get("contamination", 0.01))
        if algo == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            m = IsolationForest(contamination=cont, random_state=seed, n_jobs=-1).fit(Z)
            score = -m.score_samples(Z)
            flag = m.predict(Z) == -1
        else:
            from sklearn.neighbors import LocalOutlierFactor
            m = LocalOutlierFactor(contamination=cont)
            flag = m.fit_predict(Z) == -1
            score = -m.negative_outlier_factor_
        hist, edges = np.histogram(score, 40)
        out["score_hist"] = {"edges": edges.round(5).tolist(), "counts": hist.tolist(), "threshold": _f(score[flag].min()) if flag.any() else None}
        out["metrics"] = {"flagged": int(flag.sum()), "flagged_share": float(flag.mean())}
        if y is not None and y.nunique() == 2:
            from sklearn.metrics import average_precision_score, roc_auc_score
            minority = y.value_counts().index[-1]
            yt = (y == minority).to_numpy()
            out["metrics"].update(auc_vs_minority_class=_f(roc_auc_score(yt, score)), ap_vs_minority_class=_f(average_precision_score(yt, score)),
                                  minority_class=str(minority))
        top = np.argsort(-score)[:25]
        rows = df.iloc[top].copy()
        rows.insert(0, "anomaly_score", score[top].round(4))
        out["top_rows"] = {"columns": list(rows.columns), "rows": [[_cell(v) for v in r] for r in rows.itertuples(index=False)]}
        idx, P = _project2d(Z, proj, seed)
        out["points"] = {"x": P[:, 0].round(4).tolist(), "y": P[:, 1].round(4).tolist(), "label": flag[idx].astype(int).tolist(),
                         "score": score[idx].round(4).tolist()}
    elif task == "project":
        from sklearn.decomposition import PCA
        n = min(len(cols), 20)
        pca = PCA(n, random_state=seed).fit(Z)
        out["explained_variance_ratio"] = pca.explained_variance_ratio_.round(5).tolist()
        out["loadings"] = [{"component": f"pc{i + 1}", "top": [{"column": cols[j], "weight": _f(pca.components_[i, j])}
                                                                for j in np.argsort(-np.abs(pca.components_[i]))[:6]]} for i in range(min(n, 6))]
        idx, P = _project2d(Z, proj, seed)
        color = spec.get("color_by")
        lab = None
        if color and y is not None and color == ds.get("target"):
            lab = y.astype(str).to_numpy()[idx].tolist()
        out["points"] = {"x": P[:, 0].round(4).tolist(), "y": P[:, 1].round(4).tolist(), "label": lab}
        if proj == "tsne":
            out["note"] = "t-SNE is for visual exploration only (no transform for new rows); use a PCA step for feature reduction."
    else:
        raise ValueError(f"unknown task {task}")
    out["projection"] = proj
    return out


def _cell(v):
    if isinstance(v, (float, np.floating)):
        return _f(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return None if v is None else str(v)


def recipe_step(spec: dict) -> dict:
    """The recipe step that reproduces an exploration result on the full table (fit on train rows)."""
    task, algo = spec.get("task"), spec.get("algorithm")
    cols = spec.get("columns") or []
    if task == "cluster":
        if algo == "agglomerative":
            raise ValueError("agglomerative clustering can't label new rows; use KMeans, GMM or HDBSCAN for a saved column")
        return {"type": "cluster", "params": {"columns": cols, "algorithm": algo, "k": int(spec.get("k", 5)), "eps": float(spec.get("eps", 0.5)),
                                              "min_cluster_size": int(spec.get("min_cluster_size", 25)), "name": spec.get("name") or "cluster",
                                              "seed": int(spec.get("seed", 0))}}
    if task == "anomaly":
        return {"type": "anomaly", "params": {"columns": cols, "algorithm": algo, "contamination": float(spec.get("contamination", 0.01)),
                                              "name": spec.get("name") or "anomaly", "seed": int(spec.get("seed", 0))}}
    if task == "project":
        return {"type": "pca", "params": {"columns": cols, "n_components": int(spec.get("n_components") or 0), "variance": float(spec.get("variance", 0.95)),
                                          "keep_original": bool(spec.get("keep_original", False)), "prefix": "pc"}}
    raise ValueError("nothing to apply")
