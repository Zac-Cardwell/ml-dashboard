"""Unsupervised models as tracked training sessions (any tabular dataset; labels, if present, are only used to score).

Fitted on the train split, scored on val / test, multi-seed like every other session:
  clustering                KMeans, Gaussian mixture, HDBSCAN — silhouette, Davies-Bouldin, Calinski-Harabasz; ARI / NMI
                            against the dataset's labels when it has a classification target (external validation)
  anomaly_detection         Isolation Forest, LOF — flagged share; AUC / average precision vs the minority class if labeled
  dimensionality_reduction  PCA — explained variance, reconstruction RMSE
Use a sweep over n_clusters (New training → hyperparameter search) for an elbow / silhouette-vs-k comparison.
"""
from __future__ import annotations

import time

import numpy as np

from ..plugins.base import ModelPlugin, Param
from ..plugins.registry import register


class UnsupervisedModel(ModelPlugin):
    framework = "sklearn"
    tasks = ()
    modalities = ("tabular",)
    supports_eval = True
    session_task = "clustering"

    def compatible(self, task, modality):
        return modality == "tabular"

    def meta(self):
        return {**super().meta(), "any_task": True, "session_task": self.session_task}

    def build(self, p, seed):
        raise NotImplementedError

    # ------------------------------------------------------------------ shared
    @staticmethod
    def _labels(data):
        """Ground-truth labels for external validation (classification targets only)."""
        return data.task == "classification"

    def train(self, data, ctx):
        (Xtr, ytr), (Xva, yva), (Xte, yte) = data.train, data.val, data.test
        est = self.build(ctx.params, ctx.seed)
        ctx.set_progress(0.05, f"fitting {type(est).__name__} on {len(Xtr):,} rows")
        t0 = time.time()
        est.fit(Xtr)
        summary = {"fit_time_s": time.time() - t0, "n_train": len(Xtr)}
        summary.update(self.score(est, data, ctx, {"train": Xtr, "val": Xva, "test": Xte}, {"train": ytr, "val": yva, "test": yte}))
        if ctx.save_model:
            import joblib
            joblib.dump(est, ctx.model_path("model.joblib"))
        return summary

    def evaluate(self, data, ctx):
        import joblib
        est = joblib.load(ctx.init_path / "model.joblib")
        (Xtr, ytr), (Xva, yva), (Xte, yte) = data.train, data.val, data.test
        return self.score(est, data, ctx, {"val": Xva, "test": Xte}, {"val": yva, "test": yte})

    def score(self, est, data, ctx, Xs, ys) -> dict:
        raise NotImplementedError

    @staticmethod
    def _scatter(ctx, X, color, name="projection", title=""):
        from sklearn.decomposition import PCA
        idx = np.random.default_rng(0).permutation(len(X))[:3000]
        if X.shape[1] < 2 or len(idx) < 3:
            return
        P = PCA(2, random_state=0).fit_transform(X[idx])
        ctx.log_artifact(name, "scatter", {"x": P[:, 0].round(4).tolist(), "y": P[:, 1].round(4).tolist(),
                                           "label": [str(v) for v in np.asarray(color)[idx]], "title": title or "2-D PCA projection (test rows)"})


# ============================================================================= clustering
class _Clustering(UnsupervisedModel):
    session_task = "clustering"

    def predict(self, est, X, Xtr):
        return est.predict(X)

    def score(self, est, data, ctx, Xs, ys):
        from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                                     normalized_mutual_info_score, silhouette_score)
        out = {}
        Xtr = data.train[0]
        names = data.info.get("feature_names") or []
        for split, X in Xs.items():
            if not len(X):
                continue
            lab = self.predict(est, X, Xtr)
            ok = lab >= 0
            k = len(set(lab[ok]))
            out[f"{split}/n_clusters"] = k
            if k >= 2 and ok.sum() > k:
                sub = np.random.default_rng(ctx.seed).permutation(np.flatnonzero(ok))[:10_000]
                out[f"{split}/silhouette"] = float(silhouette_score(X[sub], lab[sub]))
                out[f"{split}/davies_bouldin"] = float(davies_bouldin_score(X[ok], lab[ok]))
                out[f"{split}/calinski_harabasz"] = float(calinski_harabasz_score(X[ok], lab[ok]))
            if (~ok).any():
                out[f"{split}/noise_share"] = float((~ok).mean())
            if self._labels(data) and len(ys[split]):
                out[f"{split}/ari"] = float(adjusted_rand_score(ys[split], lab))
                out[f"{split}/nmi"] = float(normalized_mutual_info_score(ys[split], lab))
            if split == "test":
                sizes = dict(zip(*np.unique(lab, return_counts=True)))
                ctx.log_artifact("cluster_sizes", "table", {"columns": ["cluster", "rows", "share"],
                                                            "rows": [[int(c), int(n), float(n / len(lab))] for c, n in sizes.items()]})
                prof_rows = []
                for c in sorted(sizes):
                    z = X[lab == c].mean(0)
                    top = np.argsort(-np.abs(z))[:4]
                    prof_rows.append([int(c), ", ".join(f"{names[i] if i < len(names) else i} {z[i]:+.2f}" for i in top)])
                ctx.log_artifact("cluster_profiles", "table", {"columns": ["cluster", "most distinguishing features (standardized mean)"],
                                                               "rows": prof_rows})
                self._scatter(ctx, X, lab, title="Test rows in 2-D (PCA), colored by cluster")
                if self._labels(data) and len(ys[split]):
                    from ..training.utils import confusion
                    classes = data.info.get("class_names") or sorted(set(ys[split].tolist()))
                    import pandas as pd
                    ct = pd.crosstab(lab, ys[split])
                    ctx.log_artifact("clusters_vs_labels", "table", {"columns": ["cluster"] + [str(classes[int(c)]) if int(c) < len(classes) else str(c) for c in ct.columns],
                                                                     "rows": [[int(i)] + [int(v) for v in r] for i, r in zip(ct.index, ct.values)]})
        if hasattr(est, "inertia_"):
            out["train/inertia"] = float(est.inertia_)
        if hasattr(est, "bic"):
            out["train/bic"] = float(est.bic(Xtr))
        return out


@register
class KMeansModel(_Clustering):
    id = "unsup.kmeans"
    name = "K-means"
    description = "Centroid clustering; fast on millions of rows (MiniBatch option). Sweep n_clusters for an elbow plot"

    def params(self):
        return [Param("n_clusters", "int", 5, min=2, max=200, group="Model"),
                Param("init", "choice", "k-means++", choices=["k-means++", "random"], group="Model"),
                Param("n_init", "int", 4, "Restarts (best inertia kept)", 1, 50, group="Model"),
                Param("minibatch", "bool", False, "MiniBatchKMeans (much faster on very large tables)", group="Model")]

    def build(self, p, seed):
        from sklearn.cluster import KMeans, MiniBatchKMeans
        cls = MiniBatchKMeans if p["minibatch"] else KMeans
        return cls(p["n_clusters"], init=p["init"], n_init=p["n_init"], random_state=seed)


@register
class GMMModel(_Clustering):
    id = "unsup.gmm"
    name = "Gaussian mixture"
    description = "Soft clustering with elliptical clusters; BIC for choosing the number of components"

    def params(self):
        return [Param("n_components", "int", 5, min=1, max=100, group="Model"),
                Param("covariance_type", "choice", "full", choices=["full", "tied", "diag", "spherical"], group="Model")]

    def build(self, p, seed):
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(p["n_components"], covariance_type=p["covariance_type"], random_state=seed)


class _KnnAssign:
    """Wraps a density clusterer (fit on a bounded sample) so new rows get the label of the nearest clustered
    training row — density methods have no predict() of their own."""

    def __init__(self, est, max_rows: int = 200_000, seed: int = 0):
        self.est, self.knn, self.max_rows, self.seed = est, None, max_rows, seed

    def fit(self, X):
        from sklearn.neighbors import KNeighborsClassifier
        X = X[np.random.default_rng(self.seed).permutation(len(X))[: self.max_rows]]
        lab = self.est.fit_predict(X)
        core = lab >= 0
        self.knn = KNeighborsClassifier(1).fit(X[core], lab[core]) if core.any() else None
        self.labels_ = lab
        return self

    def predict(self, X):
        return self.knn.predict(X) if self.knn is not None else np.full(len(X), -1)


@register
class HDBSCANModel(_Clustering):
    id = "unsup.hdbscan"
    name = "HDBSCAN"
    description = "Density clustering: finds the number of clusters itself and marks noise (-1). Fit on ≤ 200k train rows"

    def params(self):
        return [Param("min_cluster_size", "int", 25, min=2, group="Model"),
                Param("min_samples", "int", 0, "0 = same as min_cluster_size", 0, group="Model"),
                Param("max_fit_rows", "int", 200_000, "Fit on a random sample of at most this many train rows", 1000, group="Model")]

    def build(self, p, seed):
        from sklearn.cluster import HDBSCAN
        return _KnnAssign(HDBSCAN(min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"] or None),
                          p["max_fit_rows"], seed)


# ============================================================================= anomaly detection
class _Anomaly(UnsupervisedModel):
    session_task = "anomaly_detection"

    def score(self, est, data, ctx, Xs, ys):
        from sklearn.metrics import average_precision_score, roc_auc_score
        out = {}
        for split, X in Xs.items():
            if not len(X):
                continue
            s = -est.score_samples(X)
            flag = est.predict(X) == -1
            out[f"{split}/flagged_share"] = float(flag.mean())
            y = ys[split]
            if self._labels(data) and len(set(y.tolist())) == 2:
                vals, counts = np.unique(y, return_counts=True)
                minority = vals[np.argmin(counts)]
                yt = (y == minority).astype(int)
                out[f"{split}/auc"] = float(roc_auc_score(yt, s))
                out[f"{split}/average_precision"] = float(average_precision_score(yt, s))
            if split == "test":
                hist, edges = np.histogram(s, 40)
                ctx.log_artifact("anomaly_scores", "table", {"columns": ["score from", "score to", "rows"],
                                                             "rows": [[float(edges[i]), float(edges[i + 1]), int(hist[i])] for i in range(len(hist))]})
                self._scatter(ctx, X, np.where(flag, "anomaly", "normal"), title="Test rows in 2-D (PCA): flagged anomalies")
        return out


@register
class IsolationForestModel(_Anomaly):
    id = "unsup.isolation_forest"
    name = "Isolation Forest"
    description = "Tree-based anomaly detector; scales to large tables"

    def params(self):
        return [Param("n_estimators", "int", 200, min=10, group="Model"),
                Param("contamination", "float", 0.01, "Expected share of anomalies", 0.0001, 0.5, group="Model"),
                Param("max_samples", "int", 256, "Rows per tree", 16, group="Model")]

    def build(self, p, seed):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=p["n_estimators"], contamination=p["contamination"], max_samples=p["max_samples"],
                               random_state=seed, n_jobs=-1)


@register
class LOFModel(_Anomaly):
    id = "unsup.lof"
    name = "Local Outlier Factor"
    description = "Density-based anomaly detector (novelty mode: fit on train, score new rows)"

    def params(self):
        return [Param("n_neighbors", "int", 20, min=2, group="Model"),
                Param("contamination", "float", 0.01, min=0.0001, max=0.5, group="Model")]

    def build(self, p, seed):
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(n_neighbors=p["n_neighbors"], contamination=p["contamination"], novelty=True, n_jobs=-1)


# ============================================================================= dimensionality reduction
@register
class PCAModel(UnsupervisedModel):
    id = "unsup.pca"
    name = "PCA"
    description = "Principal components: how many dimensions the data really has (explained variance, reconstruction error)"
    session_task = "dimensionality_reduction"

    def params(self):
        return [Param("n_components", "int", 0, "0 = keep enough components for the variance target", 0, group="Model"),
                Param("variance", "float", 0.95, "Variance to explain when n_components = 0", 0.5, 0.9999, group="Model"),
                Param("whiten", "bool", False, group="Model")]

    def build(self, p, seed):
        from sklearn.decomposition import PCA
        return PCA(p["n_components"] or p["variance"], whiten=p["whiten"], random_state=seed)

    def score(self, est, data, ctx, Xs, ys):
        out = {"n_components": int(est.n_components_), "n_features": int(est.n_features_in_)}
        out["train/explained_variance"] = float(est.explained_variance_ratio_.sum())
        cum = np.cumsum(est.explained_variance_ratio_)
        for i, v in enumerate(cum):
            ctx.log(i + 1, {"explained_variance_cumulative": float(v), "explained_variance_component": float(est.explained_variance_ratio_[i])})
        for split, X in Xs.items():
            if not len(X) or split == "train":
                continue
            R = est.inverse_transform(est.transform(X))
            out[f"{split}/reconstruction_rmse"] = float(np.sqrt(((X - R) ** 2).mean()))
            tot = ((X - X.mean(0)) ** 2).sum()
            out[f"{split}/explained_variance"] = float(1 - ((X - R) ** 2).sum() / tot) if tot else None
        names = data.info.get("feature_names") or []
        rows = []
        for i in range(min(6, est.n_components_)):
            top = np.argsort(-np.abs(est.components_[i]))[:5]
            rows.append([f"pc{i + 1}", float(est.explained_variance_ratio_[i]),
                         ", ".join(f"{names[j] if j < len(names) else j} {est.components_[i, j]:+.2f}" for j in top)])
        ctx.log_artifact("components", "table", {"columns": ["component", "explained variance", "top loadings"], "rows": rows})
        X = Xs.get("test") if len(Xs.get("test", [])) else Xs.get("val")
        if X is not None and len(X):
            self._scatter(ctx, X, np.zeros(len(X)), title="Test rows on the first two components")
        return out
