"""Built-in datasets: sklearn toy sets, synthetic, CSV (uploaded or on disk), torchvision,
image folders, and Hugging Face text datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config
from ..plugins.base import DataBundle, DatasetPlugin, Param
from ..plugins.registry import register

# ---------------------------------------------------------------- helpers

SPLIT_PARAMS = [
    Param("val_fraction", "float", 0.15, "Fraction of data held out for validation", 0.0, 0.9, group="Data split"),
    Param("test_fraction", "float", 0.15, "Fraction of data held out for test", 0.0, 0.9, group="Data split"),
    Param("split_seed", "int", 0, "Seed for the train/val/test split. -1 = use each run's seed", -1, group="Data split"),
]
TABULAR_PARAMS = SPLIT_PARAMS + [
    Param("standardize", "bool", True, "Standardize features (fit on train)", group="Preprocessing"),
    Param("max_samples", "int", 0, "Subsample to at most N rows (0 = all)", 0, group="Preprocessing"),
]


def _split_seed(params: dict, seed: int) -> int:
    s = params.get("split_seed", 0)
    return seed if s is None or s < 0 else s


def split_indices(n: int, y, params: dict, seed: int, task: str):
    """Train/val/test index arrays (stratified for classification), after optional subsampling."""
    from sklearn.model_selection import train_test_split
    rs = _split_seed(params, seed)
    idx = np.arange(n)
    m = params.get("max_samples") or 0
    if m and m < n:
        idx = np.sort(np.random.default_rng(rs).choice(n, m, replace=False))
    vf, tf = params.get("val_fraction", 0.15), params.get("test_fraction", 0.15)
    strat = lambda ii: y[ii] if task == "classification" else None
    tr, te = idx, idx[:0]
    if tf > 0:
        tr, te = train_test_split(idx, test_size=tf, random_state=rs, stratify=strat(idx))
    va = tr[:0]
    if vf > 0:
        tr, va = train_test_split(tr, test_size=vf / (1 - tf), random_state=rs, stratify=strat(tr))
    return tr, va, te


def tabular_bundle(X, y, params, seed, task, feature_names=None, class_names=None, indices=None) -> DataBundle:
    """Split raw features, fit the preprocessor on train only, and transform every split.

    X may be a DataFrame with raw (possibly categorical / missing) columns or a numeric array.
    The fitted TabularPreprocessor travels with the model so serving can accept raw records.
    """
    import pandas as pd

    from ..preprocessing import TabularPreprocessor
    if isinstance(X, pd.DataFrame):
        df = X.reset_index(drop=True)
    else:
        X = np.asarray(X, dtype=np.float32)
        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=[str(c) for c in names])
    y = np.asarray(y)
    y = y.astype(np.int64) if task == "classification" else y.astype(np.float32)
    tr, va, te = indices if indices is not None else split_indices(len(df), y, params, seed, task)
    pre = TabularPreprocessor(standardize=params.get("standardize", True)).fit(df.iloc[tr])
    split = lambda ii: (pre.transform(df.iloc[ii]), y[ii])
    info = {"n_features": len(pre.feature_names), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "feature_names": pre.feature_names, "preprocessor": pre.to_dict(), "raw_columns": list(df.columns)}
    if task == "classification":
        info["n_classes"] = int(y.max()) + 1
        info["class_names"] = [str(c) for c in (class_names if class_names is not None else range(info["n_classes"]))]
    raw = {"train": (df.iloc[tr].reset_index(drop=True), y[tr]), "val": (df.iloc[va].reset_index(drop=True), y[va]),
           "test": (df.iloc[te].reset_index(drop=True), y[te])}
    return DataBundle(task, "tabular", split(tr), split(va), split(te), info, raw=raw)


# ---------------------------------------------------------------- sklearn toy datasets

def sk_dataset(did: str, title: str, loader: str, task: str, desc: str, builtin: bool = True):
    """A dataset plugin for an sklearn loader. builtin=False returns the class unregistered (for optional plugin files)."""
    class _D(DatasetPlugin):
        id = did
        name = title
        description = desc

        def params(self):
            return TABULAR_PARAMS

        def load(self, params, seed):
            import sklearn.datasets as skd
            fn = getattr(skd, loader)
            kw = {"data_home": str(config.CACHE_DIR / "sklearn")} if loader.startswith("fetch") else {}
            d = fn(**kw)
            return tabular_bundle(d.data, d.target, params, seed, task, d.get("feature_names"), d.get("target_names"))

        def data_paths(self):
            if not loader.startswith("fetch"):
                return []           # bundled with scikit-learn, nothing downloaded
            return [q for q in (config.CACHE_DIR / "sklearn").glob(f"{loader[len('fetch_'):].split('_')[0]}*")]

    _D.task = task
    _D.__name__ = f"SK_{did}"
    return register(_D) if builtin else _D


# One default dataset per problem type ships with the dashboard; more live in user_plugins/datasets/optional/
sk_dataset("sklearn.iris", "Iris", "load_iris", "classification", "150 flowers, 4 features, 3 classes")
sk_dataset("sklearn.diabetes", "Diabetes", "load_diabetes", "regression", "442 patients, 10 features, disease progression")


# ---------------------------------------------------------------- CSV

def read_table(path) -> "pd.DataFrame":
    """CSV or Parquet from a local path or a cloud URL (s3:// gs:// az://; see mltrain/cloud.py for credentials)."""
    import pandas as pd

    from .. import cloud
    p = str(path)
    opts = {"storage_options": cloud.storage_options(p)} if cloud.provider_of(p) else {}
    if cloud.provider_of(p):
        cloud.filesystem(p)          # clear message if the provider's library is missing
    return pd.read_parquet(p, **opts) if p.lower().endswith((".parquet", ".pq")) else pd.read_csv(p, **opts)


def load_csv(path: Path, target: str, task: str, params: dict, seed: int) -> DataBundle:
    import pandas as pd
    df = read_table(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    y_raw = df.pop(target)
    drop = [c for c in params.get("drop_columns", "").split(",") if c.strip()]
    df = df.drop(columns=[c.strip() for c in drop if c.strip() in df.columns])
    class_names = None
    if task == "classification":
        cats = pd.Categorical(y_raw)
        y, class_names = cats.codes, list(cats.categories)
    else:
        y = y_raw.astype(np.float32).values
    return tabular_bundle(df, y, params, seed, task, None, class_names)


CSV_EXTRA = [Param("drop_columns", "str", "", "Comma-separated columns to ignore (ids etc.)", group="Preprocessing")]


def uploaded_csv_datasets() -> list[DatasetPlugin]:
    """One dataset plugin per uploaded CSV (data/uploads/datasets/<slug>.csv + .json)."""
    out = []
    for meta_path in sorted((config.UPLOADS_DIR / "datasets").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        csv_path = meta_path.with_suffix(".csv")

        class _U(DatasetPlugin):
            id = f"upload.{meta_path.stem}"
            name = meta.get("name", meta_path.stem)
            description = meta.get("description") or f"Uploaded CSV, target '{meta['target']}'"
            task = meta["task"]
            _meta = meta
            _path = csv_path

            def params(self):
                return TABULAR_PARAMS + CSV_EXTRA

            def load(self, params, seed):
                return load_csv(self._path, self._meta["target"], self._meta["task"], params, seed)

        out.append(_U())
    return out


def _csv_path_dataset(task: str):
    class _C(DatasetPlugin):
        id = f"csv.path.{task}"
        name = f"CSV / Parquet file ({task})"
        description = "A CSV or Parquet file on this machine or in cloud storage (s3://, gs://, az://); choose the target column"

        def params(self):
            return [Param("path", "str", "", "Local path, or s3://bucket/key.csv · gs://bucket/key.parquet · az://container/key.csv",
                          group="Source"),
                    Param("target", "str", "", "Target column name", group="Source")] + TABULAR_PARAMS + CSV_EXTRA

        def load(self, params, seed):
            from .. import cloud
            p = params["path"] if cloud.provider_of(params["path"]) else Path(params["path"]).expanduser()
            return load_csv(p, params["target"], task, params, seed)

    _C.task = task
    return register(_C)


_csv_path_dataset("classification")
_csv_path_dataset("regression")


# ---------------------------------------------------------------- vision

VISION_PARAMS = [
    Param("val_fraction", "float", 0.1, "Fraction of the train split used for validation", 0.0, 0.5, group="Data split"),
    Param("split_seed", "int", 0, "-1 = use each run's seed", -1, group="Data split"),
    Param("max_train_samples", "int", 0, "Subsample the training set (0 = all)", 0, group="Data split"),
    Param("augment", "bool", False, "Random crop + horizontal flip on train", group="Preprocessing"),
]


def _split_torch(full_train, test, params, seed, info):
    import torch
    from torch.utils.data import Subset
    g = torch.Generator().manual_seed(_split_seed(params, seed))
    idx = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(len(idx) * params.get("val_fraction", 0.1))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    if params.get("max_train_samples"):
        tr_idx = tr_idx[: params["max_train_samples"]]
    train, val = Subset(full_train, tr_idx), Subset(full_train, val_idx)
    info.update(n_train=len(train), n_val=len(val), n_test=len(test) if test is not None else 0)
    return train, val


def torchvision_dataset(did: str, title: str, cls_name: str, shape, mean, std, desc: str, builtin: bool = True):
    class _V(DatasetPlugin):
        id = did
        name = title
        description = desc
        modality = "image"

        def params(self):
            return VISION_PARAMS

        def load(self, params, seed):
            import torchvision
            import torchvision.transforms as T
            from torch.utils.data import Subset
            cls = getattr(torchvision.datasets, cls_name)
            root = str(config.CACHE_DIR / "torchvision")
            base = [T.ToTensor(), T.Normalize(mean, std)]
            aug = [T.RandomCrop(shape[1], padding=4), T.RandomHorizontalFlip()] if params.get("augment") else []
            train_aug = cls(root, train=True, download=True, transform=T.Compose(aug + base))
            train_plain = cls(root, train=True, download=True, transform=T.Compose(base))
            test = cls(root, train=False, download=True, transform=T.Compose(base))
            info = {"input_shape": list(shape), "n_classes": len(train_plain.classes),
                    "class_names": [str(c) for c in train_plain.classes],
                    "image_norm": {"mean": list(mean), "std": list(std), "size": list(shape[1:]), "channels": shape[0]}}
            train, val = _split_torch(train_aug, test, params, seed, info)
            val = Subset(train_plain, val.indices)
            return DataBundle("classification", "image", train, val, test, info)

        def data_paths(self):
            root = config.CACHE_DIR / "torchvision"
            names = {"CIFAR10": ["cifar-10-batches-py", "cifar-10-python.tar.gz"],
                     "CIFAR100": ["cifar-100-python", "cifar-100-python.tar.gz"]}.get(cls_name, [cls_name])
            return [root / n for n in names]

    _V.__name__ = f"TV_{cls_name}"
    return register(_V) if builtin else _V


torchvision_dataset("vision.mnist", "MNIST", "MNIST", (1, 28, 28), (0.1307,), (0.3081,), "70k handwritten digits, 28x28 grayscale")


@register
class ImageFolderDataset(DatasetPlugin):
    id = "vision.image_folder"
    name = "Image folder on disk"
    description = "A directory with one sub-folder per class (torchvision ImageFolder layout)"
    modality = "image"

    def params(self):
        return [Param("path", "str", "", "Absolute path to the root folder", group="Source"),
                Param("image_size", "int", 64, "Images are resized to N x N", 8, 1024, group="Preprocessing"),
                Param("grayscale", "bool", False, group="Preprocessing"),
                Param("test_fraction", "float", 0.15, min=0.0, max=0.9, group="Data split")] + VISION_PARAMS

    def load(self, params, seed):
        import torch
        import torchvision.transforms as T
        from torch.utils.data import Subset
        from torchvision.datasets import ImageFolder
        s = params["image_size"]
        ch = 1 if params["grayscale"] else 3
        tf = [T.Resize((s, s))] + ([T.Grayscale()] if ch == 1 else []) + [T.ToTensor(), T.Normalize([0.5] * ch, [0.5] * ch)]
        aug = [T.RandomHorizontalFlip()] if params.get("augment") else []
        full = ImageFolder(params["path"], transform=T.Compose(aug + tf))
        plain = ImageFolder(params["path"], transform=T.Compose(tf))
        g = torch.Generator().manual_seed(_split_seed(params, seed))
        idx = torch.randperm(len(full), generator=g).tolist()
        n_test = int(len(idx) * params["test_fraction"])
        n_val = int(len(idx) * params["val_fraction"])
        test, val, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
        if params.get("max_train_samples"):
            tr = tr[: params["max_train_samples"]]
        info = {"input_shape": [ch, s, s], "n_classes": len(full.classes), "class_names": full.classes,
                "image_norm": {"mean": [0.5] * ch, "std": [0.5] * ch, "size": [s, s], "channels": ch},
                "n_train": len(tr), "n_val": len(val), "n_test": len(test)}
        return DataBundle("classification", "image", Subset(full, tr), Subset(plain, val), Subset(plain, test), info)


# ---------------------------------------------------------------- Hugging Face text

def _get_field(row: dict, path: str):
    for part in path.split("."):
        row = row[part]
    return row


def _hf_splits(params: dict, seed: int):
    """Load an HF dataset and return train/val/test HF Dataset objects."""
    from datasets import load_dataset
    ds = load_dataset(params["dataset"], params.get("config") or None, cache_dir=str(config.CACHE_DIR / "hf"))
    rs = _split_seed(params, seed)
    train = ds["train"].shuffle(seed=rs)
    val = ds["validation"] if "validation" in ds else None
    test = ds["test"] if "test" in ds else None
    label = params.get("label_field")
    if test is not None and label and label in test.column_names and min(test[label][:100]) < 0:
        test = None  # unlabeled test split (e.g. GLUE)
    if test is None:
        n = max(1, int(len(train) * params.get("test_fraction", 0.1)))
        test, train = train.select(range(n)), train.select(range(n, len(train)))
    if val is None:
        n = max(1, int(len(train) * params.get("val_fraction", 0.1)))
        val, train = train.select(range(n)), train.select(range(n, len(train)))

    def cap(d, k):
        n = params.get(k) or 0
        return d.shuffle(seed=rs).select(range(min(n, len(d)))) if n else d

    return cap(train, "max_train_samples"), cap(val, "max_eval_samples"), cap(test, "max_eval_samples")


HF_COMMON = [
    Param("max_train_samples", "int", 5000, "Cap training examples (0 = all)", 0, group="Data split"),
    Param("max_eval_samples", "int", 1000, "Cap val/test examples (0 = all)", 0, group="Data split"),
    Param("val_fraction", "float", 0.1, "Used only if the dataset has no validation split", 0.0, 0.5, group="Data split"),
    Param("test_fraction", "float", 0.1, "Used only if the dataset has no labeled test split", 0.0, 0.5, group="Data split"),
    Param("split_seed", "int", 0, "-1 = use each run's seed", -1, group="Data split"),
]


def hf_text_dataset(did, title, desc, dataset, cfg, text_field, label_field, builtin: bool = True):
    class _T(DatasetPlugin):
        id = did
        name = title
        description = desc
        modality = "text"

        def params(self):
            return [Param("dataset", "str", dataset, "Hugging Face dataset id", group="Source"),
                    Param("config", "str", cfg, "Dataset config/subset (optional)", group="Source"),
                    Param("text_field", "str", text_field, group="Source"),
                    Param("label_field", "str", label_field, group="Source")] + HF_COMMON

        def load(self, params, seed):
            tr, va, te = _hf_splits(params, seed)
            tf, lf = params["text_field"], params["label_field"]
            feat = tr.features.get(lf)
            names = list(feat.names) if hasattr(feat, "names") else sorted({int(x) for x in tr[lf]})
            conv = lambda d: [{"text": str(_get_field(r, tf)), "label": int(r[lf])} for r in d]
            info = {"n_classes": len(names), "class_names": [str(n) for n in names],
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
            return DataBundle("classification", "text", conv(tr), conv(va), conv(te), info)

        def data_paths(self):
            return hf_cache_paths(dataset)

    return register(_T) if builtin else _T


def hf_cache_paths(dataset: str) -> list:
    """Downloaded files of a Hugging Face dataset in the dashboard's cache."""
    if not dataset:
        return []
    key = dataset.replace("/", "___").lower()
    root = config.CACHE_DIR / "hf"
    return [q for q in root.glob("*") if q.name.lower().startswith(key)] if root.exists() else []


hf_text_dataset("hf.sst2", "SST-2", "Sentence sentiment, binary (GLUE)", "stanfordnlp/sst2", "", "sentence", "label")
hf_text_dataset("hf.text_classification", "HF text classification (any)", "Any Hugging Face dataset with a text and label column",
             "", "", "text", "label")


def _hf_seq2seq(did, title, desc, dataset, cfg, src, tgt, prefix):
    class _S(DatasetPlugin):
        id = did
        name = title
        description = desc
        task = "seq2seq"
        modality = "text"

        def params(self):
            return [Param("dataset", "str", dataset, "Hugging Face dataset id", group="Source"),
                    Param("config", "str", cfg, "Dataset config/subset", group="Source"),
                    Param("source_field", "str", src, "Dotted path to source text", group="Source"),
                    Param("target_field", "str", tgt, "Dotted path to target text", group="Source"),
                    Param("source_prefix", "str", prefix, "Prepended to every source (T5-style task prefix)", group="Source")] + HF_COMMON

        def load(self, params, seed):
            tr, va, te = _hf_splits(params, seed)
            sf, tf, pre = params["source_field"], params["target_field"], params.get("source_prefix") or ""
            conv = lambda d: [{"source": pre + str(_get_field(r, sf)), "target": str(_get_field(r, tf))} for r in d]
            info = {"n_train": len(tr), "n_val": len(va), "n_test": len(te)}
            return DataBundle("seq2seq", "text", conv(tr), conv(va), conv(te), info)

        def data_paths(self):
            return hf_cache_paths(dataset)

    return register(_S)


_hf_seq2seq("hf.opus_books_en_fr", "Opus Books EN→FR", "Book sentence translation English to French",
            "Helsinki-NLP/opus_books", "en-fr", "translation.en", "translation.fr", "translate English to French: ")
_hf_seq2seq("hf.seq2seq", "HF seq2seq (any)", "Any Hugging Face dataset with source/target text (translation, summarization)",
            "", "", "source", "target", "")
