"""Load a saved model (registry version or run model dir) as a uniform Predictor.

Inputs per modality:   tabular -> dict of raw column values     image -> base64 PNG/JPEG or pixel array
                       text    -> string (or {"text": ...})     rl    -> observation (list / array)
Outputs: {"prediction": label | value | text | action, "score": confidence or value, "proba": {...}?}
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .features import decode_image, image_stats, obs_features, text_of, text_stats


class Predictor:
    kind = "tabular"

    def __init__(self, path: Path, meta: dict, device: str = "cpu"):
        self.path, self.meta, self.device = Path(path), meta, device
        self.task = meta["task"]
        info = meta.get("info") or {}
        self.classes: list[str] | None = info.get("class_names")
        self.lock = threading.Lock()
        self.vocab: set | None = None    # text models: training vocabulary (set from the profile)

    # --- override ---------------------------------------------------------
    def _predict(self, instances: list) -> list[dict]:
        raise NotImplementedError

    def features(self, x) -> dict:
        return {}

    def payload(self, x) -> Any:
        return x

    # --- public -----------------------------------------------------------
    def predict(self, instances: list) -> list[dict]:
        with self.lock:
            return self._predict(instances)

    def _cls_out(self, proba: np.ndarray) -> list[dict]:
        out = []
        names = self.classes or [str(i) for i in range(proba.shape[1])]
        for p in proba:
            i = int(np.argmax(p))
            out.append({"prediction": names[i], "class_index": i, "score": float(p[i]),
                        "proba": {names[k]: round(float(v), 5) for k, v in enumerate(p)},
                        # P(positive) for binary problems: the standard score to monitor
                        **({"positive_score": float(p[1])} if len(p) == 2 else {})})
        return out


# ---------------------------------------------------------------- tabular
class TabularMixin:
    def _prep(self, instances):
        from ..preprocessing import TabularPreprocessor
        pre_d = (self.meta.get("info") or {}).get("preprocessor")
        if not pre_d:
            raise ValueError("This model was saved before raw-input preprocessing was recorded; retrain it to serve it")
        return TabularPreprocessor.from_dict(pre_d).transform([dict(x) for x in instances])

    def features(self, x):
        return {k: v for k, v in dict(x).items()}


class SklearnPredictor(TabularMixin, Predictor):
    def __init__(self, path, meta, device="cpu"):
        super().__init__(path, meta, device)
        import joblib
        self.est = joblib.load(self.path / "model.joblib")

    def _predict(self, instances):
        X = self._prep(instances)
        if self.task == "classification":
            if hasattr(self.est, "predict_proba"):
                return self._cls_out(self.est.predict_proba(X))
            idx = self.est.predict(X)
            k = len(self.classes or []) or int(idx.max()) + 1
            return self._cls_out(np.eye(k)[idx.astype(int)])
        return [{"prediction": float(v), "score": float(v)} for v in self.est.predict(X)]


# ---------------------------------------------------------------- torch
class TorchPredictor(Predictor):
    def __init__(self, path, meta, device="cpu"):
        super().__init__(path, meta, device)
        import torch

        from ..plugins.registry import get_registry
        plugin = get_registry().get("model", meta["model_id"])
        info = dict(meta.get("info") or {})
        self.kind = meta["modality"]
        info["output_dim"] = info.get("n_classes", 1) if self.task == "classification" else 1
        if self.kind == "tabular":
            info.setdefault("input_shape", [info["n_features"]])
        info["input_dim"] = int(np.prod(info["input_shape"]))
        self.info = info
        self.model = plugin.build(meta["model_params"], info)
        self.model.load_state_dict(torch.load(self.path / "model.pt", map_location=device, weights_only=True))
        self.model.to(device).eval()

    def _tensor(self, instances):
        import torch
        if self.kind == "tabular":
            return torch.as_tensor(TabularMixin._prep(self, instances), device=self.device)
        norm = self.info.get("image_norm") or {"mean": [0.5], "std": [0.5]}
        c, h, w = self.info["input_shape"]
        xs = []
        for x in instances:
            a = self._image(x, c, h, w)
            a = (a - np.asarray(norm["mean"], np.float32)[:, None, None]) / np.asarray(norm["std"], np.float32)[:, None, None]
            xs.append(a)
        return torch.as_tensor(np.stack(xs), device=self.device)

    @staticmethod
    def _image(x, c, h, w) -> np.ndarray:
        """Decode -> resize -> channels -> CHW float in [0,1] (the same steps as ToTensor)."""
        import cv2
        a = decode_image(x).astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
            a = a.transpose(1, 2, 0)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        if a.ndim == 2:
            a = a[:, :, None]
        if c == 1 and a.shape[-1] == 3:
            a = a.mean(-1, keepdims=True)
        elif c == 3 and a.shape[-1] == 1:
            a = np.repeat(a, 3, -1)
        return a.transpose(2, 0, 1)

    def _predict(self, instances):
        import torch
        with torch.no_grad():
            out = self.model(self._tensor(instances)).float()
        if self.task == "classification":
            return self._cls_out(torch.softmax(out, -1).cpu().numpy())
        return [{"prediction": float(v), "score": float(v)} for v in out.squeeze(-1).cpu().numpy()]

    def features(self, x):
        if self.kind == "tabular":
            return dict(x)
        c, h, w = self.info["input_shape"]
        return image_stats(self._image(x, c, h, w).transpose(1, 2, 0).squeeze())

    def payload(self, x):
        return x if self.kind == "tabular" else None   # don't store raw images in the log


# ---------------------------------------------------------------- hugging face
class HFPredictor(Predictor):
    kind = "text"

    def __init__(self, path, meta, device="cpu"):
        super().__init__(path, meta, device)
        from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.path)
        cls = AutoModelForSequenceClassification if self.task == "classification" else AutoModelForSeq2SeqLM
        self.model = cls.from_pretrained(self.path).to(device).eval()
        self.p = meta["model_params"]

    def _predict(self, instances):
        import torch
        texts = [text_of(x) for x in instances]
        if self.task == "classification":
            enc = self.tok(texts, truncation=True, max_length=self.p.get("max_length", 256), padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                return self._cls_out(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        enc = self.tok(texts, truncation=True, max_length=self.p.get("max_source_length", 256), padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, num_beams=self.p.get("num_beams", 1), max_new_tokens=self.p.get("max_target_length", 128),
                                      output_scores=True, return_dict_in_generate=True)
        seqs = self.tok.batch_decode(out.sequences, skip_special_tokens=True)
        scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True) if self.p.get("num_beams", 1) == 1 else None
        res = []
        for i, s in enumerate(seqs):
            conf = float(torch.exp(scores[i][scores[i] > -1e9].mean())) if scores is not None else None
            res.append({"prediction": s, "score": conf, "output_length": len(s.split())})
        return res

    def features(self, x):
        return text_stats(text_of(x), self.vocab)

    def payload(self, x):
        return text_of(x)[:2000]


# ---------------------------------------------------------------- reinforcement learning
class SB3Predictor(Predictor):
    kind = "rl"

    def __init__(self, path, meta, device="cpu"):
        super().__init__(path, meta, device)
        from ..plugins.registry import get_registry
        plugin = get_registry().get("model", meta["model_id"])
        self.model = plugin.algo_cls.load(self.path / "model.zip", device=device)
        self.discrete = hasattr(self.model.action_space, "n")

    def _predict(self, instances):
        obs = np.asarray([np.asarray(x.get("obs") if isinstance(x, dict) else x, dtype=np.float32) for x in instances])
        actions, _ = self.model.predict(obs, deterministic=True)
        out = []
        for a in actions:
            a = a.tolist() if hasattr(a, "tolist") else a
            out.append({"prediction": int(a) if self.discrete else a,
                        "score": float(a) if isinstance(a, (int, float)) else float(np.linalg.norm(a))})
        return out

    def features(self, x):
        return obs_features(x.get("obs") if isinstance(x, dict) else x)

    def payload(self, x):
        return x.get("obs") if isinstance(x, dict) else x


FRAMEWORKS = {"sklearn": SklearnPredictor, "xgboost": SklearnPredictor, "lightgbm": SklearnPredictor,
              "torch": TorchPredictor, "hf": HFPredictor, "sb3": SB3Predictor}


def load_predictor(path: Path, device: str = "cpu") -> Predictor:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    cls = FRAMEWORKS.get(meta.get("framework"))
    if cls is None:
        from ..plugins.registry import get_registry
        plugin = get_registry().get("model", meta["model_id"])
        if hasattr(plugin, "load_predictor"):          # custom plugins can provide their own
            return plugin.load_predictor(path, meta, device)
        raise ValueError(f"No predictor for framework '{meta.get('framework')}'; add load_predictor() to the plugin")
    p = cls(path, meta, device)
    if p.kind == "tabular" and meta.get("modality") != "tabular" and cls is not TorchPredictor:
        p.kind = meta.get("modality")
    if (path / "profile.json").exists() and p.kind == "text":
        vocab = json.loads((path / "profile.json").read_text()).get("vocab")
        p.vocab = set(vocab) if vocab else None
    return p
