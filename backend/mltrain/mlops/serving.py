"""Online serving + prediction logging.

Every prediction (served here, or pushed by an external deployment) becomes a row in the prediction log:
monitored features, prediction, score, latency, model version, and later its ground-truth label.
Champion/challenger: the challenger can run in 'shadow' (scored and logged, never returned) or
'canary' (a fraction of live traffic is answered by it).
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

from .. import db
from .predictors import Predictor, load_predictor


class ServingManager:
    def __init__(self):
        self._cache: dict[tuple[str, str], Predictor] = {}
        self._lock = threading.Lock()

    def predictor(self, version: dict, device: str = "cpu") -> Predictor:
        key = (version["id"], device)
        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= 8:                      # simple bound on memory
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = load_predictor(Path(version["path"]), device)
            return self._cache[key]

    def evict(self, version_id: str) -> None:
        with self._lock:
            for k in [k for k in self._cache if k[0] == version_id]:
                self._cache.pop(k)

    # ------------------------------------------------------------------ predict
    def predict(self, dep: dict, instances: list, request_ids: list[str] | None = None) -> dict:
        if dep["status"] != "active":
            raise HTTPException(409, "Deployment is paused")
        if not dep["champion_version_id"]:
            raise HTTPException(409, "Deployment has no production model")
        if not instances:
            return {"predictions": [], "request_ids": []}
        rids = request_ids or [uuid.uuid4().hex for _ in instances]
        if len(rids) != len(instances):
            raise HTTPException(400, "request_ids must match instances")
        device = (dep["config"] or {}).get("device", "cpu")
        with db.get_conn() as c:
            champ = db.one(c, "SELECT * FROM model_versions WHERE id=?", (dep["champion_version_id"],))
            chall = db.one(c, "SELECT * FROM model_versions WHERE id=?", (dep["challenger_version_id"],)) if dep["challenger_version_id"] else None
        use = champ
        if chall and dep["challenger_mode"] == "canary" and random.random() < dep["canary_fraction"]:
            use = chall
        pred = self.predictor(use, device)
        t0 = time.perf_counter()
        try:
            outs = pred.predict(instances)
            err = None
        except Exception as e:
            outs, err = [None] * len(instances), f"{type(e).__name__}: {e}"
        latency = (time.perf_counter() - t0) * 1000 / len(instances)
        rows = [self._row(dep["id"], use["id"], pred, x, o, rid, latency, err, 0) for x, o, rid in zip(instances, outs, rids)]
        if chall and dep["challenger_mode"] == "shadow" and not err:
            try:
                sp = self.predictor(chall, device)
                t1 = time.perf_counter()
                souts = sp.predict(instances)
                sl = (time.perf_counter() - t1) * 1000 / len(instances)
                rows += [self._row(dep["id"], chall["id"], sp, x, o, rid, sl, None, 1) for x, o, rid in zip(instances, souts, rids)]
            except Exception:
                pass   # a failing shadow must never affect production traffic
        self._insert(rows)
        if err:
            raise HTTPException(422, f"Prediction failed: {err}")
        return {"predictions": outs, "request_ids": rids, "model_version": use["version"], "model_name": use["model_name"]}

    @staticmethod
    def _row(dep_id, vid, pred: Predictor, x, out, rid, latency, err, shadow, ts=None):
        try:
            feats = pred.features(x)
        except Exception:
            feats = {}
        try:
            payload = pred.payload(x)
        except Exception:
            payload = None
        score = None
        if out:
            score = out.get("positive_score", out.get("score"))
        return (dep_id, vid, rid, ts or time.time(), json.dumps(feats, default=_jd), json.dumps(payload, default=_jd),
                json.dumps(out["prediction"] if out else None, default=_jd), score, latency, err, shadow, "served")

    @staticmethod
    def _insert(rows):
        with db.get_conn() as c:
            c.executemany("INSERT INTO predictions(deployment_id,version_id,request_id,ts,features,payload,prediction,score,"
                          "latency_ms,error,shadow,source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    # ------------------------------------------------------------------ external deployments
    def log_external(self, dep: dict, records: list[dict]) -> int:
        """Prediction logs pushed by a deployment hosted elsewhere (see the client SDK)."""
        with db.get_conn() as c:
            champ = db.one(c, "SELECT * FROM model_versions WHERE id=?", (dep["champion_version_id"],)) if dep["champion_version_id"] else None
            versions = {v["version"]: v["id"] for v in db.rows(c, "SELECT id, version FROM model_versions WHERE model_name=?", (dep["model_name"],))}
        pred = self.predictor(champ) if champ else None
        rows = []
        for r in records:
            x = r.get("input", r.get("features"))
            feats = r.get("features") if r.get("features") is not None and r.get("input") is None else None
            if feats is None and pred is not None and x is not None:
                try:
                    feats = pred.features(x)
                except Exception:
                    feats = {}
            out = r.get("output") or {"prediction": r.get("prediction"), "score": r.get("score")}
            payload = pred.payload(x) if pred is not None and x is not None else x
            score = out.get("positive_score", out.get("score"))
            vid = versions.get(r.get("model_version"), champ["id"] if champ else None)
            rows.append((dep["id"], vid, r.get("request_id") or uuid.uuid4().hex, float(r.get("ts") or time.time()),
                         json.dumps(feats or {}, default=_jd), json.dumps(payload, default=_jd), json.dumps(out.get("prediction"), default=_jd),
                         score, r.get("latency_ms"), r.get("error"), int(bool(r.get("shadow"))), "pushed"))
        self._insert(rows)
        return len(rows)

    @staticmethod
    def feedback(dep: dict, items: list[dict]) -> int:
        """Ground-truth labels that arrive after the prediction (joined by request_id)."""
        n = 0
        with db.get_conn() as c:
            for it in items:
                n += c.execute("UPDATE predictions SET label=?, label_ts=? WHERE deployment_id=? AND request_id=?",
                               (json.dumps(it["label"], default=_jd), time.time(), dep["id"], str(it["request_id"]))).rowcount
        return n


def _jd(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


SERVING = ServingManager()
