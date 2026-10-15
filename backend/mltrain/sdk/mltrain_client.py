"""mltrain client SDK — single file, standard library only. Copy it into any app.

Use it two ways:

1) Call a model served by the dashboard
       c = Client("http://192.168.1.50:8000", "loan-default", api_key="...")
       out = c.predict([{"age": 31, "annual_income": 52000, ...}])     # logged automatically
       c.feedback(out["request_ids"][0], label="default")              # when the truth is known

2) Model hosted elsewhere: push prediction logs so the dashboard can monitor it
       c.log(input=features_dict, prediction="repaid", score=0.12, request_id="abc", latency_ms=8.2)
       c.flush()   # logs are batched and sent in the background; flush() on shutdown

The dashboard never needs network access to your deployment — everything is pushed.
"""
from __future__ import annotations

import atexit
import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


class Client:
    def __init__(self, url: str, deployment: str, api_key: str, timeout: float = 30, batch_size: int = 200,
                 flush_interval: float = 2.0):
        self.base = url.rstrip("/") + f"/api/serve/{deployment}"
        self.key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._q: queue.Queue = queue.Queue()
        self._fq: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------ http
    def _post(self, path: str, body: dict) -> Any:
        req = urllib.request.Request(self.base + path, data=json.dumps(body, default=str).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-API-Key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code} {e.read().decode(errors='replace')[:500]}") from None

    # ------------------------------------------------------------------ served models
    def predict(self, instances: list, request_ids: list[str] | None = None) -> dict:
        """Synchronous prediction from a model served by the dashboard."""
        return self._post("/predict", {"instances": instances, "request_ids": request_ids})

    def info(self) -> Any:
        req = urllib.request.Request(self.base + "/info", headers={"X-API-Key": self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    # ------------------------------------------------------------------ logging & feedback (batched, async)
    def log(self, input: Any, prediction: Any, score: float | None = None, request_id: str | None = None,
            latency_ms: float | None = None, model_version: int | None = None, error: str | None = None, **extra) -> str:
        """Record a prediction made by a model hosted elsewhere. Returns the request id."""
        rid = request_id or uuid.uuid4().hex
        self._q.put({"request_id": rid, "input": input, "prediction": prediction, "score": score, "latency_ms": latency_ms,
                     "model_version": model_version, "error": error, "ts": time.time(), **extra})
        return rid

    def feedback(self, request_id: str, label: Any) -> None:
        """Report the ground truth for an earlier prediction (any time later)."""
        self._fq.put({"request_id": request_id, "label": label})

    def flush(self) -> None:
        self._drain(self._q, "/log", "records")
        self._drain(self._fq, "/feedback", "items")

    def _drain(self, q: queue.Queue, path: str, key: str) -> None:
        while not q.empty():
            batch = []
            while not q.empty() and len(batch) < self.batch_size:
                batch.append(q.get_nowait())
            if batch:
                for attempt in range(3):
                    try:
                        self._post(path, {key: batch})
                        break
                    except Exception:
                        time.sleep(0.5 * (attempt + 1))

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.flush_interval)
            try:
                self.flush()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self.flush()
        except Exception:
            pass
