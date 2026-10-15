"""RunContext: the object a model plugin uses to talk to the dashboard while training."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Callable


class RunContext:
    FLUSH_INTERVAL = 1.0      # seconds between metric writes
    LIVE_FRAME_INTERVAL = 0.12  # seconds between live preview frames (~8 fps)

    def __init__(self, run: dict, session: dict, run_dir: Path, info: dict | None = None,
                 init_path: Path | None = None):
        self.run_id: str = run["id"]
        self.session_id: str = session["id"]
        self.seed: int = run["seed"]
        self.params: dict = session["model_params"]           # model + training params
        self.problem_params: dict = session["problem_params"]
        self.device: str = session["device"]
        self.mode: str = session.get("mode") or "train"       # train | evaluate
        self.options: dict = session.get("options") or {}
        self.resume: bool = bool(run.get("resume"))
        self.run_dir = run_dir
        self.model_dir = run_dir / "model"                     # final saved model (kept if save_model)
        self.checkpoint_dir = run_dir / "checkpoint"           # resumable state (removed on completion)
        self.media_dir = run_dir / "media"
        self.init_path: Path | None = init_path                # saved model to start from / evaluate
        self.info: dict = info or {}                           # dataset info (filled by worker)
        self._buf: list[tuple] = []
        self._last_flush = time.time()
        self._last_frame = 0.0
        self._stop = False
        from .. import db                                      # (imported lazily: standalone exports have no DB)
        self._db = db
        self._conn = self._db.connect()
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT, self._on_sigterm)

    # ---- options ----------------------------------------------------------
    @property
    def save_model(self) -> bool:
        return self.mode == "train" and bool(self.options.get("save_model", True))

    @property
    def snapshots(self) -> bool:
        return bool(self.options.get("snapshots", True))

    def opt(self, name: str, default=None):
        return self.options.get(name, default)

    def snapshot_schedule(self, total: float, count: int | None = None) -> Callable[[float], bool]:
        """Returns due(step) -> True once each time `step` crosses one of `count` evenly spaced points in (0, total]."""
        count = int(count or self.options.get("snapshot_count", 10))
        if not self.snapshots or count <= 0 or total <= 0:
            return lambda step: False
        marks = sorted({max(1, round(total * (i + 1) / count)) for i in range(count)})
        state = {"i": 0}

        def due(step: float) -> bool:
            hit = False
            while state["i"] < len(marks) and step >= marks[state["i"]]:
                state["i"] += 1
                hit = True
            return hit

        return due

    # ---- logging ----------------------------------------------------------
    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record scalar metrics at a step (epoch, timestep, ...).

        Keys use 'split/name' form, e.g. 'train/loss', 'val/acc', 'rollout/ep_reward'.
        Curves with the same name across splits are plotted together.
        """
        t = time.time()
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                self._buf.append((self.run_id, int(step), k, v, t))
        if t - self._last_flush > self.FLUSH_INTERVAL:
            self.flush()

    def log_artifact(self, name: str, kind: str, data: Any) -> None:
        """Store a final, non-scalar result: kind = confusion_matrix | table | text | json."""
        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts(run_id,name,kind,data) VALUES (?,?,?,?)",
            (self.run_id, name, kind, json.dumps(data, default=_json_default)),
        )

    def log_media(self, step: int, name: str, kind: str, data: Any = None, file: Path | None = None) -> None:
        """Record a learning snapshot at `step`.

        kind: video | image (pass `file`, a path inside run_dir) or a structured snapshot
        (text_samples | decision_boundary | curve_fit | image_predictions | confusion_matrix) passed as `data`.
        """
        rel = str(Path(file).relative_to(self.run_dir)) if file else None
        self._conn.execute("INSERT INTO media(run_id,step,name,kind,path,data,t) VALUES (?,?,?,?,?,?,?)",
                           (self.run_id, int(step), name, kind, rel,
                            json.dumps(data, default=_json_default) if data is not None else None, time.time()))

    def live_frame(self, frame) -> None:
        """Publish an RGB frame (H x W x 3 uint8) as the run's live preview (throttled)."""
        now = time.time()
        if frame is None or now - self._last_frame < self.LIVE_FRAME_INTERVAL:
            return
        self._last_frame = now
        import cv2
        h, w = frame.shape[:2]
        if w > 480:
            frame = cv2.resize(frame, (480, int(h * 480 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            tmp = self.run_dir / "live.jpg.tmp"
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, self.run_dir / "live.jpg")

    def set_progress(self, frac: float, text: str = "") -> None:
        self._db.update_run(self._conn, self.run_id, progress=max(0.0, min(1.0, float(frac))), progress_text=text)

    def flush(self) -> None:
        if self._buf:
            self._conn.executemany("INSERT INTO metrics(run_id,step,key,value,t) VALUES (?,?,?,?,?)", self._buf)
            self._buf.clear()
        self._last_flush = time.time()
        # also pick up stop requests made via the API
        row = self._conn.execute("SELECT status FROM runs WHERE id=?", (self.run_id,)).fetchone()
        if row and row["status"] == "stopping":
            self._stop = True

    def truncate_after(self, step: int) -> None:
        """Resume support: drop metrics/snapshots logged after the checkpoint step."""
        self.flush()
        self._conn.execute("DELETE FROM metrics WHERE run_id=? AND step>?", (self.run_id, step))
        self._conn.execute("DELETE FROM media WHERE run_id=? AND step>?", (self.run_id, step))

    # ---- control ----------------------------------------------------------
    def should_stop(self) -> bool:
        """Poll this in training loops; True once the user pressed Stop."""
        if time.time() - self._last_flush > self.FLUSH_INTERVAL:
            self.flush()
        return self._stop

    def _on_sigterm(self, *_):
        self._stop = True

    def checkpoint_path(self, name: str = "model.pt") -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / name

    def model_path(self, name: str) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir / name

    @property
    def keep(self) -> str:
        """best | final | both — which weights to save (model/ holds best unless keep == 'final'; model_final/ the final ones)."""
        return self.options.get("keep", "both") if self.save_model else "none"

    @property
    def final_dir(self) -> Path:
        return self.run_dir / "model_final"

    def final_path(self, name: str) -> Path:
        self.final_dir.mkdir(parents=True, exist_ok=True)
        return self.final_dir / name

    def media_path(self, name: str) -> Path:
        self.media_dir.mkdir(parents=True, exist_ok=True)
        return self.media_dir / name

    def clear_checkpoint(self) -> None:
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

    def print(self, *args) -> None:
        print(*args, flush=True)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)
