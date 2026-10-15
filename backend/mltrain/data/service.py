"""Demo stream process management (survives server restarts, like the MLOps traffic simulator)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import psutil

from .. import config

STATE = config.DATA_DIR / "live" / "stream.json"
DEMO_DB = config.DATA_DIR / "live" / "loan_applications.db"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE and "mltrain.data.demo_stream" in " ".join(p.cmdline())
    except psutil.Error:
        return False


def state() -> dict | None:
    if not STATE.exists():
        return None
    s = json.loads(STATE.read_text())
    return s if _alive(s.get("pid")) else None


def start(rate: float = 120 / 3600, incident: str = "none", label_delay: float = 60) -> dict:
    stop()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    log = STATE.parent / "stream.log"
    offset = log.stat().st_size if log.exists() else 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(config.ROOT / "backend"), env.get("PYTHONPATH", "")])
    with open(STATE.parent / "stream.log", "ab") as logf:
        p = subprocess.Popen([sys.executable, "-m", "mltrain.data.demo_stream", "--db", str(DEMO_DB), "--rate", str(rate),
                              "--incident", incident, "--label-delay", str(label_delay)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    s = {"pid": p.pid, "rate": rate, "incident": incident, "label_delay": label_delay, "started": time.time(), "log_offset": offset}
    STATE.write_text(json.dumps(s))
    return s


def stop() -> bool:
    s = json.loads(STATE.read_text()) if STATE.exists() else None
    STATE.unlink(missing_ok=True)
    if not s or not _alive(s.get("pid")):
        return False
    os.kill(s["pid"], signal.SIGTERM)
    for _ in range(50):
        if not _alive(s["pid"]):
            return True
        time.sleep(0.1)
    os.kill(s["pid"], signal.SIGKILL)
    return True


def wait_for_backfill(timeout: float = 180) -> None:
    """The stream backfills its history first; wait until this run logs that it is streaming."""
    log = STATE.parent / "stream.log"
    offset = json.loads(STATE.read_text()).get("log_offset", 0) if STATE.exists() else 0
    t0 = time.time()
    while time.time() - t0 < timeout:
        if log.exists():
            with open(log, "rb") as f:
                f.seek(offset)
                if b"streaming" in f.read():
                    return
        if not state():
            raise RuntimeError("demo stream exited during backfill; see data/live/stream.log")
        time.sleep(1)
    raise TimeoutError("demo database backfill did not finish in time")
