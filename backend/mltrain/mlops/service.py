"""Background loop: scheduled monitoring snapshots, automation triggers and progress, demo traffic processes."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time

from .. import config, db
from . import automation
from .deployments import cfg_of, run_monitor

log = logging.getLogger("mltrain.mlops")


class MLOpsService:
    TICK = 15.0

    def __init__(self):
        self._task: asyncio.Task | None = None
        self.pending_demo: str | None = None   # training session of the one-click demo
        self.traffic: dict[str, dict] = {}     # deployment id -> {proc, scenario, started, ...}

    def start(self) -> None:
        self.adopt_traffic()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()   # demo traffic keeps running; it is re-adopted on the next start

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                log.exception("mlops tick failed")
            await asyncio.sleep(self.TICK)

    def tick(self) -> None:
        now = time.time()
        with db.get_conn() as c:
            deps = db.rows(c, "SELECT * FROM deployments WHERE status='active'")
            running = db.rows(c, "SELECT * FROM automation_runs WHERE status='running'")
        for dep in deps:
            cfg = cfg_of(dep)
            if dep["champion_version_id"] and (not dep["last_monitor_at"] or now - dep["last_monitor_at"] >= cfg["interval_minutes"] * 60):
                try:
                    snap = run_monitor(dep["id"])
                    automation.check_triggers(dep, snap)
                except Exception:
                    log.exception("monitoring %s failed", dep["id"])
        for run in running:
            try:
                automation.progress(run)
            except Exception:
                log.exception("automation run %s failed", run["id"])
        if self.pending_demo:
            self._finish_demo()
        for dep_id, t in list(self.traffic.items()):
            if not self._alive(t["proc"]):
                self.traffic.pop(dep_id)
                self._save_traffic()

    def _finish_demo(self) -> None:
        """When the demo model finishes training: register it and deploy it with demo-friendly settings."""
        from ..analysis import session_status
        from .deployments import create
        from .registry import register_version
        with db.get_conn() as c:
            st = session_status(db.rows(c, "SELECT status FROM runs WHERE session_id=?", (self.pending_demo,)))
        if st in ("running", "queued"):
            return
        sid, self.pending_demo = self.pending_demo, None
        if st != "completed":
            log.warning("demo training session %s ended as %s", sid, st)
            return
        v = register_version("loan-default", sid, None, "Demo credit-risk model (XGBoost)")
        with db.get_conn() as c:
            create(c, "loan-default", v["id"], "internal",
                   {"window_minutes": 10, "interval_minutes": 1, "min_samples": 30},
                   {"enabled": True, "triggers": ["drift", "performance"], "cooldown_hours": 0.2,
                    "data": {"source": "feedback", "feedback_days": 1, "min_labeled": 400}, "min_improvement": 0.005,
                    "auto_promote": False})

    # ------------------------------------------------------------------ demo traffic
    # Simulator processes outlive server restarts; their state is persisted and re-adopted on startup.
    STATE_FILE = config.DATA_DIR / "traffic.json"

    def _save_traffic(self) -> None:
        import json
        self.STATE_FILE.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("proc", "logf")} for k, v in self.traffic.items()}))

    def adopt_traffic(self) -> None:
        import json

        import psutil
        if not self.STATE_FILE.exists():
            return
        for dep_id, t in json.loads(self.STATE_FILE.read_text()).items():
            try:
                p = psutil.Process(t["pid"])
                if "mltrain.mlops.simulator" in " ".join(p.cmdline()):
                    self.traffic[dep_id] = {**t, "proc": p}
            except psutil.Error:
                pass
        self._save_traffic()

    @staticmethod
    def _alive(proc) -> bool:
        if hasattr(proc, "poll"):
            return proc.poll() is None
        try:
            return proc.is_running() and proc.status() != "zombie"
        except Exception:
            return False

    def start_traffic(self, dep: dict, url: str, scenario: str, rps: float, ramp_minutes: float, label_delay: float,
                      label_fraction: float) -> None:
        self.stop_traffic(dep["id"], quiet=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(config.ROOT / "backend"), env.get("PYTHONPATH", "")])
        with open(config.DATA_DIR / f"traffic_{dep['id']}.log", "ab") as logf:
            proc = subprocess.Popen([sys.executable, "-m", "mltrain.mlops.simulator", "--url", url, "--deployment", dep["id"],
                                     "--key", dep["api_key"], "--scenario", scenario, "--rps", str(rps), "--ramp-minutes", str(ramp_minutes),
                                     "--label-delay", str(label_delay), "--label-fraction", str(label_fraction)],
                                    stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.traffic[dep["id"]] = {"proc": proc, "pid": proc.pid, "scenario": scenario, "rps": rps, "ramp_minutes": ramp_minutes,
                                   "label_delay": label_delay, "started": time.time()}
        self._save_traffic()
        with db.get_conn() as c:
            from .deployments import event
            event(c, dep["id"], "traffic", f"Demo traffic started: scenario '{scenario}', {rps} req/s, ramp {ramp_minutes} min")

    def stop_traffic(self, dep_id: str, quiet: bool = False) -> bool:
        t = self.traffic.pop(dep_id, None)
        self._save_traffic()
        if not t:
            return False
        try:
            os.kill(t["pid"], signal.SIGTERM)
            for _ in range(50):
                if not self._alive(t["proc"]):
                    break
                time.sleep(0.2)
            else:
                os.kill(t["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        if hasattr(t["proc"], "wait"):
            try:
                t["proc"].wait(timeout=1)
            except Exception:
                pass
        if not quiet:
            with db.get_conn() as c:
                from .deployments import event
                event(c, dep_id, "traffic", "Demo traffic stopped")
        return True

    def traffic_state(self, dep_id: str) -> dict | None:
        t = self.traffic.get(dep_id)
        if not t or not self._alive(t["proc"]):
            return None
        return {k: v for k, v in t.items() if k not in ("proc", "pid")} | {"elapsed_min": (time.time() - t["started"]) / 60}


MLOPS = MLOpsService()
