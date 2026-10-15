"""Job queue: launches queued runs as worker subprocesses within GPU/CPU concurrency limits."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time

import psutil

from . import config, db

log = logging.getLogger("mltrain.scheduler")
STOP_GRACE_S = 60  # seconds to wait for a graceful stop before SIGTERM, then SIGKILL


class _Job:
    def __init__(self, run_id: str, resource: str, popen: subprocess.Popen | None = None, pid: int | None = None):
        self.run_id = run_id
        self.resource = resource
        self.popen = popen
        self.pid = popen.pid if popen else pid
        self.stop_seen_at: float | None = None
        self.term_sent_at: float | None = None
        self.logf = None

    def alive(self) -> bool:
        if self.popen is not None:
            return self.popen.poll() is None
        try:
            p = psutil.Process(self.pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def signal(self, sig, group: bool = True) -> None:
        try:
            if group:
                os.killpg(self.pid, sig)    # worker + its dataloader children
            else:
                os.kill(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


class Scheduler:
    def __init__(self, cfg: dict):
        from .power import PowerManager
        self.cfg = cfg
        self.power = PowerManager(cfg)
        self.jobs: dict[str, _Job] = {}
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._recover()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        self.power.release()

    def _recover(self) -> None:
        """After a server restart, adopt workers still alive and fail the ones that died."""
        with db.get_conn() as c:
            for r in db.rows(c, "SELECT r.id, r.pid, s.device FROM runs r JOIN sessions s ON s.id=r.session_id "
                                "WHERE r.status IN ('running','stopping')"):
                alive = False
                if r["pid"]:
                    try:
                        alive = "mltrain.worker" in " ".join(psutil.Process(r["pid"]).cmdline())
                    except psutil.Error:
                        pass
                if alive:
                    self.jobs[r["id"]] = _Job(r["id"], _resource(r["device"]), pid=r["pid"])
                else:
                    db.update_run(c, r["id"], status="failed", ended_at=db.now(), error="Worker died (server restart?)")

    async def _loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("scheduler tick failed")
            await asyncio.sleep(1.0)

    # ---------------------------------------------------------------- tick
    def tick(self) -> None:
        with db.get_conn() as c:
            self._reap(c)
            self._handle_stops(c)
            self._launch(c)
            waiting = c.execute("SELECT COUNT(*) FROM runs WHERE status='queued'").fetchone()[0]
        # keep the machine from suspending while anything runs or waits (incl. jobs scheduled for later)
        if self.jobs:
            self.power.update(True, f"Training {len(self.jobs)} run(s)")
        elif waiting:
            self.power.update(True, f"{waiting} queued / scheduled run(s)")
        else:
            self.power.update(False, "")

    def _reap(self, c) -> None:
        for run_id, job in list(self.jobs.items()):
            if job.alive():
                continue
            if job.logf:
                job.logf.close()
            del self.jobs[run_id]
            row = db.one(c, "SELECT status FROM runs WHERE id=?", (run_id,))
            if row and row["status"] in ("running", "queued"):
                code = job.popen.returncode if job.popen else None
                db.update_run(c, run_id, status="failed", ended_at=db.now(),
                              error=f"Worker exited unexpectedly (code {code}). Out of memory? See the log.")
            elif row and row["status"] == "stopping":
                db.update_run(c, run_id, status="stopped", ended_at=db.now(), progress_text="stopped")

    def _handle_stops(self, c) -> None:
        for r in db.rows(c, "SELECT id FROM runs WHERE status='stopping'"):
            job = self.jobs.get(r["id"])
            if job is None:
                db.update_run(c, r["id"], status="stopped", ended_at=db.now())
            elif job.stop_seen_at is None:
                # the worker polls the DB, sees 'stopping' and finishes gracefully (evaluates, saves)
                job.stop_seen_at = time.time()
            elif job.term_sent_at is None and time.time() - job.stop_seen_at > STOP_GRACE_S:
                job.term_sent_at = time.time()
                job.signal(signal.SIGTERM, group=False)
            elif job.term_sent_at and time.time() - job.term_sent_at > STOP_GRACE_S:
                job.signal(signal.SIGKILL)

    def _launch(self, c) -> None:
        limits = {"gpu": int(self.cfg.get("max_gpu_jobs", 1)), "cpu": int(self.cfg.get("max_cpu_jobs", 2))}
        used = {"gpu": 0, "cpu": 0}
        for j in self.jobs.values():
            used[j.resource] += 1
        # highest priority first, then first-come-first-served; scheduled sessions wait for their start time
        queued = db.rows(c, "SELECT r.id, s.device, s.framework FROM runs r JOIN sessions s ON s.id=r.session_id "
                            "WHERE r.status='queued' AND (s.scheduled_at IS NULL OR s.scheduled_at<=?) "
                            "ORDER BY s.priority DESC, r.created_at, r.seed", (time.time(),))
        for r in queued:
            res = _resource(r["device"])
            if used[res] >= limits[res]:
                continue
            self._spawn(r["id"], res, r["device"], r["framework"])
            used[res] += 1

    def _spawn(self, run_id: str, resource: str, device: str, framework: str | None = None) -> None:
        run_dir = config.RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(config.ROOT / "backend"), str(config.ROOT), env.get("PYTHONPATH", "")])
        env["PYTHONUNBUFFERED"] = "1"
        # hide the GPU from CPU runs so they can't grab VRAM by accident; XGBoost's GPU build
        # crashes on multiclass problems without a visible device even with device='cpu'
        if device == "cpu" and framework != "xgboost":
            env["CUDA_VISIBLE_DEVICES"] = ""
        logf = open(run_dir / "log.txt", "ab")
        p = subprocess.Popen([sys.executable, "-m", "mltrain.worker", run_id], stdout=logf, stderr=subprocess.STDOUT,
                             env=env, cwd=str(config.ROOT), start_new_session=True)
        job = _Job(run_id, resource, popen=p)
        job.logf = logf
        self.jobs[run_id] = job
        with db.get_conn() as c:
            db.update_run(c, run_id, pid=p.pid)
        log.info("launched run %s (pid %s, %s)", run_id, p.pid, resource)

    def queue_state(self) -> dict:
        used = {"gpu": 0, "cpu": 0}
        for j in self.jobs.values():
            used[j.resource] += 1
        with db.get_conn() as c:
            row = c.execute("SELECT SUM(s.scheduled_at IS NULL OR s.scheduled_at<=?), SUM(s.scheduled_at>?) FROM runs r "
                            "JOIN sessions s ON s.id=r.session_id WHERE r.status='queued'", (time.time(), time.time())).fetchone()
        return {"running": used, "limits": {"gpu": self.cfg.get("max_gpu_jobs"), "cpu": self.cfg.get("max_cpu_jobs")},
                "queued": row[0] or 0, "scheduled": row[1] or 0, "power": self.power.state()}


def _resource(device: str) -> str:
    return "gpu" if device.startswith("cuda") else "cpu"
