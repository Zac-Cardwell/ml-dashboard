"""Keep the computer awake while training — without keeping the screen on.

A logind *sleep* inhibitor (systemd-inhibit --what=sleep) stops automatic suspend; it deliberately does NOT
inhibit 'idle', so the desktop still dims, locks and turns the display off as usual. KDE Plasma, GNOME and
others honor it. The inhibited command is `tail --pid=<server pid>`, so if the dashboard dies the inhibitor
disappears with it and the machine can sleep normally again.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

log = logging.getLogger("mltrain.power")


class PowerManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.since: float | None = None
        self.reason = ""
        self.available = shutil.which("systemd-inhibit") is not None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("keep_awake", True))

    def update(self, busy: bool, reason: str) -> None:
        want = self.enabled and busy and self.available
        alive = self.proc is not None and self.proc.poll() is None
        if want and not alive:
            try:
                self.proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=sleep", "--mode=block", "--who=ML Dashboard",
                     f"--why={reason} (the screen may still turn off)", "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.since, self.reason = time.time(), reason
                log.info("sleep inhibitor on: %s", reason)
            except OSError as e:
                log.warning("could not inhibit sleep: %s", e)
                self.available = False
        elif not want and alive:
            self.release()
        elif want:
            self.reason = reason

    def release(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            log.info("sleep inhibitor off")
        self.proc, self.since, self.reason = None, None, ""

    def state(self) -> dict:
        active = self.proc is not None and self.proc.poll() is None
        return {"enabled": self.enabled, "available": self.available, "active": active,
                "reason": self.reason if active else "", "since": self.since if active else None}


def display_off() -> str:
    """Turn the monitors off now (they wake on mouse/keyboard input). Returns a status message."""
    env = dict(os.environ)
    desktop = env.get("XDG_CURRENT_DESKTOP", "").upper()
    cmds = []
    if shutil.which("kscreen-doctor"):
        cmds.append(["kscreen-doctor", "--dpms", "off"])                       # KDE Plasma (Wayland)
    if "GNOME" in desktop and shutil.which("busctl"):
        cmds.append(["busctl", "--user", "set-property", "org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig",
                     "org.gnome.Mutter.DisplayConfig", "PowerSaveMode", "i", "1"])  # GNOME
    if env.get("DISPLAY") and shutil.which("xset"):
        cmds.append(["xset", "dpms", "force", "off"])                            # X11
    if not cmds:
        raise RuntimeError("No supported display-power tool found (kscreen-doctor, GNOME busctl or xset)")
    errors = []
    for cmd in cmds:
        try:
            # started detached: kscreen-doctor blocks until the screen wakes on some setups
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            try:
                _, err = p.communicate(timeout=3)
                if p.returncode == 0:
                    return f"Display turned off ({cmd[0]})"
                errors.append(f"{cmd[0]}: {err.decode(errors='replace').strip()[:200]}")
            except subprocess.TimeoutExpired:
                return f"Display turned off ({cmd[0]})"
        except OSError as e:
            errors.append(f"{cmd[0]}: {e}")
    raise RuntimeError("; ".join(errors) or "failed")
