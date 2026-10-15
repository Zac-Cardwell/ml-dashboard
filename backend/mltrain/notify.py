"""Notifications: Discord, Slack, ntfy (phone push), email (SMTP) and generic webhooks.

Channels are configured in Settings (stored in data/config.json). Each channel subscribes to event types.
Messages are sent from a background thread with retries; repeated events with the same key are rate-limited.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import smtplib
import socket
import ssl
import threading
import time
import urllib.request
from email.message import EmailMessage

log = logging.getLogger("mltrain.notify")

EVENTS = {
    "session_completed": "Training session finished",
    "session_failed": "Training session failed or partly failed",
    "group_completed": "Sweep / benchmark / hyperparameter search finished",
    "alert_critical": "Deployment critical alert",
    "alert_warning": "Deployment warning",
    "automation": "Automated retraining: challenger ready, promoted or rejected",
    "data_source": "Live data source unhealthy (stale, schema change, drift, quality)",
    "schedule": "Scheduled training finished",
}
DEFAULT_EVENTS = ["session_completed", "session_failed", "group_completed", "alert_critical", "automation", "data_source", "schedule"]
CHANNEL_TYPES = ("discord", "slack", "ntfy", "email", "webhook")
RATE_LIMIT_S = 600          # same event key at most once per 10 minutes
SEVERITY_COLOR = {"info": 0x2A78D6, "success": 0x0CA30C, "warning": 0xFAB219, "critical": 0xD03B3B}


class Notifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.q: queue.Queue = queue.Queue()
        self.recent: dict[str, float] = {}
        self.history: list[dict] = []           # last sends (for the Settings page)
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------------ config helpers
    @property
    def channels(self) -> list[dict]:
        return list((self.cfg.get("notifications") or {}).get("channels") or [])

    def dashboard_url(self) -> str:
        url = (self.cfg.get("notifications") or {}).get("dashboard_url")
        if url:
            return url.rstrip("/")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
        except OSError:
            ip = "localhost"
        return f"http://{ip}:{os.environ.get('MLTRAIN_PORT', '8000')}"

    # ------------------------------------------------------------------ public
    def notify(self, event: str, title: str, message: str, path: str = "", severity: str = "info",
               key: str | None = None, data: dict | None = None) -> None:
        """Queue a notification for every enabled channel subscribed to `event`."""
        now = time.time()
        if key:
            if now - self.recent.get(key, 0) < RATE_LIMIT_S:
                return
            self.recent[key] = now
        targets = [ch for ch in self.channels if ch.get("enabled", True) and event in (ch.get("events") or DEFAULT_EVENTS)]
        if not targets:
            return
        msg = {"event": event, "title": title, "message": message, "url": self.dashboard_url() + path if path else self.dashboard_url(),
               "severity": severity, "data": data or {}, "ts": now}
        for ch in targets:
            self.q.put((ch, msg))

    def send_test(self, ch: dict) -> None:
        """Synchronous test send (raises on failure)."""
        self._send(ch, {"event": "test", "title": "ML Dashboard test notification", "severity": "success",
                        "message": "Notifications are working. You'll get messages here for the events you selected.",
                        "url": self.dashboard_url(), "data": {}, "ts": time.time()})

    # ------------------------------------------------------------------ delivery
    def _worker(self) -> None:
        while True:
            ch, msg = self.q.get()
            for attempt in range(3):
                try:
                    self._send(ch, msg)
                    self._record(ch, msg, None)
                    break
                except Exception as e:
                    if attempt == 2:
                        log.warning("notification via %s failed: %s", ch.get("name") or ch.get("type"), e)
                        self._record(ch, msg, str(e))
                    time.sleep(2 * (attempt + 1))

    def _record(self, ch, msg, err) -> None:
        self.history.insert(0, {"channel": ch.get("name") or ch.get("type"), "title": msg["title"], "ts": msg["ts"], "error": err})
        del self.history[50:]

    @staticmethod
    def _post(url: str, body: bytes, headers: dict) -> None:
        req = urllib.request.Request(url, data=body, method="POST", headers={"User-Agent": "ml-dashboard/1.0", **headers})
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()

    def _send(self, ch: dict, m: dict) -> None:
        t = ch.get("type")
        if t == "discord":
            body = {"username": "ML Dashboard", "embeds": [{"title": m["title"][:256], "description": m["message"][:4000], "url": m["url"],
                                                            "color": SEVERITY_COLOR.get(m["severity"], 0x2A78D6)}]}
            self._post(ch["url"], json.dumps(body).encode(), {"Content-Type": "application/json"})
        elif t == "slack":
            body = {"text": f"*{m['title']}*\n{m['message']}\n<{m['url']}|Open in dashboard>"}
            self._post(ch["url"], json.dumps(body).encode(), {"Content-Type": "application/json"})
        elif t == "ntfy":
            server = (ch.get("server") or "https://ntfy.sh").rstrip("/")
            prio = {"critical": "urgent", "warning": "high"}.get(m["severity"], "default")
            tags = {"critical": "rotating_light", "warning": "warning", "success": "white_check_mark"}.get(m["severity"], "robot")
            headers = {"Title": m["title"].encode("ascii", "replace").decode(), "Priority": prio, "Tags": tags, "Click": m["url"]}
            if ch.get("token"):
                headers["Authorization"] = f"Bearer {ch['token']}"
            self._post(f"{server}/{ch['topic']}", m["message"].encode(), headers)
        elif t == "email":
            em = EmailMessage()
            em["Subject"] = f"[ML Dashboard] {m['title']}"
            em["From"] = ch.get("from") or ch["username"]
            em["To"] = ch["to"]
            em.set_content(f"{m['message']}\n\n{m['url']}")
            port = int(ch.get("port") or 587)
            if port == 465:
                with smtplib.SMTP_SSL(ch["host"], port, context=ssl.create_default_context(), timeout=20) as s:
                    s.login(ch["username"], ch["password"])
                    s.send_message(em)
            else:
                with smtplib.SMTP(ch["host"], port, timeout=20) as s:
                    s.starttls(context=ssl.create_default_context())
                    s.login(ch["username"], ch["password"])
                    s.send_message(em)
        elif t == "webhook":
            self._post(ch["url"], json.dumps(m, default=str).encode(), {"Content-Type": "application/json"})
        else:
            raise ValueError(f"unknown channel type {t}")


_notifier: Notifier | None = None


def get_notifier() -> Notifier:
    global _notifier
    if _notifier is None:
        from .api.state import CFG
        _notifier = Notifier(CFG)
    return _notifier


def notify(*args, **kwargs) -> None:
    """Safe to call from anywhere in the server process; never raises."""
    try:
        get_notifier().notify(*args, **kwargs)
    except Exception:
        log.exception("notify failed")
