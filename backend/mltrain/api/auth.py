"""Shared-password login (signed cookie) plus a bearer API token for scripts / LLM tools."""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, WebSocket
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

COOKIE = "mltrain_session"
MAX_AGE = 60 * 60 * 24 * 30  # 30 days


class Auth:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    def signer(self) -> TimestampSigner:
        # password is part of the key so changing it logs everyone out
        return TimestampSigner(self.cfg["secret_key"] + self.cfg["password"])

    def check_password(self, pw: str) -> bool:
        return hmac.compare_digest(pw.encode(), self.cfg["password"].encode())

    def make_cookie(self) -> str:
        return self.signer.sign(b"ok").decode()

    def _valid_cookie(self, value: str | None) -> bool:
        if not value:
            return False
        try:
            self.signer.unsign(value, max_age=MAX_AGE)
            return True
        except (BadSignature, SignatureExpired):
            return False

    def _valid_token(self, header: str | None) -> bool:
        if not header or not header.lower().startswith("bearer "):
            return False
        return hmac.compare_digest(header[7:].strip().encode(), self.cfg["api_token"].encode())

    def is_authed(self, req: Request | WebSocket) -> bool:
        return self._valid_cookie(req.cookies.get(COOKIE)) or self._valid_token(req.headers.get("authorization"))

    def require(self, req: Request) -> None:
        if not self.is_authed(req):
            raise HTTPException(401, "Not authenticated")
