"""Process-wide state shared by the API modules."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, Request

from .. import config
from ..scheduler import Scheduler
from .auth import Auth

CFG = config.load_config()
AUTH = Auth(CFG)
SCHED = Scheduler(CFG)
GPU_AVAILABLE = shutil.which("nvidia-smi") is not None


def require_auth(request: Request) -> None:
    AUTH.require(request)


def protected_router() -> APIRouter:
    return APIRouter(prefix="/api", dependencies=[Depends(require_auth)])
