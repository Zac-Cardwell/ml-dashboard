"""FastAPI application: REST + WebSocket API and the built frontend.

Every endpoint returns plain JSON, so the same API serves the dashboard, scripts and LLM tools.
Interactive API docs: http://<host>:8000/docs
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import config, db
from ..plugins.registry import KIND_DIR, get_registry
from . import data as data_api
from . import experiments, exports, insights, llm, mlops, modelinfo, sessions
from ..mlops.service import MLOPS
from .auth import COOKIE, MAX_AGE
from .network import NetworkGuard
from .state import AUTH, CFG, GPU_AVAILABLE, SCHED, protected_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("mltrain")
GUARD = NetworkGuard(CFG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    reg = get_registry()
    log.info("loaded %s models, %s datasets, %s envs (%s plugin errors)",
             *(len(reg.plugins[k]) for k in ("model", "dataset", "env")), len(reg.errors))
    if reg.errors:
        for e in reg.errors:
            log.warning("plugin error in %s: %s", e["file"], e["error"].strip().splitlines()[-1])
    SCHED.start()
    MLOPS.start()
    insights.BACKGROUND.start()
    log.info("LAN-only access: %s", "on" if GUARD.enabled else "OFF")
    if CFG.get("password_generated"):
        log.warning("Dashboard password: %s  (change it in Settings; stored in %s)", CFG["password"], config.CONFIG_PATH)
    yield
    await SCHED.stop()
    await MLOPS.stop()
    await insights.BACKGROUND.stop()


app = FastAPI(title="ML Training Dashboard", version="0.2.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def lan_only(request: Request, call_next):
    host = request.client.host if request.client else None
    if not GUARD.allowed(host):
        log.warning("refused request from non-LAN address %s", host)
        return PlainTextResponse("This dashboard is only available on the local network.", status_code=403)
    return await call_next(request)


public = APIRouter(prefix="/api")
api = protected_router()


# ===================================================================== auth
class LoginBody(BaseModel):
    password: str


@public.get("/auth/status")
def auth_status(request: Request):
    return {"authenticated": AUTH.is_authed(request)}


@public.post("/auth/login")
def login(body: LoginBody, response: Response):
    if not AUTH.check_password(body.password):
        raise HTTPException(401, "Wrong password")
    response.set_cookie(COOKIE, AUTH.make_cookie(), max_age=MAX_AGE, httponly=True, samesite="lax")
    return {"ok": True}


@public.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


# ===================================================================== system / settings
def _gpu_stats() -> list[dict]:
    if not GPU_AVAILABLE:
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        gpus = []
        for line in out.strip().splitlines():
            name, util, used, total, temp = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "util": float(util), "mem_used": float(used), "mem_total": float(total), "temp": float(temp)})
        return gpus
    except Exception:
        return []


@api.get("/system")
def system():
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(str(config.DATA_DIR))
    return {"gpus": _gpu_stats(), "cpu_percent": psutil.cpu_percent(), "cpu_count": psutil.cpu_count(),
            "mem_used": vm.used, "mem_total": vm.total, "disk_free": du.free, "disk_total": du.total,
            "queue": SCHED.queue_state()}


class SettingsBody(BaseModel):
    max_gpu_jobs: int | None = Field(None, ge=0, le=16)
    max_cpu_jobs: int | None = Field(None, ge=0, le=64)
    lan_only: bool | None = None
    allowed_networks: list[str] | None = None
    keep_awake: bool | None = None


@api.get("/settings")
def get_settings():
    return {"max_gpu_jobs": CFG["max_gpu_jobs"], "max_cpu_jobs": CFG["max_cpu_jobs"], "api_token": CFG["api_token"],
            "gpu_available": GPU_AVAILABLE, "lan_only": GUARD.enabled, "allowed_networks": CFG.get("allowed_networks", []),
            "keep_awake": CFG.get("keep_awake", True), "power": SCHED.power.state()}


@api.put("/settings")
def put_settings(body: SettingsBody, request: Request):
    import ipaddress
    data = body.model_dump(exclude_none=True)
    for n in data.get("allowed_networks", []):
        try:
            ipaddress.ip_network(n, strict=False)
        except ValueError:
            raise HTTPException(400, f"Not a network: {n}")
    CFG.update(data)
    if not GUARD.allowed(request.client.host if request.client else None):   # don't lock yourself out
        CFG["lan_only"] = False
    config.save_config(CFG)
    return get_settings()


@api.post("/system/display-off")
def display_off():
    """Turn the host's monitors off now; training continues (and sleep stays blocked while jobs run)."""
    from ..power import display_off as _off
    try:
        return {"message": _off()}
    except RuntimeError as e:
        raise HTTPException(500, str(e))


class PasswordBody(BaseModel):
    new_password: str = Field(min_length=4)


@api.put("/settings/password")
def change_password(body: PasswordBody, response: Response):
    CFG["password"] = body.new_password
    CFG.pop("password_generated", None)
    config.save_config(CFG)
    response.set_cookie(COOKIE, AUTH.make_cookie(), max_age=MAX_AGE, httponly=True, samesite="lax")
    return {"ok": True}


# ===================================================================== plugins
@api.get("/plugins")
def plugins():
    reg = get_registry()
    return {"models": [p.meta() for p in reg.list("model")],
            "datasets": [p.meta() for p in reg.list("dataset")],
            "envs": [p.meta() for p in reg.list("env")],
            "errors": reg.errors}


@api.post("/plugins/reload")
def reload_plugins():
    reg = get_registry(reload=True)
    return {"errors": reg.errors}


@api.get("/plugins/files")
def plugin_files():
    out = []
    for kind, sub in KIND_DIR.items():
        for p in sorted((config.USER_PLUGINS_DIR / sub).glob("*.py")):
            out.append({"kind": kind, "name": p.name, "size": p.stat().st_size, "modified": p.stat().st_mtime})
    uploads = []
    for m in sorted((config.UPLOADS_DIR / "datasets").glob("*.json")):
        uploads.append({"slug": m.stem, **json.loads(m.read_text())})
    return {"plugin_files": out, "csv_datasets": uploads}


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*\.py$")


@api.post("/plugins/upload")
async def upload_plugin(kind: str = Form(...), file: UploadFile = File(...)):
    if kind not in KIND_DIR:
        raise HTTPException(400, f"kind must be one of {list(KIND_DIR)}")
    name = file.filename or ""
    if not _SAFE_NAME.match(name):
        raise HTTPException(400, "File must be a .py with a simple name (letters, digits, _ -)")
    data = await file.read()
    if len(data) > 2_000_000:
        raise HTTPException(400, "Plugin file too large")
    dest = config.USER_PLUGINS_DIR / KIND_DIR[kind] / name.replace("-", "_")
    dest.write_bytes(data)
    reg = get_registry(reload=True)
    errs = [e for e in reg.errors if dest.name in e["file"]]
    added = [p.meta() for p in reg.list(kind) if p.source.endswith(dest.name)]
    return {"saved": str(dest.relative_to(config.ROOT)), "added": added, "errors": errs}


def _plugin_path(kind: str, name: str) -> Path:
    if kind not in KIND_DIR or not _SAFE_NAME.match(name):
        raise HTTPException(400, "bad plugin path")
    path = config.USER_PLUGINS_DIR / KIND_DIR[kind] / name
    if not path.exists():
        raise HTTPException(404, "not found")
    return path


@api.get("/plugins/file/{kind}/{name}", response_class=PlainTextResponse)
def read_plugin_file(kind: str, name: str):
    return _plugin_path(kind, name).read_text()


@api.delete("/plugins/file/{kind}/{name}")
def delete_plugin_file(kind: str, name: str):
    _plugin_path(kind, name).unlink()
    get_registry(reload=True)
    return {"ok": True}


@api.post("/datasets/upload_csv")
async def upload_csv(file: UploadFile = File(...), name: str = Form(...), target: str = Form(...),
                     task: str = Form(...), description: str = Form("")):
    import pandas as pd
    if task not in ("classification", "regression"):
        raise HTTPException(400, "task must be classification or regression")
    slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "dataset"
    dest = config.UPLOADS_DIR / "datasets" / f"{slug}.csv"
    dest.write_bytes(await file.read())
    try:
        cols = list(pd.read_csv(dest, nrows=5).columns)
    except Exception as e:
        dest.unlink()
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if target not in cols:
        dest.unlink()
        raise HTTPException(400, f"Target '{target}' not in columns {cols}")
    dest.with_suffix(".json").write_text(json.dumps({"name": name, "target": target, "task": task,
                                                     "description": description, "columns": cols}))
    get_registry(reload=True)
    return {"id": f"upload.{slug}"}


@api.delete("/datasets/upload/{slug}")
def delete_csv(slug: str):
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise HTTPException(400, "bad slug")
    for ext in (".csv", ".json"):
        (config.UPLOADS_DIR / "datasets" / f"{slug}{ext}").unlink(missing_ok=True)
    get_registry(reload=True)
    return {"ok": True}


app.include_router(public)
app.include_router(api)
app.include_router(sessions.router)
app.include_router(experiments.router)
app.include_router(llm.router)
app.include_router(mlops.router)
app.include_router(insights.router)
app.include_router(data_api.router)
app.include_router(exports.router)
app.include_router(modelinfo.router)
app.include_router(mlops.serve)
sessions.register_ws(app, GUARD)


# ===================================================================== frontend
@app.exception_handler(404)
async def spa_fallback(request: Request, exc):
    if request.url.path.startswith("/api") or not (config.FRONTEND_DIST / "index.html").exists():
        detail = getattr(exc, "detail", "Not found")
        return JSONResponse({"detail": detail}, status_code=404)
    return FileResponse(config.FRONTEND_DIST / "index.html")


if (config.FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(config.FRONTEND_DIST / "favicon.svg")


@app.get("/", include_in_schema=False)
def index():
    idx = config.FRONTEND_DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return JSONResponse({"detail": "Frontend not built. Run ./run.sh or `npm run build` in frontend/."})
