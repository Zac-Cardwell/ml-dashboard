"""Portfolio exports: build a standalone GitHub-ready repo from sessions, download it, and (on explicit request) push it."""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import portfolio
from .state import protected_router

router = protected_router()


class ExportBody(BaseModel):
    session_ids: list[str]
    title: str
    description: str = ""
    author: str = ""
    email: str = ""
    license: str = "MIT"              # MIT | Apache-2.0 | none
    include_models: str = "none"      # none | best | both
    include_videos: bool = True
    git_init: bool = True
    overrides: dict[str, dict] = {}   # session_id -> {title, description}


class PushBody(BaseModel):
    remote: str = ""
    create_with_gh: bool = False
    private: bool = False
    confirm: bool = False             # the UI sends this only after the user confirms publishing


def _dir(name: str):
    try:
        return portfolio.export_dir(name)
    except FileNotFoundError:
        raise HTTPException(404, "export not found")


@router.get("/exports")
def list_exports():
    return {"exports": portfolio.list_exports(), "identity": portfolio.git_identity()}


@router.post("/exports")
async def create_export(body: ExportBody):
    try:
        return await run_in_threadpool(portfolio.build_export, body.session_ids, body.title, body.description, body.author,
                                       body.email, body.license, body.include_models, body.include_videos, body.git_init,
                                       body.overrides)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/exports/{name}/readme")
def export_readme(name: str):
    return PlainTextResponse((_dir(name) / "README.md").read_text())


@router.get("/exports/{name}/file/{path:path}")
def export_file(name: str, path: str):
    d = _dir(name).resolve()
    p = (d / path).resolve()
    if not p.is_relative_to(d) or not p.is_file() or ".git" in p.relative_to(d).parts:
        raise HTTPException(404, "file not found")
    return FileResponse(p)


@router.get("/exports/{name}/tree")
def export_tree(name: str):
    d = _dir(name)
    return [{"path": str(p.relative_to(d)), "bytes": p.stat().st_size} for p in sorted(d.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(d).parts]


@router.get("/exports/{name}/zip")
async def export_zip(name: str):
    _dir(name)
    z = await run_in_threadpool(portfolio.zip_export, name)
    return FileResponse(z, filename=z.name, media_type="application/zip")


@router.delete("/exports/{name}")
def delete_export(name: str):
    _dir(name)
    portfolio.delete_export(name)
    return {"ok": True}


@router.post("/exports/{name}/push")
async def push_export(name: str, body: PushBody):
    _dir(name)
    if not body.confirm:
        raise HTTPException(400, "Publishing must be confirmed")
    try:
        return await run_in_threadpool(portfolio.push_export, name, body.remote, body.create_with_gh, body.private)
    except ValueError as e:
        raise HTTPException(400, str(e))
