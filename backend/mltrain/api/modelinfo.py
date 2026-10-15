"""Model reference & tooling: architecture summaries, loss catalog, linked project folders, generation playground."""
from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import architecture, config, db, plugins_admin, storage
from ..plugins.registry import get_registry, linked_folders, save_linked_folders
from .state import protected_router

router = protected_router()
_GEN_LOCK = threading.Lock()


def _model(mid: str):
    try:
        return get_registry().get("model", mid)
    except KeyError:
        raise HTTPException(404, f"model {mid} not found")


@router.get("/models/{mid}/details")
def model_details(mid: str):
    return architecture.details(_model(mid), get_registry())


class ArchBody(BaseModel):
    params: dict[str, Any] = {}
    data_shape: list[int] | None = None
    modality: str | None = None
    max_depth: int = 4
    n_classes: int = 10


@router.post("/models/{mid}/architecture")
async def model_architecture(mid: str, body: ArchBody):
    plugin = _model(mid)
    try:
        return await run_in_threadpool(architecture.summarize, plugin, body.params, body.data_shape, body.modality,
                                       max(1, min(body.max_depth, 12)), body.n_classes)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/losses")
def list_losses():
    from ..generative import losses
    return {"losses": losses.catalog(), "errors": losses.errors()}


# ===================================================================== linked project folders
class ScanBody(BaseModel):
    path: str


def _scan_file(p: Path) -> dict:
    """What a .py file contains, without executing it (AST): nn.Module-like classes, loss-like functions."""
    try:
        tree = ast.parse(p.read_text(errors="ignore"))
    except SyntaxError as e:
        return {"file": p.name, "error": f"syntax error line {e.lineno}"}
    classes, funcs = [], []
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            bases = [ast.unparse(b) for b in n.bases]
            methods = [m.name for m in n.body if isinstance(m, ast.FunctionDef)]
            classes.append({"name": n.name, "bases": bases, "methods": methods,
                            "torch": any(b.split(".")[-1] in ("Module", "nn.Module") for b in bases)})
        elif isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            funcs.append({"name": n.name, "args": [a.arg for a in n.args.args]})
    main = any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)
    torch_classes = [c for c in classes if c["torch"]]
    loss_like = [f for f in funcs if len(f["args"]) >= 2 and any(k in f["name"].lower() for k in ("loss", "criterion"))]
    return {"file": p.name, "classes": classes, "functions": funcs, "has_main": main,
            "suggest": "model" if torch_classes else "losses" if loss_like else None,
            "lines": len(p.read_text(errors="ignore").splitlines())}


@router.get("/plugins/linked")
def get_linked():
    reg = get_registry()
    out = []
    for link in linked_folders():
        files = []
        for f in link.get("files", []):
            full = str(Path(link["path"]) / f["file"])
            files.append({**f, "plugins": reg.files.get(full, []),
                          "notes": [n["note"] for n in reg.notes if n["file"] == full],
                          "errors": [e["error"] for e in reg.errors if e["file"] == full]})
        out.append({**link, "exists": Path(link["path"]).is_dir(), "files": files})
    return out


@router.post("/plugins/linked/scan")
def scan_folder(body: ScanBody):
    folder = Path(body.path).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"Not a folder: {folder}")
    files = sorted(p for p in folder.glob("*.py") if not p.name.startswith("."))
    return {"path": str(folder.resolve()), "files": [_scan_file(p) for p in files[:200]]}


class LinkedBody(BaseModel):
    links: list[dict[str, Any]]


@router.put("/plugins/linked")
async def put_linked(body: LinkedBody):
    clean = []
    for l in body.links:
        folder = Path(str(l.get("path", ""))).expanduser()
        if not folder.is_dir():
            raise HTTPException(400, f"Not a folder: {folder}")
        files = [{"file": f["file"], "kind": f.get("kind", "model")} for f in l.get("files", [])
                 if (folder / f["file"]).is_file() and f.get("kind", "model") in ("model", "dataset", "env", "losses")]
        clean.append({"path": str(folder.resolve()), "name": l.get("name") or folder.name, "files": files})
    save_linked_folders(clean)
    reg = await run_in_threadpool(get_registry, True)
    return {"links": get_linked(), "errors": reg.errors}


# ===================================================================== generation playground
class GenBody(BaseModel):
    n: int = 16
    seed: int = 0
    variant: str = "best"
    sampler: str | None = None        # diffusion: ddpm | ddim
    steps: int | None = None          # diffusion DDIM steps
    prompt: str = ""                  # text
    length: int = 200                 # text tokens
    temperature: float | None = None
    top_k: int | None = None


def _generate(run: dict, sess: dict, body: GenBody) -> dict:
    import torch

    from ..generative import media
    from ..generative.lm import LanguageModel, tokenizer_from_spec
    from ..plugins.base import coerce_params
    plugin = get_registry().get("model", sess["model_id"])
    mdir = storage.model_dir(run["id"], body.variant)
    gen_path = mdir / "generation.json"
    if not (mdir / "model.pt").exists() or not gen_path.exists():
        raise ValueError("This run has no saved generative model (saving was off, or it was deleted)")
    info = json.loads(gen_path.read_text())
    p = coerce_params(plugin.params(), sess["model_params"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = storage.run_dir(run["id"]) / "generated"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with _GEN_LOCK:
        if isinstance(plugin, LanguageModel):
            plugin._tok = tokenizer_from_spec(info["tokenizer"])
            info["vocab_size"] = plugin._tok.size
        net = plugin.build(p, info).to(device)
        plugin.load_weights(net, mdir / "model.pt", device)
        net.eval()
        g = torch.Generator(device=device).manual_seed(int(body.seed))
        t0 = time.time()
        if isinstance(plugin, LanguageModel):
            texts = [net.generate(body.prompt, max(1, min(body.length, 2000)), body.temperature, body.top_k, generator=g)
                     for _ in range(max(1, min(body.n, 8)))]
            return {"kind": "text", "prompt": body.prompt, "texts": texts, "seconds": round(time.time() - t0, 2)}
        n = max(1, min(body.n, 64))
        kw = {}
        if body.sampler:
            kw["sampler"] = body.sampler
        if body.steps:
            kw["steps"] = body.steps
        import inspect
        accepted = inspect.signature(net.sample).parameters
        x = net.sample(n, generator=g, **{k: v for k, v in kw.items() if k in accepted})
        mod = info["modality"]
        rel = lambda path: str(Path(path).relative_to(storage.run_dir(run["id"])))
        res = {"kind": mod, "seconds": round(time.time() - t0, 2)}
        if mod == "image":
            res["image"] = rel(media.save_image_grid(x, out_dir / f"{stamp}_s{body.seed}.png"))
        elif mod == "video":
            res["video"] = rel(media.save_video_grid(x[:16], out_dir / f"{stamp}_s{body.seed}.mp4", info.get("fps", 8)))
        elif mod == "audio":
            a = media.save_audio_set(x[:8], info.get("sample_rate", 16000), out_dir / f"{stamp}_s{body.seed}", "clip")
            res.update(clips=[rel(c) for c in a["clips"]], specs=[rel(s) for s in a["specs"]], spectrogram=rel(a["spectrogram"]))
        return res


@router.post("/runs/{rid}/generate")
async def generate(rid: str, body: GenBody):
    with db.get_conn() as c:
        run = db.one(c, "SELECT * FROM runs WHERE id=?", (rid,))
        sess = db.one(c, "SELECT * FROM sessions WHERE id=?", (run["session_id"],)) if run else None
    if not run:
        raise HTTPException(404, "run not found")
    try:
        return await run_in_threadpool(_generate, run, sess, body)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


# ===================================================================== removing datasets / environments
def _kind(kind: str) -> str:
    if kind not in ("dataset", "env"):
        raise HTTPException(400, "kind must be dataset or env")
    return kind


@router.get("/plugins/{kind}/{pid}/removal")
def plugin_removal_plan(kind: str, pid: str):
    try:
        return plugins_admin.plan(pid, _kind(kind))
    except KeyError:
        raise HTTPException(404, f"{kind} not found")


@router.delete("/plugins/{kind}/{pid}")
async def delete_plugin(kind: str, pid: str, delete_data: bool = True):
    try:
        return await run_in_threadpool(plugins_admin.remove, pid, delete_data, _kind(kind))
    except KeyError:
        raise HTTPException(404, f"{kind} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/plugins-removed/{kind}")
def removed_plugins(kind: str):
    return plugins_admin.removed(_kind(kind))


class RestoreBody(BaseModel):
    key: str


@router.post("/plugins-removed/{kind}/restore")
async def restore_plugin(kind: str, body: RestoreBody):
    try:
        return await run_in_threadpool(plugins_admin.restore, body.key, _kind(kind))
    except KeyError:
        raise HTTPException(404, "not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
