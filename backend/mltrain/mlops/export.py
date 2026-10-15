"""Export a registry version as a self-contained serving package (FastAPI app + Dockerfile + SDK).

The exported service answers /predict locally and pushes prediction logs (and feedback) back to this
dashboard with the client SDK, so the dashboard can monitor it without any inbound connection.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from .. import config

SERVE_PY = '''"""Standalone model server exported from the ML dashboard.

    pip install -r requirements.txt
    DASHBOARD_URL=http://<dashboard>:8000 DEPLOYMENT_ID={dep} API_KEY=<key> uvicorn serve:app --host 0.0.0.0 --port 8080
Without DASHBOARD_URL it serves without monitoring.
"""
import os
import time
import uuid

os.environ.setdefault("MLTRAIN_ROOT", os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MLTRAIN_DATA", "/tmp/mltrain-data")

from fastapi import Body, FastAPI, HTTPException  # noqa: E402

from mltrain.mlops.predictors import load_predictor  # noqa: E402
from mltrain_client import Client  # noqa: E402

predictor = load_predictor(os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model")), os.environ.get("DEVICE", "cpu"))
VERSION = {version}
client = Client(os.environ["DASHBOARD_URL"], os.environ.get("DEPLOYMENT_ID", "{dep}"), os.environ["API_KEY"]) \\
    if os.environ.get("DASHBOARD_URL") else None
app = FastAPI(title="{name} v{version}")


@app.get("/health")
def health():
    return {{"status": "ok", "model": "{name}", "version": VERSION}}


@app.post("/predict")
def predict(body: dict = Body(...)):
    instances = body.get("instances") or []
    rids = body.get("request_ids") or [uuid.uuid4().hex for _ in instances]
    t0 = time.perf_counter()
    try:
        outs = predictor.predict(instances)
    except Exception as e:
        if client:
            for x, rid in zip(instances, rids):
                client.log(input=x, prediction=None, request_id=rid, error=str(e), model_version=VERSION)
        raise HTTPException(422, str(e))
    latency = (time.perf_counter() - t0) * 1000 / max(len(instances), 1)
    if client:
        for x, o, rid in zip(instances, outs, rids):
            client.log(input=x, prediction=o["prediction"], score=o.get("positive_score", o.get("score")),
                       request_id=rid, latency_ms=latency, model_version=VERSION)
    return {{"predictions": outs, "request_ids": rids, "model_version": VERSION}}


@app.post("/feedback")
def feedback(body: dict = Body(...)):
    for it in body.get("items", []):
        if client:
            client.feedback(it["request_id"], it["label"])
    return {{"ok": True}}
'''

DOCKERFILE = '''FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLTRAIN_ROOT=/app MLTRAIN_DATA=/tmp/mltrain-data
EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
'''

README = '''# {name} v{version} — exported serving package

* `serve.py` — FastAPI app: `POST /predict {{"instances": [...]}}`, `POST /feedback {{"items": [{{"request_id", "label"}}]}}`, `GET /health`
* `model/` — the model, its fitted preprocessing (`meta.json`), training reference profile and model card
* `mltrain_client.py` — SDK used to push prediction logs + labels back to the dashboard for monitoring
* `mltrain/`, `user_plugins/` — the runtime needed to load the model

Run locally:
    pip install -r requirements.txt
    DASHBOARD_URL={url} DEPLOYMENT_ID={dep} API_KEY=<deployment key> uvicorn serve:app --port 8080

Docker:
    docker build -t {name}:v{version} .
    docker run -p 8080:8080 -e DASHBOARD_URL={url} -e DEPLOYMENT_ID={dep} -e API_KEY=<key> {name}:v{version}

Example input:
    {example}
'''

BASE_REQS = {"sklearn": ["scikit-learn", "joblib"], "xgboost": ["xgboost", "scikit-learn"], "lightgbm": ["lightgbm", "scikit-learn"],
             "torch": ["torch", "torchvision", "opencv-python-headless", "pillow"], "hf": ["torch", "transformers", "sentencepiece"],
             "sb3": ["stable-baselines3", "sb3-contrib", "gymnasium"]}


def build_export(version: dict, dep_id: str | None, dashboard_url: str) -> bytes:
    vdir = Path(version["path"])
    meta = json.loads((vdir / "meta.json").read_text())
    card = json.loads((vdir / "card.json").read_text()) if (vdir / "card.json").exists() else {}
    versions = card.get("versions") or {}
    pins = {"scikit-learn": "scikit-learn", "torch": "torch", "transformers": "transformers", "xgboost": "xgboost",
            "lightgbm": "lightgbm", "stable-baselines3": "stable-baselines3", "gymnasium": "gymnasium", "numpy": "numpy"}
    reqs = ["fastapi", "uvicorn", "numpy", "pandas", "scipy"] + BASE_REQS.get(meta["framework"], [])
    reqs = [f"{r}=={versions[r]}" if r in versions and r in pins else r for r in dict.fromkeys(reqs)]
    dep = dep_id or version["model_name"]
    fmt = {"name": version["model_name"], "version": version["version"], "dep": dep, "url": dashboard_url,
           "example": json.dumps(card.get("example_input") or {}, default=str)}
    buf = io.BytesIO()
    root = config.ROOT
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in vdir.rglob("*"):
            if p.is_file():
                z.write(p, f"model/{p.relative_to(vdir)}")
        for p in (root / "backend" / "mltrain").rglob("*.py"):
            z.write(p, f"mltrain/{p.relative_to(root / 'backend' / 'mltrain')}")
        for p in config.USER_PLUGINS_DIR.rglob("*.py"):
            z.write(p, f"user_plugins/{p.relative_to(config.USER_PLUGINS_DIR)}")
        z.write(root / "backend" / "mltrain" / "sdk" / "mltrain_client.py", "mltrain_client.py")
        z.writestr("serve.py", SERVE_PY.format(**fmt))
        z.writestr("Dockerfile", DOCKERFILE)
        z.writestr("requirements.txt", "\n".join(reqs) + "\n")
        z.writestr("README.md", README.format(**fmt))
    return buf.getvalue()
