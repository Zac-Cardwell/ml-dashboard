"""MCP server exposing the dashboard to LLM clients (Claude Desktop, Claude Code, ...).

It talks to a running dashboard over HTTP using the API token, so it works from any machine on the LAN.

    MLTRAIN_URL=http://192.168.1.50:8000 MLTRAIN_TOKEN=<token from Settings> python -m mltrain.mcp_server

Claude Code:  claude mcp add mltrain -e MLTRAIN_URL=... -e MLTRAIN_TOKEN=... -- \
              /path/to/conda/envs/mltrain/bin/python -m mltrain.mcp_server
(run from the backend/ directory or set PYTHONPATH=backend)
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
try:                                           # mcp >= 2
    from mcp.server.mcpserver import MCPServer
except ImportError:                            # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer


def _token() -> str:
    tok = os.environ.get("MLTRAIN_TOKEN")
    if tok:
        return tok
    from . import config   # same machine: read it from data/config.json
    return json.loads(config.CONFIG_PATH.read_text())["api_token"]


BASE = os.environ.get("MLTRAIN_URL", "http://127.0.0.1:8000").rstrip("/")
client = httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {_token()}"}, timeout=60)
mcp = MCPServer("ml-training-dashboard")


def _call(method: str, path: str, **kw) -> Any:
    r = client.request(method, path, **kw)
    if r.status_code >= 400:
        return {"error": r.status_code, "detail": r.text[:2000]}
    return r.json()


@mcp.tool()
def get_overview(limit: int = 50) -> Any:
    """Digest of recent training sessions with their configs and final metrics (mean/std across seeds). Start here."""
    return _call("GET", "/api/llm/overview", params={"limit": limit})


@mcp.tool()
def get_session(session_id: str, max_points: int = 100) -> Any:
    """Everything about one session: config, seed-averaged and per-seed curves, final stats, artifacts, snapshots, lineage."""
    return _call("GET", f"/api/sessions/{session_id}/export", params={"max_points": max_points})


@mcp.tool()
def list_plugins() -> Any:
    """Available models, datasets and RL environments, with each one's parameter schema."""
    return _call("GET", "/api/plugins")


@mcp.tool()
def list_saved_models(task: str | None = None, modality: str | None = None) -> Any:
    """Trained models kept on disk (can be fine-tuned via init_from, or evaluated with mode='evaluate')."""
    return _call("GET", "/api/models/saved", params={k: v for k, v in {"task": task, "modality": modality}.items() if v})


@mcp.tool()
def start_session(problem_kind: str, problem_id: str, model_id: str, model_params: dict | None = None,
                  problem_params: dict | None = None, seeds: list[int] | None = None, mode: str = "train",
                  options: dict | None = None, init_from: dict | None = None, parent_id: str | None = None,
                  name: str | None = None, notes: str = "") -> Any:
    """Start a training or evaluation session. problem_kind is 'dataset' or 'env'. Only pass params you want to change.
    Set parent_id to record it as a retrain of an earlier session; init_from={'session_id':..., 'strategy':'best'}
    starts from (train) or evaluates (mode='evaluate') a saved model."""
    body = {"problem_kind": problem_kind, "problem_id": problem_id, "model_id": model_id,
            "model_params": model_params or {}, "problem_params": problem_params or {}, "seeds": seeds or [0],
            "mode": mode, "options": options or {}, "init_from": init_from, "parent_id": parent_id, "name": name, "notes": notes}
    return _call("POST", "/api/sessions", json=body)


@mcp.tool()
def start_sweep(base: dict, grid: dict) -> Any:
    """Grid search. base is a start_session body; grid maps 'model.<param>' / 'problem.<param>' to lists of values."""
    return _call("POST", "/api/sweeps", json={"base": base, "grid": grid})


@mcp.tool()
def get_group(group_id: str) -> Any:
    """Results of a sweep or benchmark group: one row per member with its swept values and final metrics."""
    return _call("GET", f"/api/groups/{group_id}")


@mcp.tool()
def get_lineage(session_id: str) -> Any:
    """The retrain tree containing a session, with the parameter changes made at each step."""
    return _call("GET", f"/api/sessions/{session_id}/lineage")


@mcp.tool()
def list_deployments() -> Any:
    """Deployed models: health, live headline metric vs. training baseline, drift, traffic and open alerts."""
    return _call("GET", "/api/deployments")


@mcp.tool()
def get_deployment_analysis(deployment_id: str, minutes: float = 60) -> Any:
    """Monitoring analysis of a deployment over the last N minutes: feature drift (PSI/KS), data quality,
    prediction distribution vs. training, live performance vs. baseline, and the reasons for its health status."""
    a = _call("GET", f"/api/deployments/{deployment_id}/analysis", params={"minutes": minutes})
    if isinstance(a, dict):
        a.pop("reference", None)   # bulky reference histograms
    return a


@mcp.tool()
def list_alerts() -> Any:
    """Open monitoring alerts across all deployments."""
    return _call("GET", "/api/alerts")


@mcp.tool()
def trigger_retrain(deployment_id: str, kind: str = "retrain") -> Any:
    """kind='retrain': train a challenger on fresh data and compare with production (champion/challenger).
    kind='evaluate': re-test the production model on fresh data."""
    return _call("POST", f"/api/deployments/{deployment_id}/automation/run", json={"kind": kind})


@mcp.tool()
def stop_session(session_id: str) -> Any:
    """Stop a running session gracefully (it still evaluates and saves what it has)."""
    return _call("POST", f"/api/sessions/{session_id}/stop")


if __name__ == "__main__":
    mcp.run()
