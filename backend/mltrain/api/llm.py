"""LLM-facing endpoints: a compact digest of all sessions and tool definitions that map onto the REST API.

`GET /api/llm/tools` returns tool schemas (name / description / input_schema — the format used by the
Claude Messages API tool use, easily adapted to others). Each tool lists the HTTP call that implements it,
so a small agent loop can execute tool calls generically. `mltrain/mcp_server.py` exposes the same tools
over MCP for Claude Desktop / Claude Code.
"""
from __future__ import annotations

from .. import analysis, db
from .sessions import runs_by_session
from .state import protected_router

router = protected_router()


@router.get("/llm/overview")
def llm_overview(limit: int = 100):
    """Compact digest of recent sessions: config, headline metric and final metric means."""
    with db.get_conn() as c:
        sessions = db.rows(c, "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,))
        rb = runs_by_session(c, [s["id"] for s in sessions])
    out = []
    for s in sessions:
        runs = rb.get(s["id"], [])
        stats = analysis.summary_stats([r for r in runs if r["status"] == "completed"])
        out.append({**analysis.session_brief(s, runs), "problem_params": s["problem_params"],
                    "model_params": s["model_params"], "notes": s["notes"],
                    "final_metrics": {k: {"mean": v["mean"], "std": v["std"], "n": v["n"]} for k, v in stats.items()}})
    return {"schema": "mltrain.overview/v2", "sessions": out}


_SESSION_BODY = {
    "type": "object",
    "properties": {
        "problem_kind": {"type": "string", "enum": ["dataset", "env"]},
        "problem_id": {"type": "string", "description": "id from list_plugins (datasets or envs)"},
        "problem_params": {"type": "object"},
        "model_id": {"type": "string", "description": "id from list_plugins (models)"},
        "model_params": {"type": "object", "description": "only the params to change; others use defaults"},
        "seeds": {"type": "array", "items": {"type": "integer"}},
        "mode": {"type": "string", "enum": ["train", "evaluate"]},
        "options": {"type": "object", "description": "save_model, snapshots, record_video, ... (see get_options)"},
        "init_from": {"type": "object", "properties": {"session_id": {"type": "string"},
                                                       "strategy": {"type": "string", "enum": ["match_seed", "best", "run"]},
                                                       "run_id": {"type": "string"}},
                      "description": "start from (train) or evaluate (evaluate) a saved model of another session"},
        "parent_id": {"type": "string", "description": "link as a retrain of this session"},
        "name": {"type": "string"}, "notes": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["problem_kind", "problem_id", "model_id"],
}

TOOLS = [
    {"name": "get_overview", "http": "GET /api/llm/overview",
     "description": "Digest of recent training sessions with configs and final metrics (mean/std across seeds). Start here.",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "get_session", "http": "GET /api/sessions/{session_id}/export",
     "description": "Everything about one session: config, per-seed and seed-averaged curves, final stats, artifacts, snapshots, lineage.",
     "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    {"name": "list_plugins", "http": "GET /api/plugins",
     "description": "Available models, datasets and environments with their parameter schemas.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_saved_models", "http": "GET /api/models/saved",
     "description": "Trained models kept on disk that can be fine-tuned or evaluated.",
     "input_schema": {"type": "object", "properties": {"task": {"type": "string"}, "modality": {"type": "string"}}}},
    {"name": "start_session", "http": "POST /api/sessions",
     "description": "Start a training (or evaluation) session. Use parent_id to record it as a retrain of an earlier session.",
     "input_schema": _SESSION_BODY},
    {"name": "start_sweep", "http": "POST /api/sweeps",
     "description": "Grid search: one session per combination, e.g. grid={'model.lr':[1e-3,3e-3],'model.batch_size':[32,64]}.",
     "input_schema": {"type": "object", "properties": {"base": _SESSION_BODY, "grid": {"type": "object"}}, "required": ["base", "grid"]}},
    {"name": "stop_session", "http": "POST /api/sessions/{session_id}/stop",
     "description": "Stop a running session gracefully.",
     "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    {"name": "get_lineage", "http": "GET /api/sessions/{session_id}/lineage",
     "description": "The retrain tree containing a session, with the parameter changes at each step.",
     "input_schema": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    {"name": "list_deployments", "http": "GET /api/deployments",
     "description": "Deployed models with health (healthy/warning/critical), live headline metric vs. baseline, drift, traffic, alerts.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_deployment_analysis", "http": "GET /api/deployments/{deployment_id}/analysis",
     "description": "Monitoring analysis of a deployment over the last N minutes: per-feature drift (PSI/KS), data quality, prediction "
                    "distribution vs. training, live performance vs. baseline, health reasons.",
     "input_schema": {"type": "object", "properties": {"deployment_id": {"type": "string"}, "minutes": {"type": "number"}},
                      "required": ["deployment_id"]}},
    {"name": "list_alerts", "http": "GET /api/alerts", "description": "Open monitoring alerts across all deployments.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "trigger_retrain", "http": "POST /api/deployments/{deployment_id}/automation/run",
     "description": "Retrain a challenger on fresh data and compare it with the production model (kind='retrain'), or re-test the "
                    "production model on fresh data (kind='evaluate').",
     "input_schema": {"type": "object", "properties": {"deployment_id": {"type": "string"}, "kind": {"type": "string", "enum": ["retrain", "evaluate"]}},
                      "required": ["deployment_id"]}},
    {"name": "get_group", "http": "GET /api/groups/{group_id}",
     "description": "Results of a sweep or benchmark group, one row per member with its swept values and final metrics.",
     "input_schema": {"type": "object", "properties": {"group_id": {"type": "string"}}, "required": ["group_id"]}},
]


@router.get("/llm/tools")
def llm_tools():
    return {"schema": "mltrain.tools/v1", "tools": TOOLS,
            "auth": "Send 'Authorization: Bearer <api_token>' (Settings page). Path placeholders like {session_id} "
                    "come from the tool input; remaining input goes in the query string (GET) or JSON body (POST)."}
