"""SQLite storage shared by the API server and worker processes (WAL mode)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    problem_kind TEXT NOT NULL,          -- 'dataset' | 'env'
    problem_id TEXT NOT NULL,
    problem_params TEXT NOT NULL,        -- JSON
    model_id TEXT NOT NULL,
    model_params TEXT NOT NULL,          -- JSON
    task TEXT,
    modality TEXT,
    framework TEXT,
    seeds TEXT NOT NULL,                 -- JSON list[int]
    device TEXT NOT NULL,                -- 'cuda' | 'cpu'
    parent_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    notes TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    env_info TEXT DEFAULT '{}'           -- library versions, plugin hashes
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,                -- queued|running|stopping|completed|failed|stopped
    created_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    pid INTEGER,
    error TEXT,
    summary TEXT DEFAULT '{}',           -- JSON final metrics
    progress REAL DEFAULT 0,
    progress_text TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    t REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, id);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,                  -- confusion_matrix | table | text | json
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    name TEXT NOT NULL,                  -- e.g. 'episode', 'decision_boundary', 'samples'
    kind TEXT NOT NULL,                  -- video | image | text_samples | decision_boundary | curve_fit | image_predictions | confusion_matrix
    path TEXT,                           -- file relative to the run dir (videos, images)
    data TEXT,                           -- JSON payload for structured snapshots
    t REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_run ON media(run_id, name, step);

-- ------------------------------------------------------------------ MLOps
CREATE TABLE IF NOT EXISTS registered_models (
    name TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    task TEXT, modality TEXT, model_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL REFERENCES registered_models(name) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'none',          -- none | staging | production | archived
    source_session_id TEXT, source_run_id TEXT,
    model_id TEXT, framework TEXT, task TEXT, modality TEXT,
    path TEXT NOT NULL,                          -- immutable copy under data/registry/
    metrics TEXT DEFAULT '{}',
    description TEXT DEFAULT '',
    created_by TEXT DEFAULT 'user',              -- user | automation
    created_at REAL NOT NULL,
    UNIQUE(model_name, version)
);
CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY,                         -- url slug
    name TEXT NOT NULL,
    model_name TEXT NOT NULL REFERENCES registered_models(name),
    champion_version_id TEXT, challenger_version_id TEXT,
    challenger_mode TEXT NOT NULL DEFAULT 'none',  -- none | shadow | canary
    canary_fraction REAL NOT NULL DEFAULT 0.1,
    kind TEXT NOT NULL DEFAULT 'internal',       -- internal (served here) | external (pushes logs)
    status TEXT NOT NULL DEFAULT 'active',       -- active | paused
    api_key TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',           -- monitoring thresholds / windows / serving device
    automation TEXT NOT NULL DEFAULT '{}',       -- retrain + re-evaluation policy
    health TEXT DEFAULT 'no_data', health_reasons TEXT DEFAULT '[]',
    last_monitor_at REAL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    version_id TEXT, request_id TEXT, ts REAL NOT NULL,
    features TEXT,                               -- monitored features (raw tabular record / derived stats)
    payload TEXT,                                -- original input when small (text, tabular, RL obs)
    prediction TEXT, score REAL, latency_ms REAL, error TEXT,
    label TEXT, label_ts REAL,
    shadow INTEGER NOT NULL DEFAULT 0, source TEXT DEFAULT 'served'
);
CREATE INDEX IF NOT EXISTS idx_pred_dep_ts ON predictions(deployment_id, ts);
CREATE INDEX IF NOT EXISTS idx_pred_req ON predictions(deployment_id, request_id);
CREATE TABLE IF NOT EXISTS monitor_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    ts REAL NOT NULL, window_start REAL, window_end REAL, n INTEGER,
    health TEXT, metrics TEXT
);
CREATE INDEX IF NOT EXISTS idx_snap_dep_ts ON monitor_snapshots(deployment_id, ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    key TEXT NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL, message TEXT,
    status TEXT NOT NULL DEFAULT 'open',         -- open | acknowledged | resolved
    created_at REAL NOT NULL, updated_at REAL NOT NULL, resolved_at REAL, value REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_dep ON alerts(deployment_id, status);
CREATE TABLE IF NOT EXISTS deployment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    ts REAL NOT NULL, kind TEXT NOT NULL, message TEXT, data TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS searches (
    id TEXT PRIMARY KEY,                         -- also the group_id of its trial sessions
    name TEXT, status TEXT NOT NULL,             -- running | completed | stopped
    config TEXT NOT NULL,                        -- base session, search space, n_trials, parallel, metric, pruning
    metric TEXT, direction TEXT,
    best_session_id TEXT, best_value REAL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, notified INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS search_trials (
    search_id TEXT NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
    trial_number INTEGER NOT NULL,
    session_id TEXT,
    state TEXT NOT NULL,                         -- running | complete | pruned | failed
    value REAL, params TEXT DEFAULT '{}',
    PRIMARY KEY (search_id, trial_number)
);
CREATE TABLE IF NOT EXISTS data_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config TEXT NOT NULL,                        -- connector, table/query, timestamp / target / id columns, thresholds
    health TEXT DEFAULT 'unknown', health_reasons TEXT DEFAULT '[]',
    last_profiled_at REAL, last_error TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS source_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    ts REAL NOT NULL, health TEXT, metrics TEXT
);
CREATE INDEX IF NOT EXISTS idx_srcsnap ON source_snapshots(source_id, ts);
CREATE TABLE IF NOT EXISTS training_schedules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    template TEXT NOT NULL,                      -- SessionCreate body used for every run
    cadence TEXT NOT NULL,                       -- {kind: interval|daily|weekly, every_hours, at, weekday}
    register_as TEXT,                            -- registry model name to register each result under (optional)
    next_run_at REAL, last_run_at REAL, last_session_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS schedule_runs (
    schedule_id TEXT NOT NULL REFERENCES training_schedules(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL, created_at REAL NOT NULL,
    registered_version TEXT, status TEXT DEFAULT 'running',
    PRIMARY KEY (schedule_id, session_id)
);
CREATE TABLE IF NOT EXISTS automation_runs (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                          -- retrain | evaluate
    trigger TEXT NOT NULL,                       -- schedule | drift | performance | manual
    status TEXT NOT NULL,                        -- running | awaiting_approval | promoted | rejected | completed | failed
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    challenger_session_id TEXT, champion_session_id TEXT, challenger_version_id TEXT,
    decision TEXT, message TEXT, data TEXT DEFAULT '{}'
);
"""

# Columns added after the first release; applied with ALTER TABLE when missing.
MIGRATIONS = {
    "sessions": {
        "mode": "TEXT NOT NULL DEFAULT 'train'",       # train | evaluate
        "options": "TEXT NOT NULL DEFAULT '{}'",       # save_model, snapshots, video, ...
        "init_from": "TEXT",                           # JSON {session_id, strategy} for fine-tune / evaluate
        "group_id": "TEXT",                            # sweep / benchmark group
        "group_kind": "TEXT",                          # sweep | benchmark
        "group_params": "TEXT DEFAULT '{}'",           # the swept values of this member
        "priority": "INTEGER NOT NULL DEFAULT 0",      # -1 low, 0 normal, 1 high, 2 urgent
        "scheduled_at": "REAL",                        # don't start before this unix time
        "notified": "INTEGER NOT NULL DEFAULT 0",      # completion notification sent
    },
    "runs": {
        "init_path": "TEXT",                           # registry version dir to start from / evaluate
        "resume": "INTEGER NOT NULL DEFAULT 0",
        "init_run_id": "TEXT",                         # saved model this run starts from
    },
}

JSON_FIELDS = {"problem_params", "model_params", "seeds", "tags", "env_info", "summary", "data", "options", "init_from",
               "group_params", "metrics", "config", "params", "template", "cadence", "automation", "health_reasons", "features", "payload", "prediction", "label"}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    config.ensure_dirs()
    with get_conn() as c:
        c.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_group ON sessions(group_id)")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for k in JSON_FIELDS & d.keys():
        if isinstance(d[k], str):
            d[k] = json.loads(d[k])
    return d


def rows(c: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    return [row_to_dict(r) for r in c.execute(sql, args).fetchall()]


def one(c: sqlite3.Connection, sql: str, args: tuple = ()) -> dict[str, Any] | None:
    return row_to_dict(c.execute(sql, args).fetchone())


def update_run(c: sqlite3.Connection, run_id: str, **fields: Any) -> None:
    for k, v in list(fields.items()):
        if k in JSON_FIELDS and not isinstance(v, str):
            fields[k] = json.dumps(v)
    sets = ", ".join(f"{k}=?" for k in fields)
    c.execute(f"UPDATE runs SET {sets} WHERE id=?", (*fields.values(), run_id))


def now() -> float:
    return time.time()
