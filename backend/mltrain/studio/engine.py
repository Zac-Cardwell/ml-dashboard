"""DuckDB engine for the Data Studio.

Every ready Studio dataset is a Parquet file (data/studio/tables/<id>/v<N>.parquet) exposed as a view named by its
slug. Connections are sandboxed: once the views exist, file access is restricted to the Studio directory and the
configuration is locked, so SQL typed in the dashboard can't read or write anything else on the machine.

Scale: DuckDB streams Parquet out-of-core, so profiles / charts / SQL work on datasets far larger than memory; Python
steps (PCA, clustering, ...) fit on bounded samples and transform in batches. Moving to a warehouse later means
pointing `connect()` at it (DuckDB can ATTACH Postgres/MySQL or query BigQuery/Redshift extracts) — callers only use
`connect()`, `query_df()` and `table_ref()`.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from .. import config

MAX_RESULT_ROWS = 10_000
QUERY_TIMEOUT_S = 120


def root() -> Path:
    d = config.DATA_DIR / "studio"
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d


def table_path(ds_id: str, version: int) -> Path:
    return root() / "tables" / ds_id / f"v{version}.parquet"


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "dataset"
    return ("t_" + s if s[0].isdigit() else s)[:60]


def connect(datasets: list[dict] | None = None, extra: dict | None = None, sandbox: bool = True):
    """A fresh in-memory DuckDB connection with a view per ready dataset (by slug) and optional extra
    pandas/arrow tables. sandbox=True locks file access to the Studio directory."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads={max(1, min(16, os.cpu_count() or 4))}")
    con.execute("SET memory_limit='24GB'")          # spills to temp_directory beyond this (out-of-core)
    (root() / "tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{root() / 'tmp'}'")
    for d in datasets or []:
        v = (d.get("build") or {}).get("version") or d.get("version")
        if d.get("status") == "ready" and v:
            p = table_path(d["id"], v)
            if p.exists():
                con.execute(f"CREATE VIEW {quote_ident(d['slug'])} AS SELECT * FROM read_parquet('{p}')")
    for name, obj in (extra or {}).items():
        con.register(name, obj)
    if sandbox:
        con.execute(f"SET allowed_directories=['{root()}']")
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
    return con


def run_with_timeout(con, sql: str, params=None, timeout: float = QUERY_TIMEOUT_S):
    """Execute and return the relation result; interrupts the query after `timeout` seconds."""
    done = threading.Event()

    def watchdog():
        if not done.wait(timeout):
            try:
                con.interrupt()
            except Exception:
                pass
    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        return con.execute(sql, params) if params is not None else con.execute(sql)
    finally:
        done.set()


def query_df(con, sql: str, params=None, timeout: float = QUERY_TIMEOUT_S):
    return run_with_timeout(con, sql, params, timeout).fetchdf()


def table_ref(ds: dict) -> str:
    """FROM-clause reference to a dataset's current materialization."""
    v = (ds.get("build") or {}).get("version") or ds.get("version")
    p = table_path(ds["id"], v)
    if not v or not p.exists():
        raise ValueError(f"Dataset '{ds['name']}' isn't built yet")
    return f"read_parquet('{p}')"


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, round(time.time() - t0, 3)
