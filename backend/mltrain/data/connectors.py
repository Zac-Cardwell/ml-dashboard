"""Connectors for live data sources on this machine.

  sqlite   : a .db / .sqlite file                      {"kind": "sqlite", "path": "/data/app.db"}
  sql      : any SQLAlchemy URL (PostgreSQL, MySQL…)   {"kind": "sql", "url": "postgresql+psycopg://user:pw@localhost/db"}
  csv      : a CSV file, or a folder / glob of CSVs    {"kind": "csv", "path": "/data/logs/*.csv"}
  parquet  : a Parquet file, or a folder / glob        {"kind": "parquet", "path": "/data/events/"}

CSV / Parquet paths can also be cloud URLs — s3://bucket/prefix/, gs://bucket/*.parquet, az://container/path —
with optional per-source "auth" overrides of the global cloud settings (see mltrain/cloud.py; no secrets stored).
SQL URLs cover cloud warehouses too (Redshift, Athena, BigQuery, Azure SQL / Synapse) and may reference secrets as
${ENV_VAR}, expanded only when connecting.

Rows are read through a time window on the timestamp column; for SQL sources the filter runs in the database.
Timestamps may be epoch seconds, epoch milliseconds or datetime strings / types (auto-detected).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

MAX_ROWS = 2_000_000


class Connector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kind = cfg.get("kind")
        self.tz = cfg.get("timezone", "UTC")
        if self.kind not in ("sqlite", "sql", "csv", "parquet"):
            raise ValueError("connector kind must be sqlite, sql, csv or parquet")
        self._engine = None

    # ------------------------------------------------------------------ SQL
    @property
    def engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine
            if self.kind == "sqlite":
                p = Path(os.path.expanduser(self.cfg["path"]))
                if not p.exists():
                    raise FileNotFoundError(f"SQLite file not found: {p}")
                # read-only: the dashboard never writes to your database
                self._engine = create_engine(f"sqlite:///file:{p}?mode=ro&uri=true")
            else:
                from .. import cloud
                url = cloud.expand_secrets(self.cfg["url"])           # ${VAR} -> secret, never stored expanded
                kw = {"pool_pre_ping": True}
                if url.startswith("bigquery://") and self.cfg.get("auth", {}).get("credentials_file"):
                    kw["credentials_path"] = str(Path(self.cfg["auth"]["credentials_file"]).expanduser())
                try:
                    self._engine = create_engine(url, **kw)
                except Exception as e:      # errors can echo the URL: don't leak the expanded password
                    raise ValueError(cloud.redact(str(e)).replace(url, cloud.redact(url))) from None
        return self._engine

    @property
    def is_sql(self) -> bool:
        return self.kind in ("sqlite", "sql")

    @property
    def cloud(self) -> bool:
        from .. import cloud
        return bool(cloud.provider_of(self.cfg.get("path", "")))

    def _files(self) -> list[str]:
        ext = ".csv" if self.kind == "csv" else ".parquet"
        if self.cloud:
            from .. import cloud
            files = cloud.glob(self.cfg["path"], (ext, ext + ".gz") if ext == ".csv" else (ext, ".pq"), self.cfg.get("auth"))
            if not files:
                raise FileNotFoundError(f"No {ext} files at {self.cfg['path']}")
            return files
        p = os.path.expanduser(self.cfg["path"])
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, f"*{ext}")) + glob.glob(os.path.join(p, f"*{ext}.gz")))
        else:
            files = sorted(glob.glob(p))
        if not files:
            raise FileNotFoundError(f"No {ext} files match {p}")
        return files

    def tables(self) -> list[str]:
        if self.is_sql:
            from sqlalchemy import inspect
            insp = inspect(self.engine)
            return insp.get_table_names() + insp.get_view_names()
        return [Path(f).name for f in self._files()]

    def _from(self, table: str | None, query: str | None) -> str:
        if query:
            return f"({query}) AS src"
        if not table:
            raise ValueError("Choose a table (or provide a SQL query)")
        return self.engine.dialect.identifier_preparer.quote(table)

    def _q(self, name: str) -> str:
        return self.engine.dialect.identifier_preparer.quote(name)

    # ------------------------------------------------------------------ reading
    def sample(self, table: str | None = None, query: str | None = None, n: int = 200) -> pd.DataFrame:
        if self.is_sql:
            return pd.read_sql_query(f"SELECT * FROM {self._from(table, query)} LIMIT {int(n)}", self.engine)
        return self._read_files(nrows=n)

    def _read_files(self, nrows: int | None = None) -> pd.DataFrame:
        frames, total = [], 0
        for f in self._files():
            opts = {}
            if self.cloud:
                from .. import cloud
                opts = {"storage_options": cloud.storage_options(f, self.cfg.get("auth"))}
            df = (pd.read_csv(f, nrows=None if nrows is None else nrows - total, **opts) if self.kind == "csv"
                  else pd.read_parquet(f, **opts))
            frames.append(df)
            total += len(df)
            if nrows is not None and total >= nrows:
                break
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return df.head(nrows) if nrows is not None else df

    def read_window(self, table: str | None, query: str | None, ts_col: str, ts_format: str,
                    start: float | None = None, end: float | None = None, limit: int = MAX_ROWS) -> pd.DataFrame:
        """Rows with start <= ts < end (unix seconds), newest `limit` if capped, returned oldest -> newest,
        plus a float '__ts' column (unix seconds)."""
        if self.is_sql:
            q = self._q(ts_col)
            where, params = [], {}
            if start is not None:
                where.append(f"{q} >= :start")
                params["start"] = to_native(start, ts_format, self.tz)
            if end is not None:
                where.append(f"{q} < :end")
                params["end"] = to_native(end, ts_format, self.tz)
            sql = f"SELECT * FROM {self._from(table, query)}" + (f" WHERE {' AND '.join(where)}" if where else "") + \
                  f" ORDER BY {q} DESC LIMIT {int(limit)}"
            from sqlalchemy import text
            df = pd.read_sql_query(text(sql), self.engine, params=params)
        else:
            df = self._read_files()
        if df.empty:
            df["__ts"] = pd.Series(dtype=float)
            return df
        df["__ts"] = to_epoch(df[ts_col], ts_format, self.tz)
        if not self.is_sql:
            m = pd.Series(True, index=df.index)
            if start is not None:
                m &= df["__ts"] >= start
            if end is not None:
                m &= df["__ts"] < end
            df = df[m]
        df = df.dropna(subset=["__ts"]).sort_values("__ts")
        return df.tail(limit).reset_index(drop=True)

    def bounds(self, table: str | None, query: str | None, ts_col: str, ts_format: str) -> dict:
        """Total rows and first / last timestamp."""
        if self.is_sql:
            q = self._q(ts_col)
            from sqlalchemy import text
            with self.engine.connect() as c:
                n, lo, hi = c.execute(text(f"SELECT COUNT(*), MIN({q}), MAX({q}) FROM {self._from(table, query)}")).one()
            return {"rows": int(n), "first_ts": _one_epoch(lo, ts_format, self.tz), "last_ts": _one_epoch(hi, ts_format, self.tz)}
        df = self._read_files()
        ts = to_epoch(df[ts_col], ts_format, self.tz) if len(df) else pd.Series(dtype=float)
        return {"rows": len(df), "first_ts": float(ts.min()) if len(ts) else None, "last_ts": float(ts.max()) if len(ts) else None}

    def columns(self, table: str | None, query: str | None) -> list[dict]:
        df = self.sample(table, query, 500)
        return [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]


# ---------------------------------------------------------------- timestamps
def detect_ts_format(s: pd.Series) -> str:
    """epoch_s | epoch_ms | datetime (ISO 'YYYY-MM-DD HH:MM:SS') | datetime_T (ISO with a 'T' separator)"""
    v = s.dropna()
    if v.empty:
        return "datetime"
    if pd.api.types.is_numeric_dtype(v):
        return "epoch_ms" if float(v.median()) > 1e11 else "epoch_s"
    return "datetime_T" if "T" in str(v.iloc[0]) else "datetime"


def _local_tz():
    import datetime as _dt
    return _dt.datetime.now().astimezone().tzinfo


def to_epoch(s: pd.Series, fmt: str, tz: str = "UTC") -> pd.Series:
    """tz applies to datetimes without an explicit offset: 'UTC' or 'local' (this machine's time zone)."""
    if fmt == "epoch_s":
        return pd.to_numeric(s, errors="coerce").astype(float)
    if fmt == "epoch_ms":
        return pd.to_numeric(s, errors="coerce").astype(float) / 1000.0
    dt = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(_local_tz() if tz == "local" else "UTC", ambiguous="NaT", nonexistent="NaT")
    return (dt.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def _one_epoch(v, fmt: str, tz: str = "UTC") -> float | None:
    if v is None:
        return None
    r = to_epoch(pd.Series([v]), fmt, tz).iloc[0]
    return None if pd.isna(r) else float(r)


def to_native(t: float, fmt: str, tz: str = "UTC"):
    """unix seconds -> the column's representation, for SQL WHERE clauses."""
    if fmt == "epoch_s":
        return t
    if fmt == "epoch_ms":
        return t * 1000.0
    # ISO strings compare correctly lexicographically when the bound uses the column's layout and time zone
    return pd.Timestamp(t, unit="s", tz="UTC").tz_convert(_local_tz() if tz == "local" else "UTC").strftime("%Y-%m-%dT%H:%M:%S" if fmt == "datetime_T" else "%Y-%m-%d %H:%M:%S")


def timestamp_candidates(df: pd.DataFrame) -> list[str]:
    """Columns that look like timestamps (by name or by content)."""
    out = []
    for c in df.columns:
        name = c.lower()
        s = df[c].dropna()
        if s.empty:
            continue
        if any(k in name for k in ("time", "date", "ts", "created", "updated", "_at")):
            out.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
        elif pd.api.types.is_numeric_dtype(s) and 1e9 < float(s.median()) < 1e13:
            out.append(c)
        elif s.dtype == object and pd.to_datetime(s.head(20), errors="coerce", format="mixed").notna().mean() > 0.9:
            out.append(c)
    return out

