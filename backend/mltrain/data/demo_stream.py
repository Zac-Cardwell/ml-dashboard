"""Demo live database: a SQLite table of loan applications that keeps growing.

    python -m mltrain.data.demo_stream --db data/live/loan_applications.db --backfill-days 30 --rate 1.0

* backfill: 30 days of history; the economy turns over the last 10 days (incomes / credit scores fall,
  defaults rise), so the time window and recency weighting you train with visibly matter
* stream: new applications every second; each outcome is filled in after --label-delay seconds
  (outcomes are only known later, as in real lending)
* --incident: inject a data problem into new rows — missing_values (60% of incomes missing) or
  new_category (a loan purpose the models have never seen)
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import signal
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from ..mlops.demo import drift_params, generate

COLUMNS = ["age", "annual_income", "credit_score", "employment_years", "loan_amount", "debt_to_income", "num_credit_lines",
           "home_ownership", "loan_purpose", "region"]


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def intensity_at(t: float, now: float, drift_days: float) -> float:
    """0 until `drift_days` before now, then ramps to 1 (and stays at 1 while streaming)."""
    return float(np.clip(1 - (now - t) / (drift_days * 86400), 0, 1))


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=30, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS applications (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
              "age REAL, annual_income REAL, credit_score REAL, employment_years REAL, loan_amount REAL, debt_to_income REAL, "
              "num_credit_lines INTEGER, home_ownership TEXT, loan_purpose TEXT, region TEXT, outcome TEXT)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_app_created ON applications(created_at)")
    return c


def insert(c: sqlite3.Connection, ts: np.ndarray, df, y, labeled: bool):
    rows = []
    for t, rec, lab in zip(ts, df.to_dict("records"), y):
        vals = [None if isinstance(rec[k], float) and math.isnan(rec[k]) else rec[k] for k in COLUMNS]
        vals = [v.item() if hasattr(v, "item") else v for v in vals]
        rows.append([_iso(t), *vals, ("default" if lab else "repaid") if labeled else None])
    cur = c.executemany(f"INSERT INTO applications(created_at,{','.join(COLUMNS)},outcome) VALUES (?,{','.join('?' * len(COLUMNS))},?)", rows)
    return cur


def backfill(c, rng, days: float, per_hour: float, drift_days: float, now: float) -> int:
    if c.execute("SELECT COUNT(*) FROM applications").fetchone()[0]:
        return 0
    n = int(days * 24 * per_hour)
    ts = np.sort(now - rng.random(n) * days * 86400)
    # daily rhythm: more applications during the day
    keep = rng.random(n) < 0.55 + 0.45 * np.sin((ts % 86400) / 86400 * 2 * np.pi - np.pi / 2) ** 2
    ts = ts[keep]
    total = 0
    for chunk in np.array_split(ts, max(1, len(ts) // 5000)):
        level = intensity_at(float(chunk.mean()), now, drift_days)
        df, y = generate(len(chunk), rng, drift_params("downturn", level))
        insert(c, chunk, df, y, labeled=True)
        total += len(chunk)
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--backfill-days", type=float, default=30)
    ap.add_argument("--backfill-per-hour", type=float, default=120)
    ap.add_argument("--drift-days", type=float, default=10)
    ap.add_argument("--rate", type=float, default=120 / 3600, help="new rows per second (default matches the backfill: 120/h)")
    ap.add_argument("--label-delay", type=float, default=60)
    ap.add_argument("--incident", default="none", choices=["none", "missing_values", "new_category"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    c = connect(Path(a.db))
    rng = np.random.default_rng(a.seed)
    now = time.time()
    n = backfill(c, rng, a.backfill_days, a.backfill_per_hour, a.drift_days, now)
    if n:
        print(f"backfilled {n} rows over {a.backfill_days} days", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    pending: list[tuple[float, int, str]] = []
    print(f"streaming {a.rate} rows/s, incident={a.incident}", flush=True)
    while not stop["flag"]:
        t0 = time.time()
        k = rng.poisson(a.rate)
        if k:
            df, y = generate(k, rng, drift_params("downturn", 1.0))
            if a.incident == "missing_values":
                df.loc[rng.random(k) < 0.6, "annual_income"] = np.nan
            elif a.incident == "new_category":
                df.loc[rng.random(k) < 0.3, "loan_purpose"] = "crypto"
            insert(c, np.full(k, t0) + rng.random(k) * 0.9, df, y, labeled=False)
            last = c.execute("SELECT MAX(id) FROM applications").fetchone()[0]
            for i, lab in enumerate(y):
                pending.append((t0 + a.label_delay * rng.uniform(0.5, 1.5), last - k + 1 + i, "default" if lab else "repaid"))
        due = [p for p in pending if p[0] <= t0]
        if due:
            c.executemany("UPDATE applications SET outcome=? WHERE id=?", [(lab, rid) for _, rid, lab in due])
            pending = [p for p in pending if p[0] > t0]
        time.sleep(max(0.0, 1.0 - (time.time() - t0)))
    print("stream stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
