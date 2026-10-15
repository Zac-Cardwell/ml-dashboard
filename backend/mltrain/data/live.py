"""Live data sources: health profiling and time-windowed training datasets.

A live source is a table / query / file set that keeps growing, with a timestamp column. The dashboard
  * profiles it on a schedule — freshness, volume, schema changes, missing values, out-of-range values,
    drift (recent window vs. a trailing reference window), target shift — and keeps a health history,
  * exposes it as a training dataset `live.<id>` whose parameters choose the time window, freeze the
    snapshot, split by time, and weight recent rows more than old ones.
"""
from __future__ import annotations

import json
import logging
import re
import time

import numpy as np
import pandas as pd

from .. import db
from ..plugins.base import DatasetPlugin, Param
from .connectors import Connector, detect_ts_format
from .profiling import clean, column_profile, window_drift

log = logging.getLogger("mltrain.data")

DEFAULTS = {"profile_every_minutes": 15, "reference_days": 7, "recent_hours": 24, "quality_minutes": 30, "stale_minutes": None,
            "drift_alert": 0.25, "missing_increase_alert": 0.1, "volume_drop_ratio": 0.3, "volume_spike_ratio": 3.0}


def cfg_of(src: dict) -> dict:
    return {**DEFAULTS, **(src["config"] or {})}


def connector(cfg: dict) -> Connector:
    return Connector({**cfg["connector"], "timezone": cfg.get("timezone", "UTC")})


def feature_columns(cfg: dict, columns: list[str]) -> list[str]:
    skip = {cfg.get("ts_col"), cfg.get("target"), cfg.get("id_col"), "__ts", *(cfg.get("exclude") or [])}
    return [c for c in columns if c not in skip]


def schema_of(df) -> list[dict]:
    """Column types; 'null' for columns with no values yet (e.g. a feature that needs history), so their
    type is learned later instead of being reported as a change."""
    return [{"name": k, "dtype": "null" if df[k].isna().all() else str(df[k].dtype)} for k in df.columns if k != "__ts"]


def _family(dtype: str) -> str:
    d = dtype.lower()
    return "numeric" if any(k in d for k in ("int", "float", "double", "decimal")) else \
        "datetime" if "date" in d or "time" in d else "bool" if "bool" in d else "text"


# ===================================================================== source setup
def inspect(conn_cfg: dict, table: str | None, query: str | None) -> dict:
    """Preview a table for the 'add source' form: columns, sample rows, timestamp candidates."""
    from .connectors import timestamp_candidates
    c = Connector(conn_cfg)
    df = c.sample(table, query, 300)
    return clean({"columns": [{"name": k, "dtype": str(df[k].dtype)} for k in df.columns],
                  "timestamp_candidates": timestamp_candidates(df),
                  "sample": df.head(8).astype(object).where(df.head(8).notna(), None).to_dict("records")})


def create_source(name: str, cfg: dict) -> str:
    for k in ("connector", "ts_col"):
        if not cfg.get(k):
            raise ValueError(f"'{k}' is required")
    if cfg.get("target") and cfg.get("task") not in ("classification", "regression"):
        raise ValueError("task must be classification or regression when a target is set")
    c = Connector({**cfg["connector"], "timezone": cfg.get("timezone", "UTC")})
    sample = c.sample(cfg.get("table"), cfg.get("query"), 500)
    if cfg["ts_col"] not in sample:
        raise ValueError(f"timestamp column '{cfg['ts_col']}' not found")
    if cfg.get("target") and cfg["target"] not in sample:
        raise ValueError(f"target column '{cfg['target']}' not found")
    cfg["ts_format"] = cfg.get("ts_format") or detect_ts_format(sample[cfg["ts_col"]])
    cfg["schema_baseline"] = schema_of(sample)
    sid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "source"
    with db.get_conn() as conn:
        if db.one(conn, "SELECT id FROM data_sources WHERE id=?", (sid,)):
            sid = f"{sid}-{db.new_id()[:4]}"
        conn.execute("INSERT INTO data_sources(id,name,config,created_at) VALUES (?,?,?,?)", (sid, name, json.dumps(cfg), time.time()))
    return sid


# ===================================================================== health profiling
def profile_source(src: dict) -> dict:
    cfg = cfg_of(src)
    now = time.time()
    conn = connector(cfg)
    table, query, ts_col, fmt = cfg.get("table"), cfg.get("query"), cfg["ts_col"], cfg["ts_format"]
    m: dict = {"ts": now}
    try:
        b = conn.bounds(table, query, ts_col, fmt)
    except Exception as e:
        return _finish(src, {"ts": now, "health": "critical",
                             "reasons": [{"severity": "critical", "kind": "connection", "message": f"Cannot read the source: {e}"}]}, str(e))
    m.update(rows=b["rows"], first_ts=b["first_ts"], last_ts=b["last_ts"], lag_s=(now - b["last_ts"]) if b["last_ts"] else None)
    recent_s = cfg["recent_hours"] * 3600
    span_start = now - cfg["reference_days"] * 86400 - recent_s
    df = conn.read_window(table, query, ts_col, fmt, start=span_start, end=now + 3600, limit=1_000_000)
    reasons: list[dict] = []
    m["window_rows"] = len(df)

    # ---- volume: rows per hour
    hourly = pd.Series(dtype=float)
    if len(df):
        hourly = df.groupby((df["__ts"] // 3600).astype(int)).size()
        hourly = hourly.reindex(range(int(span_start // 3600), int(now // 3600) + 1), fill_value=0)
    full = hourly.iloc[:-1] if len(hourly) > 1 else hourly           # last bucket is the incomplete current hour
    typical = float(full.iloc[:-1][full.iloc[:-1] > 0].median()) if len(full) > 2 and (full.iloc[:-1] > 0).any() else None
    last_hour = int(full.iloc[-1]) if len(full) else 0
    m["volume"] = {"last_hour": last_hour, "typical_hour": typical,
                   "ratio": (last_hour / typical) if typical else None}
    # ---- freshness
    gap = 3600 / typical if typical else None
    if cfg["stale_minutes"]:
        stale_after = cfg["stale_minutes"] * 60                 # user-defined
    elif gap:
        stale_after = max(600.0, 5 * gap)                       # 5 typical gaps between rows, at least 10 minutes
    else:
        stale_after = 3600.0
    m["stale_after_s"] = stale_after
    if m["lag_s"] is None:
        reasons.append({"severity": "critical", "kind": "empty", "message": "The source has no rows"})
    elif m["lag_s"] > stale_after:
        reasons.append({"severity": "critical", "kind": "stale",
                        "message": f"No new data for {_dur(m['lag_s'])} (expected at least every {_dur(stale_after)})"})
    if typical and typical >= 5 and m["volume"]["ratio"] is not None:
        if m["volume"]["ratio"] < cfg["volume_drop_ratio"] and not any(r["kind"] == "stale" for r in reasons):
            reasons.append({"severity": "critical", "kind": "volume_drop",
                            "message": f"Only {last_hour} rows last hour vs. ~{typical:.0f} typical ({m['volume']['ratio']:.0%})"})
        elif m["volume"]["ratio"] > cfg["volume_spike_ratio"]:
            reasons.append({"severity": "warning", "kind": "volume_spike",
                            "message": f"{last_hour} rows last hour vs. ~{typical:.0f} typical ({m['volume']['ratio']:.1f}×)"})
    # ---- schema
    current = [{"name": k, "dtype": str(df[k].dtype)} for k in df.columns if k != "__ts"] if len(df) else []
    base = {c["name"]: c["dtype"] for c in cfg.get("schema_baseline") or []}
    cur = {c["name"]: c["dtype"] for c in current}
    changes = []
    if cur:
        changes += [{"change": "removed", "column": k} for k in base if k not in cur]
        changes += [{"change": "added", "column": k} for k in cur if k not in base]
        changes += [{"change": "type", "column": k, "from": base[k], "to": cur[k]} for k in cur
                    if k in base and base[k] != "null" and _family(base[k]) != _family(cur[k]) and not df[k].isna().all()]
        learned = {k: cur[k] for k in cur if base.get(k) == "null" and not df[k].isna().all()}
        if learned:   # first values of a previously empty column: remember its type
            cfg["schema_baseline"] = [{**c, "dtype": learned.get(c["name"], c["dtype"])} for c in cfg["schema_baseline"]]
            with db.get_conn() as c:
                raw = db.one(c, "SELECT config FROM data_sources WHERE id=?", (src["id"],))["config"]
                c.execute("UPDATE data_sources SET config=? WHERE id=?",
                          (json.dumps({**raw, "schema_baseline": cfg["schema_baseline"]}), src["id"]))
    m["schema_changes"] = changes
    for ch in changes:
        sev = "critical" if ch["change"] in ("removed", "type") else "warning"
        reasons.append({"severity": sev, "kind": "schema", "message": f"Column '{ch['column']}' " +
                        {"removed": "disappeared", "added": "appeared (not used until you update the source)",
                         "type": f"changed type {ch.get('from')} → {ch.get('to')}"}[ch["change"]]})
    # ---- quality + drift: recent window vs reference window
    recent = df[df["__ts"] >= now - recent_s] if len(df) else df
    ref = df[df["__ts"] < now - recent_s] if len(df) else df
    m["recent_rows"], m["reference_rows"] = len(recent), len(ref)
    cols = [c for c in feature_columns(cfg, list(df.columns))] + ([cfg["target"]] if cfg.get("target") else [])
    drift = window_drift(ref, recent, cols) if len(ref) >= 50 and len(recent) >= 20 else {}
    # data-quality problems (missing values, new categories, out-of-range) need fast detection: check them on a short
    # window (last `quality_minutes`, at least 20 rows) while drift uses the longer recent window
    fast = df[df["__ts"] >= now - cfg["quality_minutes"] * 60] if len(df) else df
    if len(fast) < 20:
        fast = df.tail(20) if len(df) else df
    qual = window_drift(ref, fast, cols) if len(ref) >= 50 and len(fast) else {}
    m["quality_rows"] = len(fast)
    m["columns"] = {}
    for c in cols:
        if c not in df:
            continue
        d = drift.get(c, {})
        q = qual.get(c, {})
        col = {"missing": float(fast[c].isna().mean()) if len(fast) else None,
               "ref_missing": float(ref[c].isna().mean()) if len(ref) else None,
               "psi": d.get("psi"), "mean": d.get("cur_mean"), "ref_mean": d.get("ref_mean"),
               "out_of_range": q.get("out_of_range"), "unseen": q.get("unseen"), "unseen_examples": q.get("unseen_examples"),
               "kind": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical", "is_target": c == cfg.get("target"),
               "drift_status": d.get("status"), "psi_noise": d.get("psi_noise")}
        m["columns"][c] = col
        if c == cfg.get("target"):
            continue
        if col["missing"] is not None and col["ref_missing"] is not None and col["missing"] - col["ref_missing"] >= cfg["missing_increase_alert"]:
            reasons.append({"severity": "warning", "kind": "missing", "column": c,
                            "message": f"'{c}' missing rate jumped to {col['missing']:.0%} in the last {cfg['quality_minutes']} min (was {col['ref_missing']:.0%})"})
        if col["psi"] is not None and col["psi"] >= max(cfg["drift_alert"], col["psi_noise"] or 0):
            reasons.append({"severity": "warning", "kind": "drift", "column": c,
                            "message": f"'{c}' drifted (PSI {col['psi']:.2f}) in the last {cfg['recent_hours']}h vs. the previous {cfg['reference_days']}d"})
        if (col["out_of_range"] or 0) >= 0.05:
            reasons.append({"severity": "warning", "kind": "range", "column": c,
                            "message": f"{col['out_of_range']:.0%} of recent '{c}' values are outside the historical range"})
        if (col["unseen"] or 0) >= 0.05:
            reasons.append({"severity": "warning", "kind": "categories", "column": c,
                            "message": f"{col['unseen']:.0%} of recent '{c}' values are new categories ({', '.join(col['unseen_examples'] or [])})"})
    if cfg.get("id_col") and cfg["id_col"] in recent and len(recent):
        dup = float(recent[cfg["id_col"]].duplicated().mean())
        m["duplicate_ids"] = dup
        if dup > 0:
            reasons.append({"severity": "warning", "kind": "duplicates", "message": f"{dup:.1%} of recent rows reuse an existing id"})
    # ---- target / labels
    t = cfg.get("target")
    if t and t in df and len(recent):
        m["label_coverage"] = float(recent[t].notna().mean())
        tc = m["columns"].get(t, {})
        if tc.get("psi") is not None and tc["psi"] >= max(cfg["drift_alert"], tc.get("psi_noise") or 0):
            reasons.append({"severity": "warning", "kind": "target_shift", "column": t,
                            "message": f"The target distribution shifted (PSI {tc['psi']:.2f}) — models may need retraining"})
        if cfg.get("task") == "classification":
            m["target_recent"] = {str(k): float(v) for k, v in recent[t].dropna().astype(str).value_counts(normalize=True).items()}
            m["target_reference"] = {str(k): float(v) for k, v in ref[t].dropna().astype(str).value_counts(normalize=True).items()} if len(ref) else {}
        else:
            m["target_recent"] = {"mean": float(pd.to_numeric(recent[t], errors="coerce").mean())}
            m["target_reference"] = {"mean": float(pd.to_numeric(ref[t], errors="coerce").mean())} if len(ref) else {}
    m["reasons"] = reasons
    m["health"] = "critical" if any(r["severity"] == "critical" for r in reasons) else "warning" if reasons else "healthy"
    return _finish(src, m, None)


def _finish(src: dict, m: dict, err: str | None) -> dict:
    m = clean(m)
    with db.get_conn() as c:
        c.execute("INSERT INTO source_snapshots(source_id,ts,health,metrics) VALUES (?,?,?,?)",
                  (src["id"], m["ts"], m["health"], json.dumps(m)))
        c.execute("UPDATE data_sources SET health=?, health_reasons=?, last_profiled_at=?, last_error=? WHERE id=?",
                  (m["health"], json.dumps(m.get("reasons", [])), m["ts"], err, src["id"]))
        c.execute("DELETE FROM source_snapshots WHERE source_id=? AND ts<?", (src["id"], m["ts"] - 30 * 86400))
    if m["health"] in ("warning", "critical") and m["health"] != src.get("health"):
        from ..notify import notify
        notify("data_source", f"Data source {src['name']}: {m['health']}",
               "\n".join(r["message"] for r in m.get("reasons", [])[:5]), f"/data/sources/{src['id']}",
               m["health"], key=f"source:{src['id']}:{m['health']}")
    return m


def _dur(s: float) -> str:
    s = float(s)
    return f"{s / 86400:.1f} days" if s >= 86400 else f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def tick() -> None:
    with db.get_conn() as c:
        sources = db.rows(c, "SELECT * FROM data_sources")
    now = time.time()
    for s in sources:
        if not s["last_profiled_at"] or now - s["last_profiled_at"] >= cfg_of(s)["profile_every_minutes"] * 60:
            try:
                profile_source(s)
            except Exception:
                log.exception("profiling %s failed", s["id"])


def hourly_volume(src: dict, hours: float = 168) -> dict:
    """Rows per hour and (for classification) target shares per day, for charts."""
    cfg = cfg_of(src)
    now = time.time()
    df = connector(cfg).read_window(cfg.get("table"), cfg.get("query"), cfg["ts_col"], cfg["ts_format"], start=now - hours * 3600,
                                    end=now + 3600, limit=1_000_000)
    out: dict = {"hours": [], "counts": []}
    if len(df):
        h = df.groupby((df["__ts"] // 3600).astype(int)).size().reindex(range(int((now - hours * 3600) // 3600), int(now // 3600) + 1), fill_value=0)
        out = {"hours": [int(k) * 3600 for k in h.index], "counts": h.astype(int).tolist()}
        t = cfg.get("target")
        if t and t in df:
            day = (df["__ts"] // 86400).astype(int)
            if cfg.get("task") == "classification":
                tab = pd.crosstab(day, df[t].astype(str), normalize="index")
                out["target_daily"] = {"days": [int(d) * 86400 for d in tab.index], "classes": list(tab.columns),
                                       "shares": tab.round(4).values.tolist(), "labeled": df.groupby(day)[t].apply(lambda s: float(s.notna().mean())).round(4).tolist()}
            else:
                g = pd.to_numeric(df[t], errors="coerce").groupby(day).mean()
                out["target_daily"] = {"days": [int(d) * 86400 for d in g.index], "mean": g.round(5).tolist()}
    return clean(out)


def preview(src: dict, n: int = 50) -> list[dict]:
    cfg = cfg_of(src)
    df = connector(cfg).read_window(cfg.get("table"), cfg.get("query"), cfg["ts_col"], cfg["ts_format"], limit=n)
    df = df.iloc[::-1].drop(columns=["__ts"])
    return clean(df.astype(object).where(df.notna(), None).to_dict("records"))


def column_detail(src: dict, column: str, hours: float = 24) -> dict:
    cfg = cfg_of(src)
    now = time.time()
    df = connector(cfg).read_window(cfg.get("table"), cfg.get("query"), cfg["ts_col"], cfg["ts_format"],
                                    start=now - (cfg["reference_days"] * 24 + hours) * 3600, end=now + 3600, limit=500_000)
    if column not in df:
        raise ValueError(f"column '{column}' not found")
    rec, ref = df[df["__ts"] >= now - hours * 3600], df[df["__ts"] < now - hours * 3600]
    return clean({"recent": column_profile(rec[column]) if len(rec) else None, "reference": column_profile(ref[column]) if len(ref) else None})


# ===================================================================== training datasets
WEIGHTINGS = ["none", "exponential", "linear", "step"]


def recency_weights(age_h: np.ndarray, p: dict) -> np.ndarray:
    """Per-row training weights from row age (hours before the window end), normalized to mean 1."""
    kind = p.get("weighting", "none")
    if kind == "exponential":
        w = 0.5 ** (age_h / max(p["half_life_hours"], 1e-6))
    elif kind == "linear":
        w = 1 - age_h / max(age_h.max(), 1e-6) * (1 - p["min_weight"])
    elif kind == "step":
        w = np.where(age_h <= p["recent_hours"], p["recent_weight"], 1.0)
    else:
        w = np.ones_like(age_h)
    w = np.maximum(w, p.get("min_weight", 0.0) if kind != "step" else 0.0)
    return w / w.mean() if len(w) and w.mean() > 0 else w


def live_params() -> list[Param]:
    from ..builtin.datasets import TABULAR_PARAMS
    split = [p for p in TABULAR_PARAMS if p.name in ("val_fraction", "test_fraction", "standardize")]
    return [
        Param("window", "choice", "last", "Train on the most recent period, or a fixed date range", choices=["last", "range"], group="Time window"),
        Param("lookback_hours", "float", 24 * 30, "How far back to train (hours) when window = last", 0.01, group="Time window"),
        Param("start", "str", "", "Range start (YYYY-MM-DD[ HH:MM]) when window = range", group="Time window"),
        Param("end", "str", "", "Range end (empty = now) when window = range", group="Time window"),
        Param("until", "float", 0.0, "Freeze the snapshot at this unix time (0 = when the run starts)", 0.0, group="Time window"),
        Param("max_rows", "int", 0, "Keep at most the newest N rows (0 = all)", 0, group="Time window"),
        Param("weighting", "choice", "none", "Give recent rows more weight during training", choices=WEIGHTINGS, group="Recency weighting"),
        Param("half_life_hours", "float", 24 * 7, "Exponential: a row this much older than the newest training row counts half", 0.01, group="Recency weighting"),
        Param("recent_hours", "float", 48, "Step: rows within this many hours of the newest training row get the extra weight", 0.01, group="Recency weighting"),
        Param("recent_weight", "float", 3.0, "Step: weight of recent rows (older rows = 1)", 0.0, group="Recency weighting"),
        Param("min_weight", "float", 0.05, "Linear / exponential: floor so old rows never vanish", 0.0, 1.0, group="Recency weighting"),
        Param("downsample_older_than_hours", "float", 0.0, "Keep only a fraction of rows older than this (0 = off)", 0.0, group="Recency weighting"),
        Param("downsample_rate", "float", 0.25, "Fraction of old rows to keep", 0.01, 1.0, group="Recency weighting"),
        Param("split", "choice", "time", "time = train on older rows, validate/test on the newest (recommended)", choices=["time", "random"], group="Data split"),
        *split,
        Param("split_seed", "int", 0, "Random split / downsampling seed", -1, group="Data split"),
    ]


def _parse_time(s: str) -> float:
    return pd.Timestamp(s).tz_localize("UTC").timestamp() if pd.Timestamp(s).tzinfo is None else pd.Timestamp(s).timestamp()


def make_dataset_plugin(src: dict) -> DatasetPlugin:
    cfg = cfg_of(src)

    class _Live(DatasetPlugin):
        id = f"live.{src['id']}"
        name = f"{src['name']} (live)"
        description = f"Live source · {cfg['connector'].get('kind')} · target '{cfg.get('target')}' · timestamp '{cfg['ts_col']}'"
        task = cfg.get("task") or "classification"
        modality = "tabular"
        source_id = src["id"]

        def params(self):
            return live_params()

        def load(self, params, seed):
            from ..builtin.datasets import split_indices, tabular_bundle
            from ..mlops.datasets import time_split
            end = params.get("until") or time.time()
            if params["window"] == "range":
                if not params.get("start"):
                    raise ValueError("window = range needs a start date")
                start = _parse_time(params["start"])
                end = _parse_time(params["end"]) if params.get("end") else end
            else:
                start = end - params["lookback_hours"] * 3600
            df = connector(cfg).read_window(cfg.get("table"), cfg.get("query"), cfg["ts_col"], cfg["ts_format"], start=start, end=end,
                                            limit=params.get("max_rows") or 2_000_000)
            target = cfg.get("target")
            if not target:
                raise ValueError("This source has no target column configured")
            df = df.dropna(subset=[target]).reset_index(drop=True)
            rng = np.random.default_rng(params.get("split_seed", 0) if params.get("split_seed", 0) >= 0 else seed)
            if params.get("downsample_older_than_hours"):
                old = (end - df["__ts"]) / 3600 > params["downsample_older_than_hours"]
                keep = ~old | (rng.random(len(df)) < params["downsample_rate"])
                df = df[keep].reset_index(drop=True)
            if len(df) < 30:
                raise ValueError(f"Only {len(df)} labeled rows in the chosen window — widen it")
            feats = feature_columns(cfg, list(df.columns))
            X = df[feats]
            if self.task == "classification":
                yv = df[target].astype(str)
                names = sorted(yv.unique())
                y = yv.map({n: i for i, n in enumerate(names)}).to_numpy()
            else:
                names, y = None, pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
            idx = time_split(len(df), params) if params["split"] == "time" else split_indices(len(df), y, params, seed, self.task)
            b = tabular_bundle(X, y, params, seed, self.task, class_names=names, indices=idx)
            # age relative to the newest *training* row: with a time split the newest part of the window is val/test,
            # so measuring from the window end would leave no training row "recent"
            train_ts = df["__ts"].to_numpy()[idx[0]]
            age_h = (train_ts.max() - train_ts) / 3600 if len(train_ts) else train_ts
            w = recency_weights(age_h, params)
            b.weights = {"train": w.astype(np.float32)}
            ts = df["__ts"].to_numpy()
            b.info["window"] = {"start": float(start), "end": float(end), "first_row": float(ts.min()), "last_row": float(ts.max()),
                                "rows": len(df), "weighting": params["weighting"],
                                "effective_train_rows": float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0,
                                "split": params["split"],
                                "test_from": float(ts[idx[2]].min()) if len(idx[2]) else None}
            return b

    return _Live()


def live_datasets() -> list[DatasetPlugin]:
    try:
        with db.get_conn() as c:
            sources = db.rows(c, "SELECT * FROM data_sources")
    except Exception:
        return []
    return [make_dataset_plugin(s) for s in sources if (s["config"] or {}).get("target")]
