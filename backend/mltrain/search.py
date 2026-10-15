"""Smart hyperparameter search with Optuna (TPE Bayesian optimization + median pruning).

Each trial is an ordinary training session in the queue (group_kind='search'), so it gets curves,
snapshots, saved models and lineage like any other run. A background loop:
  * asks Optuna for new parameters while fewer than `parallel` trials are running,
  * reports each running trial's validation curve to the pruner and stops hopeless trials early,
  * tells Optuna the final validation metric when a trial completes.
Studies persist in data/optuna.db, so a search survives server restarts.

The objective is a *validation* metric by default: optimizing on test would overfit the test set.
"""
from __future__ import annotations

import json
import logging
import time

from fastapi import HTTPException

from . import analysis, config, db

log = logging.getLogger("mltrain.search")

AUTO_METRIC = {  # task -> (objective metric, direction, curve used for pruning)
    "classification": ("val/acc", "maximize", "val/acc"),
    "regression": ("val/rmse", "minimize", "val/rmse"),
    "rl": ("final/mean_reward", "maximize", "eval/mean_reward"),
    "seq2seq": ("val/bleu", "maximize", "val/bleu"),
    "generation": ("val/loss", "minimize", "val/loss"),
    "clustering": ("val/silhouette", "maximize", "val/silhouette"),
    "anomaly_detection": ("val/auc", "maximize", "val/auc"),
    "dimensionality_reduction": ("val/reconstruction_rmse", "minimize", "val/reconstruction_rmse"),
}


def _optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def storage_url() -> str:
    return f"sqlite:///{config.DATA_DIR / 'optuna.db'}"


def distributions(space: dict):
    optuna = _optuna()
    out = {}
    for k, d in space.items():
        t = d.get("type")
        if t == "float":
            out[k] = optuna.distributions.FloatDistribution(float(d["low"]), float(d["high"]), log=bool(d.get("log")))
        elif t == "int":
            out[k] = optuna.distributions.IntDistribution(int(d["low"]), int(d["high"]), log=bool(d.get("log")),
                                                          step=int(d.get("step") or 1) if not d.get("log") else 1)
        elif t == "categorical":
            if not d.get("choices"):
                raise HTTPException(400, f"'{k}': categorical parameters need choices")
            out[k] = optuna.distributions.CategoricalDistribution([str(c) if isinstance(c, list) else c for c in d["choices"]])
        else:
            raise HTTPException(400, f"'{k}': type must be float, int or categorical")
        if t in ("float", "int") and float(d["low"]) >= float(d["high"]):
            raise HTTPException(400, f"'{k}': low must be < high")
        if t in ("float", "int") and d.get("log") and float(d["low"]) <= 0:
            raise HTTPException(400, f"'{k}': log scale needs low > 0")
    return out


def create(cfg: dict) -> str:
    """cfg: {name, base: SessionCreate dict, space, n_trials, parallel, metric, direction, pruning, sampler}"""
    from .api.sessions import SessionCreate
    optuna = _optuna()
    base = SessionCreate(**cfg["base"])
    from .plugins.registry import get_registry
    problem = get_registry().problem(base.problem_kind, base.problem_id)
    if not cfg.get("space"):
        raise HTTPException(400, "The search space is empty")
    for k in cfg["space"]:
        if k.split(".", 1)[0] not in ("model", "problem") or "." not in k:
            raise HTTPException(400, f"'{k}' must look like model.<param> or problem.<param>")
    distributions(cfg["space"])  # validate
    metric, direction, prune_key = AUTO_METRIC.get(problem.task, ("test/acc", "maximize", None))
    if cfg.get("metric") and cfg["metric"] != "auto":
        metric = cfg["metric"]
        direction = cfg.get("direction") if cfg.get("direction") in ("maximize", "minimize") else \
            ("minimize" if analysis_lower_better(metric) else "maximize")
    n_trials = int(cfg.get("n_trials") or 20)
    if not 1 <= n_trials <= 500:
        raise HTTPException(400, "n_trials must be 1-500")
    cfg = {**cfg, "n_trials": n_trials, "parallel": max(1, int(cfg.get("parallel") or 2)), "prune_key": cfg.get("prune_key") or prune_key,
           "pruning": bool(cfg.get("pruning", True)), "sampler": cfg.get("sampler") or "tpe", "task": problem.task}
    sid = "s" + db.new_id()
    sampler = optuna.samplers.RandomSampler(seed=0) if cfg["sampler"] == "random" else optuna.samplers.TPESampler(seed=0, n_startup_trials=min(5, n_trials))
    optuna.create_study(study_name=sid, storage=storage_url(), direction=direction, sampler=sampler)
    now = time.time()
    with db.get_conn() as c:
        c.execute("INSERT INTO searches(id,name,status,config,metric,direction,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                  (sid, cfg.get("name") or f"Search · {problem.name}", "running", json.dumps(cfg), metric, direction, now, now))
    return sid


def analysis_lower_better(metric: str) -> bool:
    return metric.split("/")[-1] in ("loss", "rmse", "mse", "mae")


def _study(sid: str, cfg: dict, direction: str):
    optuna = _optuna()
    sampler = optuna.samplers.RandomSampler() if cfg["sampler"] == "random" else optuna.samplers.TPESampler(n_startup_trials=min(5, cfg["n_trials"]))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1) if cfg["pruning"] else optuna.pruners.NopPruner()
    return optuna.load_study(study_name=sid, storage=storage_url(), sampler=sampler, pruner=pruner)


def _session_body(cfg: dict, params: dict, number: int, name: str):
    from .api.sessions import SessionCreate
    b = SessionCreate(**cfg["base"])
    for k, v in params.items():
        section, pname = k.split(".", 1)
        getattr(b, f"{section}_params")[pname] = v
    b.name = f"{name} · trial {number}"
    b.options = {**(b.options or {}), "snapshots": bool((b.options or {}).get("snapshots", False))}
    return b


def tick() -> None:
    with db.get_conn() as c:
        searches = db.rows(c, "SELECT * FROM searches WHERE status='running'")
    for s in searches:
        try:
            _tick_search(s)
        except Exception:
            log.exception("search %s tick failed", s["id"])


def _tick_search(s: dict) -> None:
    optuna = _optuna()
    from .api.sessions import create_session
    cfg = s["config"]
    study = _study(s["id"], cfg, s["direction"])
    with db.get_conn() as c:
        trials = db.rows(c, "SELECT * FROM search_trials WHERE search_id=? ORDER BY trial_number", (s["id"],))
    running = [t for t in trials if t["state"] in ("running", "pruning")]   # 'pruning' = stop requested, waiting for it to end

    for t in running:
        with db.get_conn() as c:
            runs = db.rows(c, "SELECT id, seed, status, summary FROM runs WHERE session_id=?", (t["session_id"],))
        st = analysis.session_status(runs)
        trial = optuna.trial.Trial(study, study._storage.get_trial_id_from_study_id_trial_number(study._study_id, t["trial_number"]))
        if st == "completed":
            stats = analysis.summary_stats(runs)
            value = stats.get(s["metric"], {}).get("mean")
            if value is None:
                study.tell(t["trial_number"], state=optuna.trial.TrialState.FAIL)
                _set_trial(s["id"], t["trial_number"], "failed", None)
            else:
                study.tell(t["trial_number"], value)
                _set_trial(s["id"], t["trial_number"], "complete", value)
        elif st in ("failed", "partial", "stopped", "empty"):
            pruned = t.get("state") == "pruning"
            study.tell(t["trial_number"], state=optuna.trial.TrialState.PRUNED if pruned else optuna.trial.TrialState.FAIL)
            _set_trial(s["id"], t["trial_number"], "pruned" if pruned else "failed", None)
        elif cfg["pruning"] and cfg.get("prune_key") and st == "running" and t["state"] == "running":
            # report the learning curve of the first seed; stop the trial if it's worse than the median of earlier trials
            with db.get_conn() as c:
                pts = db.rows(c, "SELECT step, value FROM metrics WHERE run_id=? AND key=? ORDER BY step", (runs[0]["id"], cfg["prune_key"]))
            last = (t["params"] or {}).get("__reported_step", -1)
            for p in pts:
                if p["step"] > last:
                    trial.report(p["value"], p["step"])   # the pruner reads the study direction itself
            if pts:
                _set_trial(s["id"], t["trial_number"], "running", None, reported_step=pts[-1]["step"])
                if trial.should_prune():
                    with db.get_conn() as c:
                        c.execute("UPDATE runs SET status='stopped', ended_at=? WHERE session_id=? AND status='queued'", (time.time(), t["session_id"]))
                        c.execute("UPDATE runs SET status='stopping' WHERE session_id=? AND status='running'", (t["session_id"],))
                    _set_trial(s["id"], t["trial_number"], "pruning", None)
                    log.info("search %s: pruned trial %s at step %s", s["id"], t["trial_number"], pts[-1]["step"])

    with db.get_conn() as c:
        trials = db.rows(c, "SELECT * FROM search_trials WHERE search_id=?", (s["id"],))
    active = sum(t["state"] in ("running", "pruning") for t in trials)
    asked = len(trials)
    dists = distributions(cfg["space"])
    while active < cfg["parallel"] and asked < cfg["n_trials"]:
        trial = study.ask(dists)
        params = dict(trial.params)   # (int_list choices are strings like "64,64"; coerce_params parses them)
        with db.get_conn() as c:
            c.execute("BEGIN")
            sess = create_session(c, _session_body(cfg, params, trial.number, s["name"]), s["id"], "search",
                                  {k: v for k, v in params.items()})
            c.execute("INSERT INTO search_trials(search_id,trial_number,session_id,state,params) VALUES (?,?,?,?,?)",
                      (s["id"], trial.number, sess, "running", json.dumps(params)))
            c.execute("COMMIT")
        active += 1
        asked += 1

    if asked >= cfg["n_trials"] and active == 0:
        _finish(s, study)
    else:
        _update_best(s, study)


def _set_trial(sid: str, number: int, state: str, value: float | None, reported_step: int | None = None) -> None:
    with db.get_conn() as c:
        if reported_step is not None:
            c.execute("UPDATE search_trials SET params=json_set(params, '$.__reported_step', ?) WHERE search_id=? AND trial_number=?",
                      (reported_step, sid, number))
        c.execute("UPDATE search_trials SET state=?, value=COALESCE(?, value) WHERE search_id=? AND trial_number=?", (state, value, sid, number))


def _update_best(s: dict, study) -> None:
    try:
        best = study.best_trial
    except ValueError:
        return
    with db.get_conn() as c:
        row = db.one(c, "SELECT session_id FROM search_trials WHERE search_id=? AND trial_number=?", (s["id"], best.number))
        c.execute("UPDATE searches SET best_session_id=?, best_value=?, updated_at=? WHERE id=?",
                  (row["session_id"] if row else None, best.value, time.time(), s["id"]))


def _finish(s: dict, study) -> None:
    _update_best(s, study)
    with db.get_conn() as c:
        c.execute("UPDATE searches SET status='completed', updated_at=? WHERE id=?", (time.time(), s["id"]))
        s2 = db.one(c, "SELECT * FROM searches WHERE id=?", (s["id"],))
        best_name = db.one(c, "SELECT name FROM sessions WHERE id=?", (s2["best_session_id"],)) if s2["best_session_id"] else None
    from .notify import notify
    params = ""
    try:
        params = ", ".join(f"{k.split('.', 1)[-1]}={v:.4g}" if isinstance(v, float) else f"{k.split('.', 1)[-1]}={v}"
                           for k, v in study.best_trial.params.items())
    except ValueError:
        pass
    notify("group_completed", f"Hyperparameter search finished: {s['name']}",
           f"Best {s['metric']} = {s2['best_value']:.4g} ({best_name['name'] if best_name else '—'})\n{params}" if s2["best_value"] is not None
           else "No trial completed successfully.", f"/compare?group={s['id']}", "success", key=f"search:{s['id']}")


def stop(sid: str) -> None:
    with db.get_conn() as c:
        c.execute("UPDATE searches SET status='stopped', updated_at=? WHERE id=?", (time.time(), sid))
        for t in db.rows(c, "SELECT session_id FROM search_trials WHERE search_id=? AND state IN ('running','pruning')", (sid,)):
            c.execute("UPDATE runs SET status='stopped', ended_at=? WHERE session_id=? AND status='queued'", (time.time(), t["session_id"]))
            c.execute("UPDATE runs SET status='stopping' WHERE session_id=? AND status='running'", (t["session_id"],))


def detail(sid: str) -> dict:
    optuna = _optuna()
    with db.get_conn() as c:
        s = db.one(c, "SELECT * FROM searches WHERE id=?", (sid,))
        if not s:
            raise HTTPException(404, "search not found")
        trials = db.rows(c, "SELECT t.*, se.name AS session_name FROM search_trials t LEFT JOIN sessions se ON se.id=t.session_id "
                            "WHERE t.search_id=? ORDER BY t.trial_number", (sid,))
    for t in trials:
        t["params"] = {k: v for k, v in (t["params"] or {}).items() if not k.startswith("__")}
    importance = {}
    try:
        study = optuna.load_study(study_name=sid, storage=storage_url())
        if sum(t["state"] == "complete" for t in trials) >= 4 and len(s["config"]["space"]) > 1:
            importance = {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception as e:
        log.debug("importance failed: %s", e)
    return {**s, "trials": trials, "importance": importance}
