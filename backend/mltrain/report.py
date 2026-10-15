"""Static result reports: summary statistics, learning-curve / confusion-matrix / comparison plots (matplotlib PNGs)
and markdown tables. Used by portfolio exports (README figures) and by their standalone `train.py`."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HEADLINE = {  # task -> (metric, higher_is_better)   (same as the dashboard)
    "classification": ("test/acc", True),
    "regression": ("test/rmse", False),
    "rl": ("final/mean_reward", True),
    "seq2seq": ("test/bleu", True),
    "generation": ("test/fid", False),
}
# tasks whose headline depends on what was measured (images: FID; text: perplexity; otherwise test loss)
HEADLINE_FALLBACK = {"generation": [("test/fid", False), ("test/perplexity", False), ("test/loss", False)]}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"
SPLIT_ORDER = {"train": 0, "rollout": 0, "val": 1, "eval": 1, "test": 2}
PRIORITY = ["ep_rew_mean", "mean_reward", "loss", "acc", "auc", "f1", "rmse", "mae", "r2", "bleu", "rouge", "ep_len_mean"]


# --------------------------------------------------------------------------- numbers
def summary_stats(summaries: dict[int, dict]) -> dict[str, dict]:
    vals = defaultdict(list)
    for seed, s in summaries.items():
        for k, v in (s or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                vals[k].append((seed, float(v)))
    out = {}
    for k, sv in sorted(vals.items()):
        a = [v for _, v in sv]
        m = sum(a) / len(a)
        sd = math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1)) if len(a) > 1 else 0.0
        out[k] = {"mean": m, "std": sd, "min": min(a), "max": max(a), "n": len(a), "per_seed": {str(s): v for s, v in sv}}
    return out


def headline(task: str | None, stats: dict) -> dict | None:
    if task in HEADLINE_FALLBACK:
        key, hib = next(((k, h) for k, h in HEADLINE_FALLBACK[task] if k in stats), (None, True))
        if key is None:
            return None
    elif task not in HEADLINE or HEADLINE[task][0] not in stats:
        return None
    else:
        key, hib = HEADLINE[task]
    return {"key": key, "higher_is_better": hib, **{k: stats[key][k] for k in ("mean", "std", "n")}}


def fmt(v, key: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) >= 100):
        return f"{int(v):,}"
    if any(t in key for t in ("acc", "f1", "precision", "recall", "auc")) and 0 <= v <= 1:
        return f"{v * 100:.2f}%"
    a = abs(v)
    return f"{v:.4g}" if a >= 1e-3 or a == 0 else f"{v:.2e}"


def fmt_pm(st: dict, key: str) -> str:
    return fmt(st["mean"], key) + (f" ± {fmt(st['std'], key)}" if st["n"] > 1 else "")


def results_table(stats: dict, keys: list[str] | None = None) -> str:
    keys = keys or [k for k in stats if not k.startswith("final_model/")]
    rank = lambda k: (0 if k.startswith(("test/", "final/")) else 1 if k.startswith("val/") else 2 if "/" in k else 3, k)
    keys = sorted(keys, key=rank)
    if all(stats[k]["n"] == 1 for k in keys):
        return "\n".join(["| Metric | Value |", "|---|---:|"] + [f"| `{k}` | {fmt(stats[k]['mean'], k)} |" for k in keys])
    rows = ["| Metric | Mean ± std | Min | Max | Seeds |", "|---|---:|---:|---:|---:|"]
    for k in keys:
        st = stats[k]
        rows.append(f"| `{k}` | {fmt_pm(st, k)} | {fmt(st['min'], k)} | {fmt(st['max'], k)} | {st['n']} |")
    return "\n".join(rows)


def best_vs_final_table(stats: dict) -> str | None:
    keys = [k[len("final_model/"):] for k in stats if k.startswith("final_model/")]
    keys = [k for k in keys if k in stats and re.match(r"^(val|test|final)/", k)]
    if not keys:
        return None
    rows = ["| Metric | Best checkpoint | Final weights |", "|---|---:|---:|"]
    for k in keys:
        rows.append(f"| `{k}` | {fmt_pm(stats[k], k)} | {fmt_pm(stats['final_model/' + k], k)} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- reading runs
def read_metrics(run_dir: Path) -> dict[str, tuple[list, list]]:
    out: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    p = Path(run_dir) / "metrics.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["value"])
            except ValueError:
                continue
            if math.isfinite(v):
                s, vs = out[row["key"]]
                s.append(int(row["step"]))
                vs.append(v)
    return dict(out)


def read_runs(exp_dir: Path) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    """-> ({seed: series}, {seed: summary}, {seed: {artifact_name: artifact}})"""
    series, summaries, artifacts = {}, {}, {}
    for d in sorted(Path(exp_dir).glob("seed_*")):
        seed = int(d.name.split("_")[1])
        series[seed] = read_metrics(d)
        if (d / "summary.json").exists():
            summaries[seed] = json.loads((d / "summary.json").read_text())
        artifacts[seed] = {p.stem: json.loads(p.read_text()) for p in sorted((d / "artifacts").glob("*.json"))} if (d / "artifacts").exists() else {}
    return series, summaries, artifacts


# --------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 10, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 110,
                         "savefig.bbox": "tight", "savefig.facecolor": "white"})
    return plt


def curve_groups(keys) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        if k.startswith(("final_model/", "time/")) or "/" not in k:
            continue
        groups[k.split("/", 1)[1]].append(k)
    def rank(name):
        if name in PRIORITY:
            return PRIORITY.index(name)
        for i, p in enumerate(PRIORITY):        # e.g. entropy_loss: after the exact matches
            if name.endswith(p):
                return len(PRIORITY) + i
        return 2 * len(PRIORITY)
    out = sorted(groups.items(), key=lambda kv: (rank(kv[0]), kv[0]))
    return [(n, sorted(ks, key=lambda k: SPLIT_ORDER.get(k.split("/")[0], 3))) for n, ks in out]


def plot_curves(series: dict[int, dict], out: Path, title: str = "", max_panels: int = 6) -> Path | None:
    """One panel per metric (train/val/test together); line = mean over seeds, band = min–max across seeds."""
    keys = sorted({k for s in series.values() for k, (st, _) in s.items() if len(st) > 1})
    groups = curve_groups(keys)[:max_panels]
    if not groups:
        return None
    plt = _plt()
    n = len(groups)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax, (name, ks) in zip(axes.flat, groups):
        for i, k in enumerate(ks):
            by_step: dict[int, list] = defaultdict(list)
            for s in series.values():
                if k in s:
                    for st, v in zip(*s[k]):
                        by_step[st].append(v)
            steps = sorted(by_step)
            if not steps:
                continue
            mean = [sum(by_step[t]) / len(by_step[t]) for t in steps]
            c = SERIES[i % len(SERIES)]
            if len(series) > 1:
                ax.fill_between(steps, [min(by_step[t]) for t in steps], [max(by_step[t]) for t in steps], color=c, alpha=0.15, linewidth=0)
            ax.plot(steps, mean, color=c, linewidth=2, label=k.split("/")[0])
        ax.set_title(name)
        ax.set_xlabel("step")
        if len(ks) > 1:
            ax.legend(loc="best")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title + (f"  (mean of {len(series)} seeds, band = min–max)" if len(series) > 1 else ""), color=INK, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_confusion(art: dict, out: Path, title: str = "Confusion matrix (test)") -> Path | None:
    data = art.get("data", art)
    m, labels = data.get("matrix"), data.get("labels")
    if not m or len(m) > 30:
        return None
    plt = _plt()
    import numpy as np
    a = np.array(m, dtype=float)
    norm = a / np.maximum(a.sum(axis=1, keepdims=True), 1)
    sz = max(3.2, 0.45 * len(m) + 1.6)
    fig, ax = plt.subplots(figsize=(sz + 0.8, sz))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=45 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    if len(m) <= 12:
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{int(a[i, j])}", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of actual class")
    ax.set_title(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_comparison(rows: list[dict], out: Path) -> Path | None:
    """rows: [{"label", "key", "mean", "std", "higher_is_better"}] for experiments sharing one headline metric."""
    if len(rows) < 2:
        return None
    plt = _plt()
    rows = sorted(rows, key=lambda r: r["mean"], reverse=rows[0]["higher_is_better"])
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.3))
    y = list(range(len(rows)))
    # dot plot (not bars): the axis can zoom in to the range that matters without a misleading truncated bar
    ax.errorbar([r["mean"] for r in rows], y, xerr=[r["std"] for r in rows], fmt="o", color=SERIES[0], markersize=7,
                ecolor=MUTED, elinewidth=1.2, capsize=3)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(rows):
        ax.annotate(fmt(r["mean"], r["key"]), (r["mean"], i), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_xlabel(f"{rows[0]['key']} ({'higher' if rows[0]['higher_is_better'] else 'lower'} is better; whiskers = std over seeds)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
