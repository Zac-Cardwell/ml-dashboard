"""Session-level options that apply to every model (saving, snapshots, videos, ...).

They are declared like plugin parameters so the dashboard renders them as a form.
"""
from __future__ import annotations

from ..plugins.base import Param

COMMON = [
    Param("save_model", "bool", True, "Save the trained model at the end (needed to fine-tune or evaluate it later)", group="Saving"),
    Param("keep", "choice", "both", "Which weights to save: the best checkpoint (by validation), the final one, or both "
          "(neural nets & RL; tree models keep one)", choices=["best", "final", "both"], group="Saving"),
    Param("keep_checkpoint", "bool", True, "Keep a resumable checkpoint if the run is stopped or fails (deleted once it completes)",
          group="Saving"),
    Param("snapshots", "bool", True, "Record learning snapshots during training (what the model predicts / generates / does)",
          group="Learning snapshots"),
    Param("snapshot_count", "int", 10, "How many snapshots to spread across training", 1, 200, group="Learning snapshots"),
]

RL = [
    Param("record_video", "bool", True, "Record an evaluation episode as video at each snapshot and at the end", group="Learning snapshots"),
    Param("video_max_steps", "int", 1000, "Max environment steps per recorded episode", 10, group="Learning snapshots"),
    Param("video_fps", "int", 30, min=1, max=120, group="Learning snapshots"),
    Param("live_preview", "bool", True, "Stream frames of the training environment to the dashboard while it trains", group="Live view"),
    Param("show_gui", "bool", False, "Open a render window on the host machine's desktop during evaluation episodes", group="Live view"),
]

EVAL = [
    Param("n_eval_episodes", "int", 20, "RL only: episodes to evaluate", 1, group="Evaluation"),
    Param("eval_splits", "choice", "val+test", "Supervised only: which splits to evaluate", choices=["test", "val+test", "train+val+test"],
          group="Evaluation"),
]


def options_for(task: str, mode: str = "train") -> list[Param]:
    ps = list(COMMON)
    if task == "rl":
        ps += RL
    if mode == "evaluate":
        ps = [p for p in ps if p.name not in ("save_model", "keep", "keep_checkpoint", "snapshot_count")] + EVAL
    return ps
