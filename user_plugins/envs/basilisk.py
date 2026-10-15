"""Basilisk spacecraft scenarios (bsk_rl) from the Basilisk sim-server in /mnt/Linux_Data_2/Projects/Basilisk_env.

The sim runs in its Docker container (`docker compose up -d` there; API on http://localhost:8000). This file turns
each scenario into a dashboard environment that SB3 can train on:

- constellation scenarios: Tuple(n x Box) observations are concatenated into one Box, and Tuple(n x Discrete)
  actions become MultiDiscrete (one choice per satellite) — use PPO / A2C / TRPO / recurrent PPO
- constellation_competition (multi-agent): one policy controls the chosen provider's satellites jointly and is
  rewarded with that provider's reward; the other provider's satellites act randomly, idle (charge) or are also
  controlled ("all")
- render_mode="rgb_array" draws a mission-control frame from the server's telemetry endpoint (battery, wheel
  momentum, storage, eclipse, faults, last action, reward), so live view and episode videos work
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mltrain.plugins.base import EnvPlugin, Param

CLIENT_DIR = os.environ.get("BASILISK_CLIENT", "/mnt/Linux_Data_2/Projects/Basilisk_env/client")
DEFAULT_URL = os.environ.get("BASILISK_URL", "http://localhost:8000")


def _client():
    try:
        import bsk_client
    except ImportError:
        if CLIENT_DIR not in sys.path:
            sys.path.insert(0, CLIENT_DIR)
        import bsk_client
    return bsk_client


@lru_cache(maxsize=32)
def _describe(url: str, scenario: str, overrides_json: str) -> dict:
    import json
    try:
        return _client().describe_scenario(url, scenario, json.loads(overrides_json))
    except Exception:
        return {}


class _BasiliskGymEnv(gym.Env):
    """gymnasium.Env adapter over the sim-server client (flattened spaces + telemetry rendering)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, url: str, scenario: str, overrides: dict, render_mode: str | None = None,
                 max_episode_steps: int = 0, control: str = "", opponents: str = "random"):
        import json
        super().__init__()
        bc = _client()
        self.url, self.scenario, self.render_mode = url.rstrip("/"), scenario, render_mode
        self.max_steps = max_episode_steps
        self.desc = _describe(self.url, scenario, json.dumps(overrides, sort_keys=True))
        self._telemetry_session = None
        try:
            self.env = bc.RemoteBasiliskEnv(self.url, scenario, overrides)
            self.multi = False
        except ValueError:                      # multi-agent scenario
            self.env = bc.RemoteMultiAgentEnv(self.url, scenario, overrides)
            self.multi = True
        except Exception as e:
            raise RuntimeError(f"Cannot reach the Basilisk sim-server at {self.url} ({e}). "
                               "Start it with `docker compose up -d` in the Basilisk_env folder.") from e

        if self.multi:
            self.agents_all = list(self.env.possible_agents)
            ctl = (control or "").strip()
            if ctl in ("", "all"):
                self.controlled = list(self.agents_all)
            else:
                self.controlled = [a for a in self.agents_all if a.rsplit("-", 1)[0] == ctl]
                if not self.controlled:
                    providers = sorted({a.rsplit("-", 1)[0] for a in self.agents_all})
                    raise ValueError(f"control: unknown provider '{ctl}'; choose one of {providers} or 'all'")
            self.opponents = opponents
            sub_obs = [self.env.observation_space(a) for a in self.controlled]
            sub_act = [self.env.action_space(a) for a in self.controlled]
            self.sat_names = list(self.agents_all)
        else:
            obs_s, act_s = self.env.observation_space, self.env.action_space
            sub_obs = list(obs_s.spaces) if isinstance(obs_s, spaces.Tuple) else [obs_s]
            sub_act = list(act_s.spaces) if isinstance(act_s, spaces.Tuple) else [act_s]
            self.sat_names = None
        self._tuple_act = self.multi or isinstance(getattr(self.env, "action_space", None), spaces.Tuple)
        self._sizes = [int(np.prod(s.shape)) for s in sub_obs]
        n = sum(self._sizes)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        if len(sub_act) == 1 and not self._tuple_act:
            self.action_space = sub_act[0]
        else:
            self.action_space = spaces.MultiDiscrete([s.n for s in sub_act])
        self._labels = self._action_labels()
        self._steps = 0
        self._ret = 0.0
        self._last_r = 0.0
        self._last_action = None

    # ---------------------------------------------------------------- gym API
    def _flat(self, obs) -> np.ndarray:
        if self.multi:
            parts = [np.asarray(obs[a], np.float32).ravel() if a in obs else np.zeros(sz, np.float32)
                     for a, sz in zip(self.controlled, self._sizes)]
        elif isinstance(obs, tuple):
            parts = [np.asarray(o, np.float32).ravel() for o in obs]
        else:
            parts = [np.asarray(obs, np.float32).ravel()]
        return np.nan_to_num(np.concatenate(parts), posinf=1e6, neginf=-1e6)

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=None if seed is None else int(seed))
        self._steps, self._ret, self._last_r, self._last_action = 0, 0.0, 0.0, None
        return self._flat(obs), {}

    def step(self, action):
        self._steps += 1
        self._last_action = np.atleast_1d(np.asarray(action)).astype(int).tolist()
        if self.multi:
            acts = {}
            mine = dict(zip(self.controlled, self._last_action))
            for a in self.env.agents:
                if a in mine:
                    acts[a] = int(mine[a])
                elif self.opponents == "idle":
                    acts[a] = self._charge_index(a)
                else:
                    acts[a] = int(self.env.action_space(a).sample())
            obs, rew, term, trunc, _ = self.env.step(acts)
            r = float(sum(rew.get(a, 0.0) for a in self.controlled))
            alive = [a for a in self.controlled if a in obs and not (term.get(a) or trunc.get(a))]
            terminated = not alive and all(term.get(a, True) for a in self.controlled)
            truncated = not alive and not terminated
            info = {}
        else:
            act = [int(a) for a in self._last_action] if self._tuple_act else int(self._last_action[0])
            obs, r, terminated, truncated, _ = self.env.step(act)
            r = float(r)
            info = {}
        if self.max_steps and self._steps >= self.max_steps and not terminated:
            truncated = True
        self._ret += r
        self._last_r = r
        return self._flat(obs), r, bool(terminated), bool(truncated), info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- helpers
    def _charge_index(self, agent: str) -> int:
        for i, name in (self._labels.get(agent) or {}).items():
            if "charge" in name:
                return int(i)
        return 0

    def _action_labels(self) -> dict:
        """{satellite: {index: label}} from the server's describe() output (best effort)."""
        a = self.desc.get("action_space") or {}
        out = {}
        if self.multi:
            for agent in self.agents_all:
                per = (a.get("per_agent") or {}).get(agent) or a
                labels = per.get("actions") or per.get("per_satellite_actions") or {}
                out[agent] = {int(k): str(v).replace("action_", "") for k, v in labels.items()}
        else:
            labels = a.get("actions") or a.get("per_satellite_actions") or {}
            out["*"] = {int(k): str(v).replace("action_", "") for k, v in labels.items()}
        return out

    def _label(self, sat_i: int, sat_name: str) -> str:
        if self._last_action is None:
            return "—"
        if self.multi:
            if sat_name not in self.controlled:
                return {"idle": "charge (idle)", "random": "random"}.get(self.opponents, "")
            idx = self.controlled.index(sat_name)
            v = self._last_action[idx] if idx < len(self._last_action) else None
            return (self._labels.get(sat_name) or {}).get(v, str(v))
        v = self._last_action[sat_i] if sat_i < len(self._last_action) else None
        return self._labels.get("*", {}).get(v, str(v)) if v is not None else "—"

    def _telemetry(self) -> dict:
        import requests
        if self._telemetry_session is None:
            self._telemetry_session = requests.Session()
        try:
            return self._telemetry_session.get(f"{self.url}/envs/{self.env.env_id}/telemetry", timeout=10).json()
        except Exception:
            return {}

    # ---------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return draw_frame(self.scenario, self._telemetry(), self._steps, self._ret, self._last_r, self._label,
                          self.controlled if self.multi else None)


# ---------------------------------------------------------------------------- frame drawing
BG, PANEL, INK, MUTED, LINE = (247, 246, 243), (255, 255, 255), (31, 35, 40), (110, 119, 129), (225, 228, 232)
GOOD, WARN, CRIT, NEUTRAL, ACCENT = (27, 175, 122), (237, 161, 0), (227, 73, 72), (89, 99, 110), (42, 120, 214)


def _status_color(frac: float | None, higher_is_good: bool) -> tuple:
    if frac is None:
        return NEUTRAL
    v = frac if higher_is_good else 1 - frac
    return GOOD if v >= 0.5 else WARN if v >= 0.2 else CRIT


def draw_frame(scenario: str, tel: dict, step: int, ret: float, last_r: float, label_fn, controlled) -> np.ndarray:
    import cv2
    sats = tel.get("satellites") or []
    W, row_h, head = 860, 46, 78
    H = head + max(1, len(sats)) * row_h + 34
    img = np.full((H, W, 3), BG, np.uint8)
    put = lambda t, xy, s=0.5, c=INK, th=1: cv2.putText(img, t, xy, cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)
    t = tel.get("sim_time") or 0.0
    put(f"Basilisk - {scenario}", (16, 28), 0.65, INK, 2)
    put(f"sim time {t / 60:6.1f} min    step {step}    episode reward {ret:8.3f}    last step {last_r:+.3f}", (16, 54), 0.48, MUTED)
    cols = [("battery", 220, True, "battery_charge_fraction"), ("wheel momentum", 390, False, "wheel_speeds_fraction"),
            ("storage", 560, None, "storage_level_fraction")]
    for name, x, _, _ in cols:
        put(name, (x, head - 6), 0.4, MUTED)
    put("last action", (725, head - 6), 0.4, MUTED)
    for i, s in enumerate(sats):
        y = head + i * row_h
        cv2.rectangle(img, (8, y + 4), (W - 8, y + row_h - 4), PANEL, -1)
        cv2.rectangle(img, (8, y + 4), (W - 8, y + row_h - 4), LINE, 1)
        alive = s.get("alive", True)
        dot = CRIT if not alive else WARN if s.get("fault_active") else GOOD
        cv2.circle(img, (24, y + row_h // 2), 6, dot, -1)
        name = s.get("name", f"sat-{i}")
        put(name, (38, y + row_h // 2 + 5), 0.5, INK if alive else MUTED, 1)
        tags = []
        if not alive:
            tags.append("FAILED")
        elif s.get("fault_active"):
            tags.append("fault")
        if s.get("in_eclipse"):
            tags.append("eclipse")
        if controlled is not None and name not in controlled:
            tags.append("opponent")
        if tags:
            put(" ".join(tags), (120, y + row_h // 2 + 5), 0.38, MUTED)
        for _, x, good, key in cols:
            v = s.get(key)
            v = None if v is None else float(np.clip(v, 0, 1))
            cv2.rectangle(img, (x, y + 17), (x + 120, y + 29), LINE, -1)
            if v is not None:
                color = ACCENT if good is None else _status_color(v, good)
                cv2.rectangle(img, (x, y + 17), (x + max(1, int(120 * v)), y + 29), color, -1)
                put(f"{v * 100:.0f}%", (x + 124, y + 28), 0.36, MUTED)
        put(str(label_fn(i, name))[:14], (725, y + row_h // 2 + 5), 0.42, INK)
    if not sats:
        put("no telemetry (episode not started or server unreachable)", (16, head + 26), 0.5, MUTED)
    put("bars: battery (green = charged) | wheel momentum (red = near saturation) | storage fill", (16, H - 12), 0.38, MUTED)
    return img


# ---------------------------------------------------------------------------- dashboard plugins
COMMON = [
    Param("server_url", "str", DEFAULT_URL, "Basilisk sim-server URL", group="Server"),
    Param("time_limit", "float", 5700.0, "Episode length in simulated seconds (~1 orbit = 5700)", 60.0, group="Episode"),
    Param("max_episode_steps", "int", 0, "Also cap decisions per episode (0 = no cap)", 0, group="Episode"),
    Param("failure_penalty", "float", -1.0, "Reward when a satellite fails (reward shaping)", group="Reward"),
    Param("anomalies", "json", {}, 'Fault injection, e.g. {"rw_fault": {"prob": 0.2, "severity": 0.7}, '
                                  '"battery_degradation": {"prob": 0.3, "severity": 0.4}}', group="Anomalies"),
    Param("overrides", "json", {}, "Any other scenario overrides / Basilisk satellite args, as JSON", group="Advanced"),
]

SCENARIOS = {
    "single_sat_ops": ("Single satellite ops", "One satellite: image upcoming targets, charge, desaturate wheels; "
                       "survive a possible reaction-wheel fault. Discrete(12) actions", [
        Param("n_targets", "int", 500, "Randomly placed imaging targets", 1, group="Scenario")]),
    "constellation_sched": ("Constellation scheduling", "Central scheduler for a Walker-Delta constellation: image / "
                            "charge / desaturate / downlink per satellite under cloud-cover uncertainty. MultiDiscrete", [
        Param("n_sats", "int", 4, "Satellites", 1, 24, group="Scenario"),
        Param("n_planes", "int", 2, "Orbital planes", 1, 12, group="Scenario"),
        Param("altitude", "float", 500.0, "Orbit altitude (km)", 200.0, 2000.0, group="Scenario"),
        Param("inc", "float", 55.0, "Inclination (deg)", 0.0, 180.0, group="Scenario")]),
    "storage_downlink": ("Storage & downlink", "Constellation where onboard storage is the binding constraint: image "
                         "only what you can downlink at ground-station passes. MultiDiscrete", [
        Param("n_sats", "int", 3, "Satellites", 1, 24, group="Scenario"),
        Param("buffer_images", "float", 3, "Storage capacity in images (the difficulty knob)", 1, group="Scenario")]),
    "rf_cued_imaging": ("RF-cued imaging", "RF-monitor satellites detect ground events; imaging satellites must "
                        "catch them within a window. MultiDiscrete", [
        Param("mode", "choice", "joint", "joint, or a curriculum phase", choices=["joint", "imaging_only", "detection_only"],
              group="Scenario"),
        Param("n_rf_sats", "int", 2, "RF-monitor satellites", 0, 12, group="Scenario"),
        Param("n_imaging_sats", "int", 2, "Imaging satellites", 0, 12, group="Scenario"),
        Param("catch_window", "float", 1200.0, "Seconds a detection stays catchable", 1.0, group="Scenario"),
        Param("detection_credit_fraction", "float", 0.2, "Reward for a detection relative to a catch", 0.0, 1.0,
              group="Reward")]),
    "constellation_competition": ("Constellation competition", "Two providers' constellations race for the same "
                                  "targets (multi-agent). One policy controls the chosen provider jointly", [
        Param("control", "str", "ProviderA", "Provider the policy controls (its reward is optimized), or 'all'",
              suggestions=["ProviderA", "ProviderB", "all"], group="Agents"),
        Param("opponents", "choice", "random", "What uncontrolled satellites do", choices=["random", "idle"], group="Agents"),
        Param("constellations", "json", [], "Provider list (empty = server default: ProviderA vs ProviderB, 3 sats each)",
              group="Scenario")]),
}
NOT_OVERRIDES = {"server_url", "max_episode_steps", "overrides", "control", "opponents"}


def _plugin(scenario: str, title: str, desc: str, extra: list[Param]):
    keys = [p.name for p in extra]

    class _B(EnvPlugin):
        id = f"basilisk.{scenario}"
        name = f"Basilisk · {title}"
        description = desc + " (needs the Basilisk sim-server running)"
        video_fps = 4      # one frame per decision; a 30 fps video of a 40-step episode would last a second
        export_note = ("Needs the Basilisk sim-server (Docker) from the Basilisk_env project running at the "
                       "`server_url` in the config, and its client: `pip install -e <Basilisk_env>/client`.")

        def params(self):
            ps = extra + COMMON
            if scenario == "storage_downlink":     # no time_limit / failure_penalty keys documented for these
                ps = [p for p in ps if p.name not in ("time_limit", "failure_penalty")]
            if scenario in ("rf_cued_imaging", "constellation_competition"):
                ps = [p for p in ps if p.name not in ("time_limit", "failure_penalty")]
            return ps

        def make(self, params, seed, render_mode=None):
            ov = {}
            for k, v in params.items():
                if k in NOT_OVERRIDES or v in (None, "", [], {}):
                    continue
                if k in keys or k in ("time_limit", "failure_penalty", "anomalies"):
                    ov[k] = v
            ov.update(params.get("overrides") or {})
            return _BasiliskGymEnv(params.get("server_url") or DEFAULT_URL, scenario, ov, render_mode,
                                  int(params.get("max_episode_steps") or 0), params.get("control", ""),
                                  params.get("opponents", "random"))

    _B.__name__ = "Basilisk_" + scenario
    return _B


for _s, (_t, _d, _x) in SCENARIOS.items():
    globals()[f"Basilisk_{_s}"] = _plugin(_s, _t, _d, _x)
