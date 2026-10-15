"""Stable-Baselines3 (and sb3-contrib) agents.

Subclass `SB3Model` (override `make_algo`) for custom agents/policies; see examples/plugins/models/ppo_custom_net.py.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np

from ..plugins.base import ModelPlugin, Param
from ..plugins.registry import register

COMMON = [
    Param("total_timesteps", "int", 100_000, min=1, group="Training"),
    Param("learning_rate", "float", 3e-4, min=1e-8, log=True, group="Training"),
    Param("gamma", "float", 0.99, "Discount factor", 0.0, 1.0, group="Training"),
    Param("n_envs", "int", 1, "Parallel environments (vectorized)", 1, group="Training"),
    Param("policy", "choice", "auto", "auto picks Cnn/MultiInput from the observation space",
          choices=["auto", "MlpPolicy", "CnnPolicy", "MultiInputPolicy"], group="Model"),
    Param("net_arch", "int_list", [64, 64], "Hidden units per layer (policy & value nets)", group="Model"),
    Param("activation", "choice", "tanh", choices=["tanh", "relu"], group="Model"),
    Param("policy_kwargs", "json", {}, "Extra policy_kwargs as JSON (merged)", group="Model"),
    Param("eval_freq", "int", 5000, "Evaluate every N timesteps (0 = off)", 0, group="Evaluation"),
    Param("n_eval_episodes", "int", 10, min=1, group="Evaluation"),
    Param("reward_threshold", "float", 0.0, "Stop when eval mean reward reaches this (0 = off)", group="Early stopping"),
    Param("no_improvement_evals", "int", 0, "Stop after N evals without a new best (0 = off)", 0, group="Early stopping"),
]


# ---------------------------------------------------------------- env helpers

def build_vec_env(env_plugin, pp: dict, seed: int, n: int, render_mode: str | None = None):
    """Vectorized env with Monitor. Returns (base DummyVecEnv, possibly wrapped VecEnv).

    SB3 requires one render_mode for all sub-envs; frames are only ever read from env 0.
    """
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def factory(i):
        def f():
            env = env_plugin.create(pp, seed + i, render_mode=render_mode)
            return env if is_wrapped(env, Monitor) else Monitor(env)
        return f

    base = DummyVecEnv([factory(i) for i in range(n)])
    base.seed(seed)
    venv = env_plugin.wrap_vec(base, pp) if hasattr(env_plugin, "wrap_vec") else base
    return base, venv


def record_episode(model, env_plugin, pp: dict, seed: int, path: Path, max_steps: int = 1000, fps: int = 30) -> dict:
    """Run one deterministic episode with rgb_array rendering and write an H.264 mp4."""
    import cv2
    import imageio.v2 as imageio
    pp = {**pp, "terminal_on_life_loss": False}   # Atari: record the whole game, not one life
    base, venv = build_vec_env(env_plugin, pp, seed, 1, render_mode="rgb_array")
    obs = venv.reset()
    state, start = None, np.ones((1,), dtype=bool)
    frames, total, length, ep_info = [], 0.0, 0, None
    try:
        for _ in range(max_steps):
            frame = base.envs[0].render()
            if frame is not None:
                frames.append(frame)
            action, state = model.predict(obs, state=state, episode_start=start, deterministic=True)
            obs, r, done, info = venv.step(action)
            total += float(r[0])
            length += 1
            start = done
            if done[0]:
                ep_info = info[0].get("episode")
                break
    finally:
        venv.close()
    fps = min(fps, getattr(env_plugin, "video_fps", None) or fps)   # e.g. one frame per high-level decision
    if frames:
        h, w = frames[0].shape[:2]
        scale = min(1.0, 640 / w)
        W, H = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) if (f.shape[1], f.shape[0]) != (W, H) else f for f in frames]
        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=6, macro_block_size=1,
                         ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    reward = float(ep_info["r"]) if ep_info else total
    return {"reward": reward, "length": int(ep_info["l"]) if ep_info else length, "frames": len(frames)}


class SB3Model(ModelPlugin):
    framework = "sb3"
    tasks = ("rl",)
    modalities = ("env",)
    uses_gpu = False   # SB3 MLP policies are usually faster on CPU
    supports_init = True
    supports_eval = True
    supports_resume = True
    algo: str = "PPO"
    algo_module: str = "stable_baselines3"
    recurrent: bool = False

    def algo_params(self) -> list[Param]:
        return []

    def params(self):
        return COMMON[:4] + self.algo_params() + COMMON[4:]

    def algo_kwargs(self, p: dict) -> dict:
        return {}

    @property
    def algo_cls(self):
        return getattr(importlib.import_module(self.algo_module), self.algo)

    def make_algo(self, policy, env, p: dict, policy_kwargs: dict, ctx):
        return self.algo_cls(policy, env, learning_rate=p["learning_rate"], gamma=p["gamma"], policy_kwargs=policy_kwargs,
                             seed=ctx.seed, device=ctx.device, verbose=0, **self.algo_kwargs(p))

    def pick_policy(self, p, obs_space) -> str:
        import gymnasium as gym
        policy = p["policy"]
        if policy == "auto":
            policy = ("MultiInputPolicy" if isinstance(obs_space, gym.spaces.Dict)
                      else "CnnPolicy" if len(getattr(obs_space, "shape", ()) or ()) == 3 else "MlpPolicy")
        if self.recurrent and "Lstm" not in policy:
            policy = policy.replace("Policy", "LstmPolicy")
        return policy

    # ------------------------------------------------------------ training
    def train(self, env_plugin, ctx):
        import torch.nn as nn
        from stable_baselines3.common.callbacks import (BaseCallback, CallbackList, EvalCallback,
                                                        StopTrainingOnNoModelImprovement,
                                                        StopTrainingOnRewardThreshold)
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.logger import KVWriter, Logger

        p, pp = ctx.params, ctx.problem_params
        live = bool(ctx.opt("live_preview", True))
        gui = bool(ctx.opt("show_gui", False))
        base, env = build_vec_env(env_plugin, pp, ctx.seed, p["n_envs"], render_mode="rgb_array" if live else None)
        try:
            _, eval_env = build_vec_env(env_plugin, pp, ctx.seed + 10_000, 1, render_mode="human" if gui else None)
        except Exception as e:  # no display available
            ctx.print(f"GUI window unavailable ({e}); evaluating without it")
            gui = False
            _, eval_env = build_vec_env(env_plugin, pp, ctx.seed + 10_000, 1)

        total = p["total_timesteps"]
        ckpt = ctx.checkpoint_dir / "model.zip"
        reset_timesteps = True
        if ctx.resume and ckpt.exists():
            model = self.algo_cls.load(ckpt, env=env, device=ctx.device)
            reset_timesteps = False
            ctx.truncate_after(model.num_timesteps)
            ctx.print(f"resuming from checkpoint at {model.num_timesteps:,} timesteps")
        elif ctx.init_path:
            model = self.algo_cls.load(ctx.init_path / "model.zip", env=env, device=ctx.device,
                                       custom_objects={"learning_rate": p["learning_rate"], "gamma": p["gamma"]})
            model.learning_rate = p["learning_rate"]
            model._setup_lr_schedule()
            ctx.print(f"fine-tuning from {ctx.init_path} (trained {model.num_timesteps:,} timesteps before)")
        else:
            pk = {"net_arch": list(p["net_arch"]), "activation_fn": nn.Tanh if p["activation"] == "tanh" else nn.ReLU}
            pk.update(p.get("policy_kwargs") or {})
            model = self.make_algo(self.pick_policy(p, env.observation_space), env, p, pk, ctx)
        ctx.print(f"{self.algo} on {env_plugin.name}; obs={env.observation_space}, act={env.action_space}")
        ctx.info.update(obs_space=str(env.observation_space), action_space=str(env.action_space))

        class CtxWriter(KVWriter):
            def write(self, key_values, key_excluded, step=0):
                ctx.log(step, {k: v for k, v in key_values.items()
                               if isinstance(v, (int, float)) and not k.startswith("time/")})

            def close(self):
                pass

        model.set_logger(Logger(folder=None, output_formats=[CtxWriter()]))
        snapshot_due = ctx.snapshot_schedule(total)
        record = ctx.snapshots and bool(ctx.opt("record_video", True))
        ckpt_every = max(1000, total // 20)
        plugin = self

        class Monitoring(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last_ckpt = model.num_timesteps
                self.last_frame = 0.0

            def _on_step(self):
                n = self.num_timesteps
                for info in self.locals.get("infos", []):
                    ep = info.get("episode")
                    if ep:
                        ctx.log(n, {"episode/reward": ep["r"], "episode/length": ep["l"]})
                if live and time.time() - self.last_frame > ctx.LIVE_FRAME_INTERVAL:
                    self.last_frame = time.time()
                    try:
                        ctx.live_frame(base.envs[0].render())
                    except Exception:
                        pass
                if self.n_calls % 200 == 0:
                    ctx.set_progress(n / total, f"{n:,}/{total:,} steps")
                if snapshot_due(n) and record:
                    plugin.snapshot_video(self.model, env_plugin, pp, ctx, n)
                if ctx.opt("keep_checkpoint", True) and n - self.last_ckpt >= ckpt_every:
                    self.model.save(ckpt)
                    self.last_ckpt = n
                return not ctx.should_stop()

        callbacks = [Monitoring()]
        eval_cb = None
        if p["eval_freq"] > 0:
            eval_cb = EvalCallback(
                eval_env, n_eval_episodes=p["n_eval_episodes"], eval_freq=max(1, p["eval_freq"] // p["n_envs"]),
                best_model_save_path=str(ctx.checkpoint_dir), deterministic=True, verbose=0, render=gui,
                callback_on_new_best=StopTrainingOnRewardThreshold(p["reward_threshold"], verbose=1) if p["reward_threshold"] else None,
                callback_after_eval=StopTrainingOnNoModelImprovement(p["no_improvement_evals"], verbose=1) if p["no_improvement_evals"] else None,
            )
            callbacks.append(eval_cb)

        t0 = time.time()
        remaining = total - model.num_timesteps if not reset_timesteps else total
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=CallbackList(callbacks), reset_num_timesteps=reset_timesteps)
        train_time = time.time() - t0
        trained = int(model.num_timesteps)
        if ctx.should_stop():   # keep resumable state
            if ctx.opt("keep_checkpoint", True):
                model.save(ckpt)

        best_path = ctx.checkpoint_dir / "best_model.zip"
        extra = {}
        keep = ctx.keep
        if best_path.exists() and not ctx.should_stop() and keep != "final":
            if keep == "both":
                fr, fs = evaluate_policy(model, eval_env, n_eval_episodes=max(p["n_eval_episodes"], 10), deterministic=True)
                extra = {"final_model/final/mean_reward": float(fr), "final_model/final/std_reward": float(fs)}
                model.save(ctx.final_path("model.zip"))
            model = self.algo_cls.load(best_path, env=eval_env, device=ctx.device)
        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=max(p["n_eval_episodes"], 10), deterministic=True)
        if record and not ctx.should_stop() and getattr(ctx, "_last_video_step", None) != trained:
            self.snapshot_video(model, env_plugin, pp, ctx, trained, name="final")
        if ctx.save_model:
            model.save(ctx.model_path("model.zip"))
        env.close()
        eval_env.close()
        return {**extra, "final/mean_reward": float(mean_r), "final/std_reward": float(std_r),
                "best_eval_reward": float(eval_cb.best_mean_reward) if eval_cb and eval_cb.best_mean_reward > -float("inf") else None,
                "timesteps_trained": trained, "train_time_s": train_time}

    def snapshot_video(self, model, env_plugin, pp, ctx, step: int, name: str = "episode"):
        path = ctx.media_path(f"{name}_{step}.mp4")
        try:
            res = record_episode(model, env_plugin, pp, ctx.seed + 20_000 + step, path,
                                 int(ctx.opt("video_max_steps", 1000)), int(ctx.opt("video_fps", 30)))
        except Exception as e:
            ctx.print(f"video recording failed: {e}")
            return
        if res["frames"]:
            ctx._last_video_step = step
            ctx.log_media(step, "episode", "video", data={"reward": res["reward"], "length": res["length"], "label": name},
                          file=path)
            ctx.log(step, {"video/episode_reward": res["reward"]})

    # ------------------------------------------------------------ evaluation only
    def evaluate(self, env_plugin, ctx):
        from stable_baselines3.common.evaluation import evaluate_policy
        pp = ctx.problem_params
        gui = bool(ctx.opt("show_gui", False))
        try:
            _, venv = build_vec_env(env_plugin, pp, ctx.seed, 1, render_mode="human" if gui else None)
        except Exception as e:
            ctx.print(f"GUI window unavailable ({e})")
            _, venv = build_vec_env(env_plugin, pp, ctx.seed, 1)
        model = self.algo_cls.load(ctx.init_path / "model.zip", env=venv, device=ctx.device)
        n = int(ctx.opt("n_eval_episodes", 20))
        rewards, lengths = [], []
        for i in range(n):
            r, l = evaluate_policy(model, venv, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
            rewards.append(r[0]); lengths.append(l[0])
            ctx.log(i + 1, {"eval/episode_reward": r[0], "eval/episode_length": l[0]})
            ctx.set_progress((i + 1) / n, f"episode {i + 1}/{n}")
            if ctx.should_stop():
                break
        venv.close()
        if ctx.snapshots and ctx.opt("record_video", True):
            for k in range(min(3, n)):
                self.snapshot_video(model, env_plugin, pp, ctx, k + 1, name=f"eval{k + 1}")
        return {"final/mean_reward": float(np.mean(rewards)), "final/std_reward": float(np.std(rewards)),
                "final/min_reward": float(np.min(rewards)), "final/max_reward": float(np.max(rewards)),
                "final/mean_length": float(np.mean(lengths)), "episodes": len(rewards)}


# ---------------------------------------------------------------- algorithms

PPO_PARAMS = [Param("n_steps", "int", 2048, "Rollout length per env", 1, group="Training"),
              Param("batch_size", "int", 64, min=1, group="Training"),
              Param("n_epochs", "int", 10, min=1, group="Training"),
              Param("gae_lambda", "float", 0.95, min=0.0, max=1.0, group="Training"),
              Param("clip_range", "float", 0.2, min=0.0, group="Training"),
              Param("ent_coef", "float", 0.0, min=0.0, group="Training")]


@register
class PPO(SB3Model):
    id = "sb3.ppo"
    name = "PPO"
    description = "Proximal Policy Optimization. Discrete or continuous actions"
    algo = "PPO"

    def algo_params(self):
        return PPO_PARAMS

    def algo_kwargs(self, p):
        return {k: p[k] for k in ("n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range", "ent_coef")}


@register
class RecurrentPPO(PPO):
    id = "sb3.recurrent_ppo"
    name = "Recurrent PPO (LSTM)"
    description = "PPO with an LSTM policy (sb3-contrib) for partially observable tasks"
    algo = "RecurrentPPO"
    algo_module = "sb3_contrib"
    recurrent = True

    def params(self):
        return super().params() + [Param("lstm_hidden_size", "int", 128, min=8, group="Model")]

    def make_algo(self, policy, env, p, policy_kwargs, ctx):
        policy_kwargs = {**policy_kwargs, "lstm_hidden_size": p["lstm_hidden_size"]}
        return super().make_algo(policy, env, p, policy_kwargs, ctx)


@register
class TRPO(SB3Model):
    id = "sb3.trpo"
    name = "TRPO"
    description = "Trust Region Policy Optimization (sb3-contrib). Discrete or continuous actions"
    algo = "TRPO"
    algo_module = "sb3_contrib"

    def algo_params(self):
        return [Param("n_steps", "int", 2048, min=1, group="Training"), Param("batch_size", "int", 128, min=1, group="Training"),
                Param("gae_lambda", "float", 0.95, min=0.0, max=1.0, group="Training"),
                Param("target_kl", "float", 0.01, min=0.0, group="Training")]

    def algo_kwargs(self, p):
        return {k: p[k] for k in ("n_steps", "batch_size", "gae_lambda", "target_kl")}


@register
class A2C(SB3Model):
    id = "sb3.a2c"
    name = "A2C"
    description = "Advantage Actor-Critic. Discrete or continuous actions"
    algo = "A2C"

    def algo_params(self):
        return [Param("n_steps", "int", 5, min=1, group="Training"),
                Param("gae_lambda", "float", 1.0, min=0.0, max=1.0, group="Training"),
                Param("ent_coef", "float", 0.0, min=0.0, group="Training")]

    def algo_kwargs(self, p):
        return {k: p[k] for k in ("n_steps", "gae_lambda", "ent_coef")}


DQN_PARAMS = [Param("buffer_size", "int", 100_000, min=1, group="Training"),
              Param("learning_starts", "int", 1000, min=0, group="Training"),
              Param("batch_size", "int", 32, min=1, group="Training"),
              Param("train_freq", "int", 4, min=1, group="Training"),
              Param("target_update_interval", "int", 1000, min=1, group="Training"),
              Param("exploration_fraction", "float", 0.1, min=0.0, max=1.0, group="Training"),
              Param("exploration_final_eps", "float", 0.05, min=0.0, max=1.0, group="Training")]


@register
class DQN(SB3Model):
    id = "sb3.dqn"
    name = "DQN"
    description = "Deep Q-Network. Discrete actions only"
    algo = "DQN"

    def algo_params(self):
        return DQN_PARAMS

    def algo_kwargs(self, p):
        return {q.name: p[q.name] for q in DQN_PARAMS}


@register
class QRDQN(DQN):
    id = "sb3.qrdqn"
    name = "QR-DQN"
    description = "Quantile Regression DQN (sb3-contrib). Discrete actions only"
    algo = "QRDQN"
    algo_module = "sb3_contrib"


class _OffPolicyContinuous(SB3Model):
    def algo_params(self):
        return [Param("buffer_size", "int", 300_000, min=1, group="Training"),
                Param("learning_starts", "int", 100, min=0, group="Training"),
                Param("batch_size", "int", 256, min=1, group="Training"),
                Param("tau", "float", 0.005, min=0.0, max=1.0, group="Training"),
                Param("train_freq", "int", 1, min=1, group="Training")]

    def algo_kwargs(self, p):
        return {k: p[k] for k in ("buffer_size", "learning_starts", "batch_size", "tau", "train_freq")}


@register
class SAC(_OffPolicyContinuous):
    id = "sb3.sac"
    name = "SAC"
    description = "Soft Actor-Critic. Continuous actions only"
    algo = "SAC"

    def params(self):
        from dataclasses import replace
        defaults = {"activation": "relu", "net_arch": [256, 256]}
        return [replace(q, default=defaults[q.name]) if q.name in defaults else q for q in super().params()]


@register
class TD3(_OffPolicyContinuous):
    id = "sb3.td3"
    name = "TD3"
    description = "Twin Delayed DDPG. Continuous actions only"
    algo = "TD3"

    def algo_params(self):
        return super().algo_params() + [Param("action_noise_sigma", "float", 0.1, min=0.0, group="Training")]

    def make_algo(self, policy, env, p, policy_kwargs, ctx):
        from stable_baselines3.common.noise import NormalActionNoise
        n = env.action_space.shape[-1]
        noise = NormalActionNoise(np.zeros(n), p["action_noise_sigma"] * np.ones(n))
        return self.algo_cls(policy, env, learning_rate=p["learning_rate"], gamma=p["gamma"], policy_kwargs=policy_kwargs,
                             action_noise=noise, seed=ctx.seed, device=ctx.device, verbose=0, **self.algo_kwargs(p))
