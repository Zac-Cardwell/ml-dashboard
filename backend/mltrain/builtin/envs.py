"""Built-in RL environments (gymnasium, Box2D, Atari)."""
from __future__ import annotations

from ..plugins.base import EnvPlugin, Param
from ..plugins.registry import register

COMMON_ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1",
               "LunarLander-v3", "LunarLanderContinuous-v3", "BipedalWalker-v3", "CarRacing-v3"]
ATARI_GAMES = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5", "ALE/Qbert-v5", "ALE/Seaquest-v5",
               "ALE/BeamRider-v5", "ALE/Enduro-v5", "ALE/MsPacman-v5"]


def _make(env_id: str, params: dict, render_mode: str | None, **extra):
    import gymnasium as gym
    if env_id.startswith("ALE/"):
        import ale_py
        gym.register_envs(ale_py)
    kw = dict(params.get("env_kwargs") or {})
    kw.update(extra)
    if params.get("max_episode_steps"):
        kw["max_episode_steps"] = params["max_episode_steps"]
    return gym.make(env_id, render_mode=render_mode, **kw)


ENV_PARAMS = [
    Param("max_episode_steps", "int", 0, "Override the episode time limit (0 = env default)", 0, group="Environment"),
    Param("env_kwargs", "json", {}, "Extra keyword args for gym.make, as JSON", group="Environment"),
]


@register
class GymEnv(EnvPlugin):
    id = "gym"
    name = "Gymnasium (any registered env)"
    description = "Any environment id registered with gymnasium, including ones installed by other packages"

    def params(self):
        return [Param("env_id", "str", "CartPole-v1", "gymnasium env id", suggestions=COMMON_ENVS + ATARI_GAMES,
                      group="Environment")] + ENV_PARAMS

    def make(self, params, seed, render_mode=None):
        return _make(params["env_id"], params, render_mode)


def gym_preset(env_id: str, desc: str, builtin: bool = True, **extra):
    """A preset for one gymnasium env id. builtin=False returns the class unregistered (for optional plugin files)."""
    class _E(EnvPlugin):
        id = f"gym.{env_id}"
        name = env_id
        description = desc

        def params(self):
            return ENV_PARAMS

        def make(self, params, seed, render_mode=None):
            return _make(env_id, params, render_mode, **extra)

    _E.__name__ = f"Gym_{env_id.replace('-', '_')}"
    return register(_E) if builtin else _E


# One default per RL problem type ships with the dashboard (discrete actions, continuous actions; pixels = the Atari
# loader); more presets live in user_plugins/envs/optional/ (git-ignored). "gym" runs any registered id.
gym_preset("CartPole-v1", "Balance a pole on a cart. Discrete actions; solved at 475+")
gym_preset("Pendulum-v1", "Swing up and balance a pendulum. Continuous actions")


@register
class AtariEnv(EnvPlugin):
    id = "atari"
    name = "Atari (ALE)"
    description = "Atari 2600 games with the standard DeepMind preprocessing (84x84 gray, frame skip, frame stack)"

    def params(self):
        return [Param("game", "str", "ALE/Pong-v5", "ALE game id", suggestions=ATARI_GAMES, group="Environment"),
                Param("frame_stack", "int", 4, "Stacked frames given to the policy", 1, 8, group="Environment"),
                Param("terminal_on_life_loss", "bool", True, group="Environment"),
                Param("max_episode_steps", "int", 0, "Override the episode time limit (0 = env default)", 0, group="Environment")]

    def make(self, params, seed, render_mode=None):
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        from stable_baselines3.common.monitor import Monitor
        # AtariWrapper does the frame skipping, so ALE's own frameskip is disabled.
        # Monitor sits inside the wrapper so episode stats cover whole games, not single lives.
        env = _make(params["game"], {"max_episode_steps": params.get("max_episode_steps")}, render_mode, frameskip=1)
        return AtariWrapper(Monitor(env), terminal_on_life_loss=params.get("terminal_on_life_loss", True))

    def wrap_vec(self, venv, params):
        """Applied by RL models to the vectorized env (training, evaluation and video envs alike)."""
        from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
        return VecTransposeImage(VecFrameStack(venv, n_stack=params["frame_stack"]))
