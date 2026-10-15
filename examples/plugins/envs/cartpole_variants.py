"""Example: an OpenAI Gym / Gymnasium environment plugin with configurable difficulty.

EnvPlugin.make() builds any gymnasium env and can wrap it. Here CartPole gets optional observation noise,
action noise (random action with probability p) and a harder pole — handy for testing how robust an agent
trained on plain CartPole is (train on the built-in env, then *evaluate* the saved model on this one).
Always forward `render_mode` so videos, the live view and the GUI window work.
"""
import gymnasium as gym
import numpy as np

from mltrain.plugins import EnvPlugin, Param


class NoisyObs(gym.ObservationWrapper):
    def __init__(self, env, std):
        super().__init__(env)
        self.std = std

    def observation(self, obs):
        return (obs + self.np_random.normal(0, self.std, size=obs.shape)).astype(np.float32)


class StickyRandomActions(gym.ActionWrapper):
    def __init__(self, env, p):
        super().__init__(env)
        self.p = p

    def action(self, act):
        return self.action_space.sample() if self.np_random.random() < self.p else act


class CartPoleVariants(EnvPlugin):
    id = "example.cartpole_variants"
    name = "CartPole variants (example)"
    description = "CartPole-v1 with observation noise, random actions and a longer/heavier pole"

    def params(self):
        return [Param("obs_noise_std", "float", 0.05, "Gaussian noise added to observations", 0.0, group="Environment"),
                Param("random_action_prob", "float", 0.05, "Chance an action is replaced by a random one", 0.0, 1.0, group="Environment"),
                Param("pole_length", "float", 0.5, "Half pole length (default 0.5)", 0.05, 5.0, group="Environment"),
                Param("pole_mass", "float", 0.1, "Pole mass (default 0.1)", 0.01, 5.0, group="Environment"),
                Param("max_episode_steps", "int", 500, min=10, group="Environment")]

    def make(self, params, seed, render_mode=None):
        env = gym.make("CartPole-v1", render_mode=render_mode, max_episode_steps=params["max_episode_steps"])
        core = env.unwrapped
        core.length = params["pole_length"]
        core.masspole = params["pole_mass"]
        core.total_mass = core.masspole + core.masscart
        core.polemass_length = core.masspole * core.length
        if params["obs_noise_std"] > 0:
            env = NoisyObs(env, params["obs_noise_std"])
        if params["random_action_prob"] > 0:
            env = StickyRandomActions(env, params["random_action_prob"])
        return env
