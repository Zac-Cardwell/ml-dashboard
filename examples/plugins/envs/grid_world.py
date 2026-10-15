"""Example: a custom RL environment — a plain gymnasium.Env.

Upload this on the Library page as an *environment*. Constructor arguments with defaults
become form fields. Train it with PPO / A2C / DQN (discrete actions).
"""
import gymnasium as gym
import numpy as np


class GridWorld(gym.Env):
    """Reach the goal corner of an N x N grid; -0.01 per step, +1 at the goal"""

    def __init__(self, size: int = 6, max_steps: int = 100):
        self.size, self.max_steps = size, max_steps
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def _obs(self):
        return (self.pos / (self.size - 1)).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0])
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        move = [(0, 1), (1, 0), (0, -1), (-1, 0)][int(action)]
        self.pos = np.clip(self.pos + move, 0, self.size - 1)
        self.t += 1
        done = bool((self.pos == self.size - 1).all())
        reward = 1.0 if done else -0.01
        return self._obs(), reward, done, self.t >= self.max_steps, {}
