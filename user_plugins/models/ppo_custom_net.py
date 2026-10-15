"""Example: a custom Stable-Baselines3 PPO agent as a plugin.

Subclass SB3Model to reuse everything the built-in RL models get (live preview, videos, eval, early stopping,
saving, fine-tuning, resume) and override only what you need — here the policy network: a residual MLP
feature extractor with LayerNorm, with its width/depth exposed as form fields.
"""
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mltrain.builtin.models_sb3 import PPO_PARAMS, SB3Model
from mltrain.plugins import Param


class ResidualExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, width: int = 128, depth: int = 2):
        super().__init__(observation_space, features_dim=width)
        self.inp = nn.Linear(int(torch.tensor(observation_space.shape).prod()), width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU()) for _ in range(depth))

    def forward(self, obs):
        h = self.inp(obs.flatten(1))
        for b in self.blocks:
            h = h + b(h)
        return h


class PPOResidual(SB3Model):
    id = "example.ppo_residual"
    name = "PPO · residual extractor (example)"
    description = "PPO whose policy/value nets share a residual LayerNorm MLP feature extractor"
    algo = "PPO"

    def algo_params(self):
        return PPO_PARAMS + [Param("extractor_width", "int", 128, min=8, group="Model"),
                             Param("extractor_depth", "int", 2, min=0, max=8, group="Model")]

    def algo_kwargs(self, p):
        return {k: p[k] for k in ("n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range", "ent_coef")}

    def make_algo(self, policy, env, p, policy_kwargs, ctx):
        policy_kwargs = {**policy_kwargs,
                         "features_extractor_class": ResidualExtractor,
                         "features_extractor_kwargs": {"width": p["extractor_width"], "depth": p["extractor_depth"]},
                         "share_features_extractor": True}
        return super().make_algo("MlpPolicy", env, p, policy_kwargs, ctx)
