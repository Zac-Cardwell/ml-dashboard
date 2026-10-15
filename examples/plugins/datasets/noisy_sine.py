"""Example: a custom dataset plugin (full control).

Subclass DatasetPlugin, declare params, and return a DataBundle from load().
For tabular data the helper `tabular_bundle` handles the train/val/test split and
standardization using the standard split parameters.
"""
import numpy as np

from mltrain.builtin.datasets import TABULAR_PARAMS, tabular_bundle
from mltrain.plugins import DatasetPlugin, Param


class NoisySine(DatasetPlugin):
    id = "example.noisy_sine"
    name = "Noisy sine (example)"
    description = "y = sin(freq * x) + noise, 1 feature regression"
    task = "regression"
    modality = "tabular"

    def params(self):
        return [Param("n_samples", "int", 2000, min=10, group="Generator"),
                Param("freq", "float", 3.0, group="Generator"),
                Param("noise", "float", 0.1, min=0.0, group="Generator")] + TABULAR_PARAMS

    def load(self, params, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-np.pi, np.pi, size=(params["n_samples"], 1))
        y = np.sin(params["freq"] * x[:, 0]) + rng.normal(0, params["noise"], params["n_samples"])
        return tabular_bundle(x, y, params, seed, "regression", feature_names=["x"])
