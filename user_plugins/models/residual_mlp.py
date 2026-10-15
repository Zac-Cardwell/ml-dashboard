"""Example: the simplest way to add a PyTorch model — a plain nn.Module.

Upload this on the Library page as a *model*. The dashboard will:
  * train it with the built-in loop (optimizer, lr, epochs, early stopping, ... all in the form)
  * turn every constructor argument WITH a default into a form field (int, float, bool, list[int], str)
  * inject these arguments automatically if your constructor asks for them:
        input_dim, input_shape, in_channels, num_classes, output_dim
"""
import torch.nn as nn


class ResidualMLP(nn.Module):
    """MLP with residual blocks and LayerNorm"""

    # optional: restrict which problems this model is offered for
    TASKS = ("classification", "regression")
    MODALITIES = ("tabular", "image")

    def __init__(self, input_dim, output_dim, width: int = 256, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, width))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
            for _ in range(n_blocks)
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)
