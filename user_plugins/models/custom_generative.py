"""A fully custom generative model: rectified flow (flow matching), which the dashboard doesn't have built in.

Implement `losses()` and `sample()` and the dashboard does the rest (data, GPU, AMP, EMA, checkpoints, snapshots,
FID, seeds). Constructor arguments with defaults become form fields; `in_channels` / `data_shape` are filled
from the dataset.
"""
import torch
import torch.nn as nn

from mltrain.generative.nets import UNet


class RectifiedFlow(nn.Module):
    """Rectified flow / flow matching: learn the velocity field from noise to data, sample with Euler steps.

    Args:
        data_shape: (channels, height, width) of one sample (filled from the dataset)
        base_channels: UNet width
        sample_steps: Euler integration steps when sampling
    """
    MODALITIES = ("image",)

    def __init__(self, data_shape: tuple = (3, 32, 32), base_channels: int = 64, sample_steps: int = 50):
        super().__init__()
        self.shape = tuple(data_shape)
        self.net = UNet(self.shape[0], 2, base_channels, (1, 2, 2), 2, (2,), 0.0)
        self.steps = sample_steps

    def losses(self, x, y=None, generator=None):
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        t = torch.rand(x.shape[0], device=x.device, generator=generator)
        xt = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * x       # straight path noise -> data
        v = self.net(xt, t * 999)                                                # time embedding expects ~[0, 1000)
        return {"loss": ((v - (x - noise)) ** 2).mean()}

    @torch.no_grad()
    def sample(self, n, y=None, generator=None):
        dev = next(self.parameters()).device
        x = torch.randn((n, *self.shape), device=dev, generator=generator)
        for i in range(self.steps):
            t = torch.full((n,), i / self.steps, device=dev)
            x = x + self.net(x, t * 999) / self.steps
        return x.clamp(-1, 1)
