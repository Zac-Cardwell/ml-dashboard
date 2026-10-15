"""Transfer learning: a classifier / regressor on top of an interchangeable encoder.

The encoder can be the built-in conv encoder, your network, a part of a model trained in this dashboard (e.g. a VAE's
encoder or a GAN's discriminator — self-supervised features), or an ImageNet backbone. Freeze it for a linear probe.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..generative.components import Slot, WithHead, check_shape, slot_params
from ..generative.nets import ConvEncoder
from ..plugins.base import Param
from ..plugins.registry import register
from .models_torch import TorchModel


class ToPM1(nn.Module):
    """Dataset-normalized images -> [-1, 1] (the range every generative component expects)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x):
        return (x * self.std + self.mean) * 2 - 1


def _enc_builtin(p, info):
    return ConvEncoder(info["data_shape"], p["features"], p["base_channels"], 256)


ENC = Slot("encoder", "encoder", "encoder", builtin=_enc_builtin, freeze_default=True, builtin_label="strided conv (built-in, trained from scratch)",
           builtin_params=[Param("base_channels", "int", 32, min=4, max=512), Param("features", "int", 256, min=8)],
           help="Feature extractor; a VAE encoder or GAN discriminator trained here, your network, or an ImageNet backbone")


@register
class EncoderClassifier(TorchModel):
    id = "torch.encoder_classifier"
    name = "Classifier · interchangeable encoder"
    description = "Transfer learning / linear probe: a head on top of a pretrained, trained-here or custom encoder"
    modalities = ("image",)

    def model_params(self):
        return slot_params([ENC]) + [Param("head_hidden", "int", 0, "Hidden units in the head (0 = linear probe)", 0, group="Head")]

    def make_optimizer(self, model, p):
        """Encoder and head as separate param groups (e.g. fine-tune a pretrained encoder at a lower lr)."""
        enc = [q for q in model[1].encoder.parameters() if q.requires_grad]
        ids = {id(q) for q in model[1].encoder.parameters()}
        groups = [{"params": [q for q in model.parameters() if id(q) not in ids and q.requires_grad]}]
        if enc:
            g = {"params": enc}
            if p.get("encoder_lr"):
                g["lr"] = p["encoder_lr"]
            if p.get("encoder_weight_decay") is not None:
                g["weight_decay"] = p["encoder_weight_decay"]
            groups.append(g)

        class _Groups:          # TorchModel.make_optimizer takes "something with .parameters()"
            def parameters(self):
                return groups
        return TorchModel.make_optimizer(_Groups(), p)

    def build(self, p, info):
        shape = list(info["input_shape"])
        enc, meta = ENC.build(p, {"data_shape": shape, "modality": "image"})
        check_shape(meta, shape, "encoder")
        norm = info.get("image_norm") or {"mean": [0.5] * shape[0], "std": [0.5] * shape[0]}
        return nn.Sequential(ToPM1(norm["mean"], norm["std"]), WithHead(enc, shape, info["output_dim"], p["head_hidden"]))
