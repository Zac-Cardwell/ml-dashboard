"""Custom loss functions: every public fn(pred, target) here becomes a choice in generative model forms."""
import torch
import torch.nn.functional as F


def edge_aware_l1(pred, target):
    """L1 plus L1 on image gradients (sharper edges than plain MSE/L1)."""
    loss = F.l1_loss(pred, target)
    if pred.dim() == 4:
        dx = lambda a: a[..., :, 1:] - a[..., :, :-1]
        dy = lambda a: a[..., 1:, :] - a[..., :-1, :]
        loss = loss + 0.5 * (F.l1_loss(dx(pred), dx(target)) + F.l1_loss(dy(pred), dy(target)))
    return loss


def log_cosh(pred, target):
    """Smooth, outlier-robust regression loss."""
    d = pred - target
    return (d + F.softplus(-2 * d) - torch.log(torch.tensor(2.0, device=d.device))).mean()
