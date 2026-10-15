"""Turning generated tensors into files the dashboard shows (image grids, wav + spectrograms, mp4s, text),
and sample-quality metrics (FID / KID for images)."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import torch


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor (..., C, H, W) -> uint8 (..., H, W, 3)."""
    a = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
    a = a.movedim(-3, -1).numpy()
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, -1)
    return a[..., :3]


def grid(imgs: np.ndarray, nrow: int | None = None, pad: int = 2, scale_to: int = 96) -> np.ndarray:
    """(N, H, W, 3) -> one tiled image; small images are upscaled (nearest) so they're visible."""
    import cv2
    n, h, w = imgs.shape[:3]
    s = max(1, scale_to // max(h, w))
    if s > 1:
        imgs = np.stack([cv2.resize(i, (w * s, h * s), interpolation=cv2.INTER_NEAREST) for i in imgs])
        h, w = h * s, w * s
    nrow = nrow or math.ceil(math.sqrt(n))
    rows = math.ceil(n / nrow)
    out = np.full((rows * (h + pad) + pad, nrow * (w + pad) + pad, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        out[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = im
    return out


def save_png(img: np.ndarray, path: Path) -> Path:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def save_image_grid(x: torch.Tensor, path: Path, nrow: int | None = None) -> Path:
    return save_png(grid(to_uint8(x), nrow), path)


def save_mp4(frames: list[np.ndarray], path: Path, fps: int = 8) -> Path:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    frames = [f[: h // 2 * 2, : w // 2 * 2] for f in frames]
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1,
                     ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    return path


def save_video_grid(x: torch.Tensor, path: Path, fps: int = 8) -> Path:
    """(N, C, T, H, W) in [-1, 1] -> mp4 where each frame tiles the N videos."""
    frames = []
    for t in range(x.shape[2]):
        frames.append(grid(to_uint8(x[:, :, t])))
    return save_mp4(frames, path, fps)


def save_trajectory(steps: list[torch.Tensor], path: Path, fps: int = 10) -> Path:
    """Diffusion denoising: one grid per recorded step -> mp4 (holds the final frame for a second)."""
    frames = [grid(to_uint8(s)) for s in steps]
    frames += [frames[-1]] * fps
    return save_mp4(frames, path, fps)


def save_wav(audio: np.ndarray, sr: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(a.tobytes())
    return path


def spectrogram_image(audio: np.ndarray, sr: int, height: int = 96, width: int = 240) -> np.ndarray:
    """Log-magnitude STFT as an RGB image (viridis-like ramp), time left -> right, low frequencies at the bottom."""
    import cv2
    n_fft = 512 if sr >= 16000 else 256
    hop = n_fft // 4
    x = np.pad(audio.astype(np.float32), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)).T * 10)
    mag = (mag / max(mag.max(), 1e-6) * 255).astype(np.uint8)[::-1]
    img = cv2.applyColorMap(cv2.resize(mag, (width, height), interpolation=cv2.INTER_AREA), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_audio_set(x: torch.Tensor, sr: int, folder: Path, prefix: str) -> dict:
    """(N, 1, L) in [-1, 1] -> one wav + one spectrogram png per clip. Returns paths."""
    a = x.detach().float().cpu().numpy()[:, 0]
    clips = [save_wav(c, sr, folder / f"{prefix}_{i}.wav") for i, c in enumerate(a)]
    specs = [save_png(spectrogram_image(c, sr), folder / f"{prefix}_{i}.png") for i, c in enumerate(a)]
    return {"clips": clips, "specs": specs, "spectrogram": specs[0]}


# ============================================================================= FID / KID
_inception = None


def _inception_model(device):
    global _inception
    if _inception is None:
        import torchvision
        from .. import config
        torch.hub.set_dir(str(config.CACHE_DIR / "torch_hub"))
        m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = torch.nn.Identity()
        _inception = m.eval()
    return _inception.to(device)


@torch.no_grad()
def inception_features(x: torch.Tensor, device, batch: int = 100) -> np.ndarray:
    """[-1, 1] images (N, C, H, W) -> 2048-d pool features (ImageNet Inception-v3)."""
    import torch.nn.functional as F
    m = _inception_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(x), batch):
        b = x[i:i + batch].to(device).float()
        if b.shape[1] == 1:
            b = b.repeat(1, 3, 1, 1)
        b = F.interpolate((b[:, :3] + 1) / 2, size=(299, 299), mode="bilinear", align_corners=False)
        out.append(m((b - mean) / std).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fid_kid(real: np.ndarray, fake: np.ndarray) -> dict:
    """Fréchet Inception Distance and Kernel Inception Distance (x1000) between two feature sets."""
    from scipy import linalg
    mu1, mu2 = real.mean(0), fake.mean(0)
    s1, s2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):          # older scipy returned (sqrtm, error) with disp=False
        covmean = covmean[0]
    covmean = np.real(covmean)
    fid = float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
    d = real.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3
    n = min(len(real), len(fake), 1000)
    r, f = real[:n], fake[:n]
    kxx, kyy, kxy = k(r, r), k(f, f), k(r, f)
    kid = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean()
    return {"fid": fid, "kid_x1000": float(kid * 1000)}
