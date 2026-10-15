"""Datasets for generative training: audio, video and text (images reuse the vision datasets, plus a flat folder).

Audio and video samples are tensors in [-1, 1]: audio (1, length), video (channels, frames, height, width).
"""
from __future__ import annotations

import math
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np

from .. import config
from ..plugins.base import DataBundle, DatasetPlugin, Param
from ..plugins.registry import register

SPLIT = [Param("val_fraction", "float", 0.05, "Held out for validation", 0.0, 0.5, group="Data split"),
         Param("test_fraction", "float", 0.05, "Held out for test (FID, test loss)", 0.0, 0.5, group="Data split")]


def _split(n: int, params: dict, seed: int):
    rng = np.random.default_rng(params.get("split_seed", 0) if params.get("split_seed", 0) >= 0 else seed)
    idx = rng.permutation(n)
    nv, nt = int(n * params.get("val_fraction", 0.05)), int(n * params.get("test_fraction", 0.05))
    return idx[nv + nt:], idx[:nv], idx[nv:nv + nt]


def _tensor_bundle(X: np.ndarray, y: np.ndarray, params, seed, modality: str, info: dict) -> DataBundle:
    import torch
    from torch.utils.data import TensorDataset
    tr, va, te = _split(len(X), params, seed)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    mk = lambda ii: TensorDataset(Xt[ii], yt[ii])
    info = {**info, "data_shape": list(X.shape[1:]), "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    return DataBundle("generation", modality, mk(tr), mk(va), mk(te), info)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


# ============================================================================= audio
def read_wav(path, target_sr: int) -> np.ndarray:
    """PCM wav -> mono float32 in [-1, 1] at target_sr (polyphase resampling)."""
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    a = np.frombuffer(raw, dt).astype(np.float32)
    a = (a - 128) / 128 if width == 1 else a / float(np.iinfo(dt).max)
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = math.gcd(sr, target_sr)
        a = resample_poly(a, target_sr // g, sr // g).astype(np.float32)
    return a


def fit_length(a: np.ndarray, length: int, peak: bool = True) -> np.ndarray:
    a = a[:length] if len(a) >= length else np.pad(a, (0, length - len(a)))
    m = np.abs(a).max()
    return (a / m * 0.95 if peak and m > 1e-6 else a).astype(np.float32)


AUDIO_PARAMS = [Param("sample_rate", "int", 8000, "Resample to this rate (Hz)", 1000, 48000, group="Audio"),
                Param("duration", "float", 1.0, "Clip length in seconds (cropped / zero-padded)", 0.05, 30.0, group="Audio")]


@register
class SyntheticTones(DatasetPlugin):
    id = "gen.audio.tones"
    name = "Synthetic tones (audio)"
    description = "Generated sine notes, chords and chirps with envelopes — no download; quick audio-model checks"
    task, modality = "generation", "audio"

    def params(self):
        return [Param("n_clips", "int", 4000, min=16, group="Audio")] + AUDIO_PARAMS + SPLIT

    def load(self, params, seed):
        rng = np.random.default_rng(1234)
        sr, L = params["sample_rate"], int(params["sample_rate"] * params["duration"])
        t = np.arange(L) / sr
        X, y = np.zeros((params["n_clips"], 1, L), np.float32), np.zeros(params["n_clips"], np.int64)
        for i in range(len(X)):
            kind = rng.integers(3)
            f0 = 110 * 2 ** (rng.integers(0, 24) / 12)
            if kind == 0:
                s = np.sin(2 * np.pi * f0 * t)
            elif kind == 1:
                s = sum(np.sin(2 * np.pi * f0 * r * t) for r in (1, 1.25, 1.5)) / 3
            else:
                f1 = f0 * rng.uniform(0.5, 2)
                s = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * t[-1])))
            env = np.minimum(1, t / 0.02) * np.exp(-t * rng.uniform(1, 6))
            X[i, 0], y[i] = fit_length(s * env, L), kind
        return _tensor_bundle(X, y, params, seed, "audio", {"sample_rate": sr, "n_classes": 3,
                                                             "class_names": ["note", "chord", "chirp"]})


@register
class AudioFolder(DatasetPlugin):
    id = "gen.audio.folder"
    name = "Audio folder (wav)"
    description = "Every .wav file under a folder (sub-folders become labels); mono, resampled, cropped/padded"
    task, modality = "generation", "audio"

    def params(self):
        return [Param("path", "str", "", "Folder with .wav files", group="Source"),
                Param("max_files", "int", 0, "Use at most N files (0 = all)", 0, group="Source")] + AUDIO_PARAMS + SPLIT

    def load(self, params, seed):
        root = Path(params["path"]).expanduser()
        files = sorted(root.rglob("*.wav"))
        if params.get("max_files"):
            files = files[: params["max_files"]]
        if not files:
            raise ValueError(f"No .wav files under {root}")
        sr, L = params["sample_rate"], int(params["sample_rate"] * params["duration"])
        X = np.stack([fit_length(read_wav(f, sr), L) for f in files])[:, None]
        labels = sorted({f.parent.name for f in files})
        y = np.array([labels.index(f.parent.name) for f in files])
        return _tensor_bundle(X, y, params, seed, "audio", {"sample_rate": sr, "n_classes": len(labels), "class_names": labels})


# ============================================================================= video
@register
class MovingMNIST(DatasetPlugin):
    id = "gen.video.moving_mnist"
    name = "Moving MNIST (video)"
    description = "Handwritten digits bouncing around a frame, generated from MNIST — the standard toy video benchmark"
    task, modality = "generation", "video"

    def params(self):
        return [Param("n_videos", "int", 4000, min=16, group="Video"),
                Param("frames", "int", 16, min=2, max=64, group="Video"),
                Param("size", "int", 32, "Frame height/width", 16, 128, group="Video"),
                Param("digits", "int", 1, "Digits per video", 1, 3, group="Video"),
                Param("fps", "int", 8, "Playback rate of generated videos", 1, 60, group="Video")] + SPLIT

    def load(self, params, seed):
        import torch
        import torch.nn.functional as F
        import torchvision
        mn = torchvision.datasets.MNIST(str(config.CACHE_DIR / "torchvision"), train=True, download=True)
        rng = np.random.default_rng(4321)
        N, T, S, k = params["n_videos"], params["frames"], params["size"], params["digits"]
        d = max(8, int(S * 0.45))
        digits = F.interpolate(mn.data[:20000, None].float() / 255, size=(d, d), mode="bilinear", antialias=True)[:, 0].numpy()
        X = np.zeros((N, 1, T, S, S), np.float32)
        y = np.zeros(N, np.int64)
        for i in range(N):
            for j in range(k):
                di = rng.integers(len(digits))
                y[i] = int(mn.targets[di]) if j == 0 else y[i]
                pos = rng.uniform(0, S - d, 2)
                ang = rng.uniform(0, 2 * np.pi)
                vel = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(1, 2.5) * S / 32
                for t in range(T):
                    r, c = pos.astype(int)
                    X[i, 0, t, r:r + d, c:c + d] = np.maximum(X[i, 0, t, r:r + d, c:c + d], digits[di])
                    pos += vel
                    for a in range(2):
                        if pos[a] < 0 or pos[a] > S - d:
                            vel[a] *= -1
                            pos[a] = np.clip(pos[a], 0, S - d)
        X = X * 2 - 1
        return _tensor_bundle(X, y, params, seed, "video", {"fps": params["fps"], "n_classes": 10,
                                                             "class_names": [str(i) for i in range(10)]})


# ============================================================================= images (flat folder)
@register
class ImageFolderFlat(DatasetPlugin):
    id = "gen.image.folder"
    name = "Image folder (any layout)"
    description = "Every image under a folder (no class sub-folders needed), resized to a square"
    task, modality = "generation", "image"

    def params(self):
        return [Param("path", "str", "", "Folder with .png / .jpg / .webp images", group="Source"),
                Param("image_size", "int", 64, "Square size", 8, 1024, group="Preprocessing"),
                Param("grayscale", "bool", False, group="Preprocessing"),
                Param("center_crop", "bool", True, "Crop to a centered square before resizing", group="Preprocessing"),
                Param("hflip", "bool", True, "Random horizontal flips while training", group="Preprocessing"),
                Param("max_images", "int", 0, "Use at most N images (0 = all)", 0, group="Source")] + SPLIT

    def load(self, params, seed):
        import torch
        import torchvision.transforms as T
        from PIL import Image
        root = Path(params["path"]).expanduser()
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        if params.get("max_images"):
            files = files[: params["max_images"]]
        if not files:
            raise ValueError(f"No images under {root}")
        s, ch = params["image_size"], 1 if params["grayscale"] else 3
        base = ([T.Lambda(lambda im: T.CenterCrop(min(im.size))(im))] if params["center_crop"] else []) + \
               [T.Resize((s, s)), T.ToTensor(), T.Normalize([0.5] * ch, [0.5] * ch)]

        class _DS(torch.utils.data.Dataset):
            def __init__(self, items, flip):
                self.items, self.tf = items, T.Compose(([T.RandomHorizontalFlip()] if flip else []) + base)

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                im = Image.open(self.items[i]).convert("L" if ch == 1 else "RGB")
                return self.tf(im), 0

        tr, va, te = _split(len(files), params, seed)
        pick = lambda ii, flip: _DS([files[i] for i in ii], flip)
        info = {"input_shape": [ch, s, s], "image_norm": {"mean": [0.5] * ch, "std": [0.5] * ch, "size": [s, s], "channels": ch},
                "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        return DataBundle("generation", "image", pick(tr, params["hflip"]), pick(va, False), pick(te, False), info)


# ============================================================================= text
TEXT_SPLIT = [Param("val_fraction", "float", 0.05, "Tail of the text used for validation", 0.0, 0.5, group="Data split"),
              Param("test_fraction", "float", 0.05, "Tail of the text used for test", 0.0, 0.5, group="Data split")]


def _text_bundle(text: str, params, info) -> DataBundle:
    n = len(text)
    a = int(n * (1 - params["val_fraction"] - params["test_fraction"]))
    b = int(n * (1 - params["test_fraction"]))
    return DataBundle("generation", "text", text[:a], text[a:b], text[b:], {**info, "n_chars": n})


@register
class TinyShakespeare(DatasetPlugin):
    id = "gen.text.shakespeare"
    name = "Tiny Shakespeare (text)"
    description = "1.1 MB of Shakespeare plays (Karpathy's char-rnn corpus) — the classic character-level LM benchmark"
    task, modality = "generation", "text"
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def params(self):
        return TEXT_SPLIT

    def data_paths(self):
        return [config.CACHE_DIR / "text" / "tinyshakespeare.txt"]

    def load(self, params, seed):
        text = _download(self.URL, config.CACHE_DIR / "text" / "tinyshakespeare.txt").read_text()
        return _text_bundle(text, params, {})


@register
class TextFiles(DatasetPlugin):
    id = "gen.text.files"
    name = "Text file(s) on disk"
    description = "A .txt/.md file, or every .txt/.md/.py file under a folder, as one training corpus"
    task, modality = "generation", "text"

    def params(self):
        return [Param("path", "str", "", "File or folder", group="Source"),
                Param("extensions", "str", ".txt,.md", "File types to include from a folder", group="Source")] + TEXT_SPLIT

    def load(self, params, seed):
        p = Path(params["path"]).expanduser()
        exts = tuple(e.strip() for e in params["extensions"].split(",") if e.strip())
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix in exts)
        if not files:
            raise ValueError(f"No text files at {p}")
        text = "\n\n".join(f.read_text(errors="ignore") for f in files)
        return _text_bundle(text, params, {"n_files": len(files)})


@register
class HFText(DatasetPlugin):
    id = "gen.text.hf"
    name = "Hugging Face text dataset"
    description = "Any Hugging Face dataset with a text column (e.g. roneneldan/TinyStories, wikitext)"
    task, modality = "generation", "text"

    def params(self):
        return [Param("dataset", "str", "roneneldan/TinyStories", suggestions=["roneneldan/TinyStories", "wikitext", "imdb"],
                      group="Source"),
                Param("config", "str", "", "Dataset config (e.g. wikitext-2-raw-v1)", group="Source"),
                Param("column", "str", "text", group="Source"),
                Param("max_records", "int", 20000, "Records taken from the train split (0 = all)", 0, group="Source")] + TEXT_SPLIT

    def load(self, params, seed):
        from datasets import load_dataset
        ds = load_dataset(params["dataset"], params["config"] or None, split="train", cache_dir=str(config.CACHE_DIR / "hf"))
        if params["max_records"]:
            ds = ds.select(range(min(params["max_records"], len(ds))))
        text = "\n\n".join(str(t) for t in ds[params["column"]] if t)
        return _text_bundle(text, params, {"n_records": len(ds)})
