"""Shared training loop for generative models (diffusion, VAE, GAN, custom) on images, audio and video.

A method plugin (`GenerativeModel` subclass) builds an `nn.Module` that follows this small contract:

    losses(x, y=None[, group])  -> dict of scalar tensors; "loss" is optimized (others are logged)
    sample(n, y=None, generator=None) -> tensor shaped like the data, values in [-1, 1]
    optional: optimizer_groups() -> {name: params}  and  update_groups(step) -> [names]   (e.g. GAN: "d", "g")
    optional: sample_trajectory(n, generator=None, frames=10) -> list of tensors (e.g. diffusion denoising)
    optional: reconstruct(x) -> tensor (e.g. VAE)
    optional: set_progress(fraction_of_training)  (e.g. KL warm-up)

Your own model can implement that contract directly (see examples/plugins/models/custom_generative.py) and the
dashboard handles everything else: data, devices, AMP, gradient accumulation, EMA, schedules, checkpoints
(best + final, resumable), fixed-noise sample snapshots, sample-quality metrics and multi-seed aggregation.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from ..plugins.base import ModelPlugin, Param
from .components import copy_sharing_frozen, saveable_state
from ..training.utils import EarlyStopping

GEN_TRAIN = [
    Param("epochs", "int", 50, min=1, group="Training"),
    Param("batch_size", "int", 64, min=1, group="Training"),
    Param("grad_accum", "int", 1, "Accumulate gradients over N batches (effective batch = batch_size x N)", 1, group="Training"),
    Param("optimizer", "choice", "adamw", choices=["adamw", "adam"], group="Training"),
    Param("lr", "float", 2e-4, "Learning rate", 1e-8, log=True, group="Training"),
    Param("weight_decay", "float", 0.0, min=0.0, log=True, group="Training"),
    Param("beta1", "float", 0.9, "Adam beta1 (GANs usually use 0.5)", 0.0, 0.999, group="Training"),
    Param("scheduler", "choice", "constant", choices=["constant", "cosine"], group="Training"),
    Param("warmup_steps", "int", 0, "Linear learning-rate warm-up (optimizer steps)", 0, group="Training"),
    Param("grad_clip", "float", 1.0, "Max gradient norm (0 = off)", 0.0, group="Training"),
    Param("ema_decay", "float", 0.999, "Exponential moving average of weights used for sampling (0 = off)", 0.0, 0.99999,
          group="Training"),
    Param("amp", "bool", False, "Mixed precision (GPU only)", group="Training"),
    Param("max_batches_per_epoch", "int", 0, "Cap batches per epoch (0 = whole dataset) — for quick runs", 0, group="Training"),
    Param("num_workers", "int", 2, "DataLoader worker processes", 0, group="Training"),
    Param("early_stopping_patience", "int", 0, "Epochs without val-loss improvement before stopping (0 = off)", 0,
          group="Early stopping"),
    Param("n_samples", "int", 16, "Samples drawn for each snapshot (fixed noise, so you see the same samples improve)", 1,
          256, group="Sampling"),
    Param("eval_samples", "int", 1000, "Images generated at the end for FID / KID (0 = skip; downloads Inception once)", 0,
          group="Sampling"),
]

IMAGE_PARAMS = [Param("image_size", "int", 0, "Resize images to N x N (0 = keep the dataset's size)", 0, 1024, group="Data")]

INJECT_NAMES = {
    "channels": ("in_ch", "in_channels", "input_channels", "channels", "out_ch", "out_channels", "output_channels", "n_channels"),
    "image_size": ("img_size", "image_size", "resolution"),
    "dims": ("dims",),
    "length": ("length", "seq_len", "n_samples_audio", "audio_length"),
    "num_classes": ("num_classes", "n_classes"),
    "data_shape": ("data_shape",),
}


def inject_kwargs(sig_params, info: dict) -> dict:
    """Constructor arguments the dataset determines (channels, size, dims, ...)."""
    shape = info["data_shape"]
    vals = {"channels": shape[0], "image_size": shape[-1], "dims": len(shape) - 1, "length": shape[-1],
            "num_classes": info.get("n_classes", 0) if info.get("conditional") else 0, "data_shape": tuple(shape)}
    out = {}
    for key, names in INJECT_NAMES.items():
        for n in names:
            if n in sig_params:
                out[n] = vals[key]
    return out


class EMA:
    """Exponential moving average of weights, with the usual warm-up: decay = min(decay, (1 + n) / (10 + n)),
    so short runs aren't dominated by the random initialization."""

    def __init__(self, model, decay: float):
        self.decay, self.n = decay, 0
        self.shadow = copy_sharing_frozen(model).eval().requires_grad_(False)   # frozen parts are shared, not copied

    def update(self, model):
        import torch
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                if s is not p:
                    s.lerp_(p.detach(), 1 - d)
            for s, b in zip(self.shadow.buffers(), model.buffers()):
                if s is not b:
                    s.copy_(b)


class GenerativeModel(ModelPlugin):
    """Base for generative method plugins. Subclasses implement `method_params()` and `build(params, info)`."""
    framework = "generative"
    tasks = ("generation",)
    modalities = ("image", "audio", "video")
    session_task = "generation"
    uses_gpu = True
    supports_init = True
    supports_eval = True
    supports_resume = True
    default_monitor = "val/loss"      # GANs override (their losses don't measure sample quality)

    def meta(self) -> dict:
        return {**super().meta(), "any_task": True, "session_task": self.session_task}

    def compatible(self, task: str, modality: str) -> bool:
        # any dataset of a supported modality works: labels are ignored (or used for class conditioning)
        return modality in self.modalities

    def method_params(self) -> list[Param]:
        return []

    def network_params(self) -> list[Param]:
        return []

    def params(self):
        return self.method_params() + self.network_params() + IMAGE_PARAMS + GEN_TRAIN

    def build(self, p: dict, info: dict):
        raise NotImplementedError

    # ------------------------------------------------------------------ data
    def prepare(self, data, p: dict, ctx):
        """-> (info, loaders, to_model). Images are mapped to [-1, 1] (undoing the dataset's normalization)."""
        import torch
        from torch.utils.data import DataLoader
        info = dict(data.info)
        mod = data.modality
        info["modality"] = mod
        if mod == "image":
            norm = info.get("image_norm") or {"mean": [0.5] * info["input_shape"][0], "std": [0.5] * info["input_shape"][0]}
            mean = torch.tensor(norm["mean"]).view(1, -1, 1, 1)
            std = torch.tensor(norm["std"]).view(1, -1, 1, 1)
            c, h, w = info["input_shape"]
            size = p.get("image_size") or 0
            shape = [c, size or h, size or w]

            def to_model(batch, device):
                x, y = batch[0], batch[1] if len(batch) > 1 else None
                x = (x * std + mean) * 2 - 1
                if size and (x.shape[-1] != size or x.shape[-2] != size):
                    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
                return x.to(device, non_blocking=True), (y.to(device) if torch.is_tensor(y) else None)
        else:
            shape = list(info["data_shape"])

            def to_model(batch, device):
                x, y = batch[0], batch[1] if len(batch) > 1 else None
                return x.to(device, non_blocking=True), (y.to(device) if torch.is_tensor(y) else None)
        info["data_shape"] = shape
        info["conditional"] = bool(p.get("class_conditional")) and bool(info.get("n_classes"))
        kw = dict(batch_size=p["batch_size"], num_workers=p["num_workers"], pin_memory=ctx.device == "cuda",
                  persistent_workers=p["num_workers"] > 0)
        dl = {"train": DataLoader(data.train, shuffle=True, drop_last=len(data.train) > p["batch_size"], **kw)}
        for split in ("val", "test"):
            ds = getattr(data, split)
            dl[split] = DataLoader(ds, shuffle=False, **kw) if ds is not None and len(ds) else None
        return info, dl, to_model

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _optimizers(net, p):
        import torch
        groups = net.optimizer_groups() if hasattr(net, "optimizer_groups") else {"main": list(net.parameters())}
        # each component (slot) is its own param group with its own lr / weight decay / beta1
        slots = net.slot_parameters() if hasattr(net, "slot_parameters") else {}
        owner = {id(q): s for s, ps in slots.items() for q in ps}
        legacy = {"discriminator": p.get("lr_d")}        # older GAN sessions stored the D learning rate as lr_d
        opts = {}
        for g, params in groups.items():
            params = [q for q in params if q.requires_grad]          # frozen components aren't optimized
            if not params:
                continue
            by: dict = {}
            for q in params:
                by.setdefault(owner.get(id(q)), []).append(q)
            pgs = []
            for slot, ps in by.items():
                pg = {"params": ps, "name": slot or g}
                if slot:
                    if p.get(f"{slot}_lr") or legacy.get(slot):
                        pg["lr"] = p.get(f"{slot}_lr") or legacy[slot]
                    if p.get(f"{slot}_weight_decay") is not None:
                        pg["weight_decay"] = p[f"{slot}_weight_decay"]
                    if p.get(f"{slot}_beta1") is not None:
                        pg["betas"] = (p[f"{slot}_beta1"], 0.999)
                pgs.append(pg)
            lr = p.get(f"lr_{g}") or p["lr"]
            cls = torch.optim.Adam if p["optimizer"] == "adam" else torch.optim.AdamW
            opts[g] = cls(pgs, lr=lr, betas=(p["beta1"], 0.999), weight_decay=p["weight_decay"])
        return opts

    @staticmethod
    def _schedulers(opts, p, total_steps):
        import torch

        def fn(step):
            w = p["warmup_steps"]
            f = min(1.0, (step + 1) / w) if w else 1.0
            if p["scheduler"] == "cosine":
                prog = min(1.0, step / max(1, total_steps))
                f *= 0.5 * (1 + math.cos(math.pi * prog))
            return f
        return {g: torch.optim.lr_scheduler.LambdaLR(o, fn) for g, o in opts.items()}

    @staticmethod
    def load_weights(net, path, device, ctx=None):
        """Load model.pt; also accepts a plain state dict of just the network (e.g. an external checkpoint)."""
        import torch
        sd = torch.load(path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        own = net.state_dict()
        if not any(k in own for k in sd) and hasattr(net, "net"):
            sd = {f"net.{k}": v for k, v in sd.items()}
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        net.load_state_dict(ok, strict=False)
        if ctx:
            missing = [k for k in own if k not in ok]
            ctx.print(f"loaded {len(ok)}/{len(own)} tensors from {path}" + (f" (not in checkpoint: {missing[:8]}…)" if missing else ""))
        return net

    def _fixed_sample(self, net, n, info, device, seed, y=None):
        import torch
        g = torch.Generator(device=device).manual_seed(int(seed) + 9973)
        if y is None and info.get("conditional"):
            y = torch.arange(n, device=device) % info["n_classes"]
        with torch.no_grad():
            return net.sample(n, y=y, generator=g)

    def log_samples(self, net, ctx, step, info, device, name="generated", trajectory=False, x_real=None):
        """Generate fixed-noise samples and record them as a snapshot for the modality."""
        from . import media
        n = ctx.params.get("n_samples", 16)
        x = self._fixed_sample(net, n, info, device, ctx.seed)
        mod = info["modality"]
        f = ctx.media_path(f"{name}_{step}")
        if mod == "image":
            ctx.log_media(step, name, "image", file=media.save_image_grid(x, f.with_suffix(".png")))
            if trajectory and hasattr(net, "sample_trajectory"):
                import torch
                g = torch.Generator(device=device).manual_seed(int(ctx.seed) + 9973)
                with torch.no_grad():
                    traj = net.sample_trajectory(min(n, 16), generator=g, frames=40)
                ctx.log_media(step, "denoising", "video", file=media.save_trajectory(traj, ctx.media_path(f"denoising_{step}.mp4")))
            if x_real is not None and hasattr(net, "reconstruct"):
                import torch
                with torch.no_grad():
                    rec = net.reconstruct(x_real[:8])
                both = torch.cat([x_real[:8], rec], 0)
                ctx.log_media(step, "reconstructions", "image",
                              file=media.save_image_grid(both, ctx.media_path(f"recon_{step}.png"), nrow=min(8, len(x_real))))
        elif mod == "video":
            ctx.log_media(step, name, "video", file=media.save_video_grid(x[: min(n, 9)], f.with_suffix(".mp4"), fps=info.get("fps", 8)))
        elif mod == "audio":
            sr = info.get("sample_rate", 16000)
            res = media.save_audio_set(x[: min(n, 6)], sr, ctx.media_dir / f"{name}_{step}", "clip")
            rel = lambda q: str(Path(q).relative_to(ctx.run_dir))
            ctx.log_media(step, name, "audio", data={"clips": [rel(c) for c in res["clips"]], "specs": [rel(s) for s in res["specs"]],
                                                       "sample_rate": sr}, file=res["spectrogram"])

    def _eval_losses(self, net, loader, to_model, device, max_batches=50, use_amp=False):
        import torch
        if loader is None:
            return {}
        tot, n = {}, 0
        net.eval()
        with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
            g = torch.Generator(device=device).manual_seed(1234)     # same noise/timesteps every evaluation
            for i, batch in enumerate(loader):
                if max_batches and i >= max_batches:
                    break
                x, y = to_model(batch, device)
                out = net.losses(x, y, generator=g) if _accepts(net.losses, "generator") else net.losses(x, y)
                for k, v in out.items():
                    tot[k] = tot.get(k, 0.0) + float(v) * len(x)
                n += len(x)
        return {k: v / max(n, 1) for k, v in tot.items()}

    def sample_quality(self, net, info, loader, to_model, device, n_eval, ctx, prefix="test"):
        """FID / KID against real images of the split (images only)."""
        import torch

        from . import media
        if info["modality"] != "image" or not n_eval or loader is None:
            return {}
        ctx.set_progress(0.99, "computing FID")
        real = []
        for batch in loader:
            real.append(to_model(batch, torch.device("cpu"))[0])
            if sum(len(r) for r in real) >= n_eval:
                break
        real = torch.cat(real)[:n_eval]
        fake, g = [], torch.Generator(device=device).manual_seed(int(ctx.seed) + 77)
        net.eval()
        with torch.no_grad():
            while sum(len(f) for f in fake) < len(real):
                if ctx.should_stop():
                    return {}
                b = min(250, len(real) - sum(len(f) for f in fake))
                y = (torch.arange(b, device=device) % info["n_classes"]) if info.get("conditional") else None
                fake.append(net.sample(b, y=y, generator=g, fast=True).cpu() if _accepts(net.sample, "fast")
                            else net.sample(b, y=y, generator=g).cpu())
        fake = torch.cat(fake)
        try:
            r = media.fid_kid(media.inception_features(real, device), media.inception_features(fake, device))
        except Exception as e:     # e.g. no internet for the Inception weights
            ctx.print(f"FID skipped: {e}")
            return {}
        return {f"{prefix}/fid": r["fid"], f"{prefix}/kid_x1000": r["kid_x1000"]}

    # ------------------------------------------------------------------ training
    def train(self, data, ctx):
        import torch
        p = ctx.params
        device = torch.device(ctx.device)
        info, dl, to_model = self.prepare(data, p, ctx)
        net = self.build(p, info).to(device)
        if ctx.init_path and not ctx.resume:
            self.load_weights(net, ctx.init_path / "model.pt", device, ctx)
        n_params = sum(x.numel() for x in net.parameters())
        ctx.print(net)
        ctx.print(f"parameters: {n_params:,} · data shape {info['data_shape']} · {info['modality']}")

        opts = self._optimizers(net, p)
        per_epoch = len(dl["train"]) if not p["max_batches_per_epoch"] else min(len(dl["train"]), p["max_batches_per_epoch"])
        total = p["epochs"] * math.ceil(per_epoch / p["grad_accum"])
        scheds = self._schedulers(opts, p, total)
        use_amp = p["amp"] and device.type == "cuda" and not getattr(net, "no_amp", False)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ema = EMA(net, p["ema_decay"]) if p["ema_decay"] > 0 else None
        sampler = lambda: ema.shadow if ema else net         # weights used for evaluation & samples

        monitor = p.get("monitor_metric") or self.default_monitor
        if dl["val"] is None and monitor == "val/loss":
            monitor = "train/loss"
        stopper = EarlyStopping(p["early_stopping_patience"], "min")
        start_epoch, step, t_spent = 0, 0, 0.0
        best_path, last_path = ctx.checkpoint_path("best.pt"), ctx.checkpoint_path("last.pt")
        if ctx.resume and last_path.exists():
            ck = torch.load(last_path, map_location=device, weights_only=False)
            net.load_state_dict(ck["model"], strict=False)
            if ema and ck.get("ema"):
                ema.shadow.load_state_dict(ck["ema"], strict=False)
                ema.n = ck.get("step", 0)
            for g, o in opts.items():
                o.load_state_dict(ck["opts"][g])
            for g, s in scheds.items():
                s.load_state_dict(ck["scheds"][g])
            start_epoch, step, t_spent = ck["epoch"], ck["step"], ck.get("time", 0.0)
            stopper.best, stopper.best_step = ck.get("best"), ck.get("best_epoch", -1)
            ctx.truncate_after(start_epoch)
            ctx.print(f"resumed at epoch {start_epoch}")

        snap_due = ctx.snapshot_schedule(p["epochs"])
        x_real = next(iter(dl["val"] or dl["train"]))
        x_real = to_model(x_real, device)[0][:8]
        t0 = time.time()
        epoch = start_epoch
        update = getattr(net, "update_groups", None)
        for epoch in range(start_epoch + 1, p["epochs"] + 1):
            net.train()
            sums, cnt = {}, 0
            for g in opts.values():
                g.zero_grad(set_to_none=True)
            for i, batch in enumerate(dl["train"]):
                if p["max_batches_per_epoch"] and i >= p["max_batches_per_epoch"]:
                    break
                if hasattr(net, "set_progress"):
                    net.set_progress((epoch - 1 + i / per_epoch) / p["epochs"])
                x, y = to_model(batch, device)
                groups = update(step) if update else list(opts)
                multi = len(opts) > 1      # e.g. GAN: each group's loss also back-props into the other network
                for g in groups:
                    if multi:
                        for o in opts.values():
                            o.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, enabled=use_amp):
                        out = net.losses(x, y, group=g) if len(opts) > 1 else net.losses(x, y)
                    scaler.scale(out["loss"] / (1 if multi else p["grad_accum"])).backward()
                    for k, v in out.items():
                        key = k if len(opts) == 1 or k != "loss" else f"{g}_loss"
                        sums[key] = sums.get(key, 0.0) + float(v.detach())
                    last_micro = multi or (i + 1) % p["grad_accum"] == 0 or (i + 1) == per_epoch
                    if last_micro:
                        o = opts[g]
                        if p["grad_clip"]:
                            scaler.unscale_(o)
                            torch.nn.utils.clip_grad_norm_([q for pg in o.param_groups for q in pg["params"]], p["grad_clip"])
                        scaler.step(o)
                        scaler.update()
                        o.zero_grad(set_to_none=True)
                        scheds[g].step()
                if (i + 1) % p["grad_accum"] == 0 or (i + 1) == per_epoch:
                    step += 1
                    if ema:
                        ema.update(net)
                cnt += 1
                if i % 20 == 0:
                    ctx.set_progress((epoch - 1 + i / per_epoch) / p["epochs"], f"epoch {epoch}/{p['epochs']} · batch {i + 1}/{per_epoch}")
                if ctx.should_stop():
                    break
            row = {f"train/{k}": v / max(cnt, 1) for k, v in sums.items()}
            row["lr"] = next(iter(opts.values())).param_groups[0]["lr"]
            pgs = [pg for o in opts.values() for pg in o.param_groups]
            if len({pg["lr"] for pg in pgs}) > 1:          # per-component learning rates: chart each
                row.update({f"lr_{pg['name']}": pg["lr"] for pg in pgs})
            row.update({f"val/{k}": v for k, v in self._eval_losses(sampler(), dl["val"], to_model, device, 50, use_amp).items()})
            if ctx.should_stop():
                ctx.log(epoch, row)
                break
            if p.get("fid_every") and epoch % p["fid_every"] == 0:
                row.update(self.sample_quality(sampler(), info, dl["val"] or dl["train"], to_model, device,
                                               min(p.get("eval_samples") or 500, 500), ctx, "val"))
            ctx.log(epoch, row)
            value = row.get(monitor)
            stop = False
            if value is not None and math.isfinite(value):
                if stopper.improved(value):
                    torch.save(saveable_state(sampler()), best_path)
                stop = stopper.step(value, epoch)
            if snap_due(epoch):
                self.log_samples(sampler(), ctx, epoch, info, device, x_real=x_real)
            torch.save({"model": saveable_state(net), "ema": saveable_state(ema.shadow) if ema else None,
                        "opts": {g: o.state_dict() for g, o in opts.items()}, "scheds": {g: s.state_dict() for g, s in scheds.items()},
                        "epoch": epoch, "step": step, "time": t_spent + time.time() - t0, "best": stopper.best,
                        "best_epoch": stopper.best_step}, last_path)
            ctx.flush()
            if stop:
                ctx.print(f"early stopping at epoch {epoch} (best {monitor}={stopper.best:.4g} @ epoch {stopper.best_step})")
                break

        final = sampler()
        summary = {"epochs_trained": epoch, "n_params": n_params, "train_time_s": t_spent + time.time() - t0}
        if stopper.best is not None:
            summary.update({"best_epoch": stopper.best_step, f"best_{monitor}": stopper.best})
        keep = ctx.keep
        has_best = best_path.exists() and monitor != "none"
        if keep in ("both", "none") and has_best:
            fin = self._eval_losses(final, dl["test"] or dl["val"], to_model, device, 0, use_amp)
            summary.update({f"final_model/test/{k}": v for k, v in fin.items()})
            if keep == "both":
                torch.save(saveable_state(final), ctx.final_path("model.pt"))
                self._write_gen_config(ctx.final_dir, info, p)
        model = final
        if has_best and keep != "final":
            model = copy_sharing_frozen(final)
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=False)
        summary.update(self._finish(model, info, dl, to_model, device, ctx, use_amp, x_real))
        if ctx.save_model:
            torch.save(saveable_state(model), ctx.model_path("model.pt"))
            self._write_gen_config(ctx.model_dir, info, p)
        return summary

    def _finish(self, model, info, dl, to_model, device, ctx, use_amp, x_real) -> dict:
        out = {f"test/{k}": v for k, v in self._eval_losses(model, dl["test"] or dl["val"], to_model, device, 0, use_amp).items()}
        if ctx.should_stop():
            return out
        self.log_samples(model, ctx, ctx.params["epochs"] + 1, info, device,
                         name="final_samples", trajectory=True, x_real=x_real)
        out.update(self.sample_quality(model, info, dl["test"] or dl["val"], to_model, device, ctx.params.get("eval_samples", 0), ctx))
        return out

    @staticmethod
    def _write_gen_config(folder: Path, info: dict, p: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        keep = {k: v for k, v in info.items() if k in ("data_shape", "modality", "sample_rate", "fps", "n_classes", "class_names",
                                                     "conditional", "vocab", "tokenizer", "block_size")}
        (folder / "generation.json").write_text(json.dumps(keep, default=str))

    # ------------------------------------------------------------------ evaluation of a saved model
    def evaluate(self, data, ctx):
        import torch
        p = ctx.params
        device = torch.device(ctx.device)
        info, dl, to_model = self.prepare(data, p, ctx)
        net = self.build(p, info).to(device)
        if ctx.init_path:
            self.load_weights(net, ctx.init_path / "model.pt", device, ctx)
        x_real = to_model(next(iter(dl["val"] or dl["train"])), device)[0][:8]
        return self._finish(net, info, dl, to_model, device, ctx, False, x_real)


def _accepts(fn, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
