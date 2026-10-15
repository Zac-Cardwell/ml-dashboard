"""PyTorch models + a generic supervised training loop.

For your own architecture either:
  * upload a file with a plain `nn.Module` (constructor args with defaults become form fields;
    `input_dim`, `input_shape`, `in_channels`, `num_classes`, `output_dim` are injected), or
  * subclass `TorchModel` and implement `build(params, info)` for full control.
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np

from ..plugins.base import ModelPlugin, Param
from ..plugins.registry import _params_from_signature, register
from ..training.utils import EarlyStopping, confusion, metric_mode, task_metrics

TRAIN_PARAMS = [
    Param("epochs", "int", 30, min=1, group="Training"),
    Param("batch_size", "int", 64, min=1, group="Training"),
    Param("optimizer", "choice", "adamw", choices=["adamw", "adam", "sgd", "rmsprop"], group="Training"),
    Param("lr", "float", 1e-3, "Learning rate", 1e-8, log=True, group="Training"),
    Param("weight_decay", "float", 0.0, min=0.0, log=True, group="Training"),
    Param("momentum", "float", 0.9, "SGD/RMSprop momentum", 0.0, 1.0, group="Training"),
    Param("scheduler", "choice", "none", choices=["none", "cosine", "step", "plateau"], group="Training"),
    Param("grad_clip", "float", 0.0, "Max gradient norm (0 = off)", 0.0, group="Training"),
    Param("amp", "bool", False, "Mixed precision (GPU only)", group="Training"),
    Param("num_workers", "int", 2, "DataLoader worker processes", 0, group="Training"),
    Param("early_stopping_patience", "int", 10, "Epochs without improvement before stopping (0 = off)", 0, group="Early stopping"),
    Param("early_stopping_metric", "str", "val/loss", "Metric to monitor",
          suggestions=["val/loss", "val/acc", "val/f1", "val/rmse", "val/r2"], group="Early stopping"),
    Param("early_stopping_min_delta", "float", 0.0, min=0.0, group="Early stopping"),
    Param("restore_best", "bool", True, "Evaluate test with the best checkpoint, not the last epoch", group="Early stopping"),
]


class TorchModel(ModelPlugin):
    framework = "torch"
    tasks = ("classification", "regression")
    modalities = ("tabular", "image")
    uses_gpu = True
    supports_init = True
    supports_eval = True
    supports_resume = True

    def model_params(self) -> list[Param]:
        return []

    def params(self):
        return self.model_params() + TRAIN_PARAMS

    def build(self, params: dict, info: dict):
        """Return an nn.Module. Output size: info['output_dim']."""
        raise NotImplementedError

    # -------------------------------------------------------------- setup shared by train / evaluate
    def _prepare(self, data, ctx, p):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        task, info = data.task, dict(data.info)
        info["output_dim"] = info.get("n_classes", 1) if task == "classification" else 1
        if data.modality == "tabular":
            info.setdefault("input_shape", [info["n_features"]])
        info["input_dim"] = int(np.prod(info["input_shape"]))
        device = torch.device(ctx.device)

        train_w = (data.weights or {}).get("train") if data.modality == "tabular" else None

        def to_ds(split, w=None):
            if data.modality == "tabular":
                X, y = split
                if not len(X):
                    return None
                # optional per-example weights (e.g. recency weighting of live data) ride along as a third tensor
                return TensorDataset(torch.as_tensor(X), torch.as_tensor(y), torch.as_tensor(w)) if w is not None else \
                    TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
            return split if split is not None and len(split) else None

        g = torch.Generator().manual_seed(ctx.seed)
        nw = p.get("num_workers", 2) if data.modality != "tabular" else 0
        mk = lambda ds, shuffle: DataLoader(ds, batch_size=p.get("batch_size", 64), shuffle=shuffle, num_workers=nw,
                                            pin_memory=device.type == "cuda", generator=g if shuffle else None,
                                            persistent_workers=nw > 0) if ds is not None else None
        loaders = {s: mk(to_ds(getattr(data, s), train_w if s == "train" else None), s == "train") for s in ("train", "val", "test")}
        return info, device, loaders

    def _snapshot_inputs(self, data, ctx):
        """Fixed examples used for learning snapshots (same ones at every snapshot)."""
        import torch
        if data.modality == "image":
            src = data.val if data.val is not None and len(data.val) else data.test
            if src is None or not len(src):
                return None
            idx = torch.randperm(len(src), generator=torch.Generator().manual_seed(0))[:16].tolist()
            xs, ys = zip(*(src[i] for i in idx))
            return torch.stack(list(xs)), np.array(ys)
        X, y = data.val if len(data.val[0]) else data.train
        return X, y

    def _take_snapshot(self, model, data, ctx, step, snap, device, info, val_pred=None):
        import torch
        from ..training.snapshots import image_predictions, tabular_snapshot
        if snap is None:
            return
        model.eval()
        if data.modality == "image":
            xs, ys = snap
            with torch.no_grad():
                probs = torch.softmax(model(xs.to(device)).float(), -1).cpu().numpy()
            ctx.log_media(step, "predictions", "image_predictions", image_predictions(xs.numpy(), ys, probs, info.get("class_names")))
        else:
            X, y = snap

            def predict(A):
                with torch.no_grad():
                    out = model(torch.as_tensor(A, device=device)).float()
                return (out.argmax(1) if data.task == "classification" else out.squeeze(-1)).cpu().numpy()

            res = tabular_snapshot(data.task, predict, X, y, info)
            if res:
                ctx.log_media(step, res[0], res[0], res[1])
        if val_pred is not None and data.task == "classification":
            ctx.log_media(step, "val_confusion", "confusion_matrix", confusion(val_pred[0], val_pred[1], info.get("class_names")))

    @staticmethod
    def load_weights(model, path, device, ctx=None):
        """Load a state dict, skipping tensors whose shapes don't match (e.g. a new classifier head)."""
        import torch
        sd = torch.load(path, map_location=device, weights_only=True)
        own = model.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        skipped = [k for k in sd if k not in ok]
        model.load_state_dict(ok, strict=False)
        if ctx:
            ctx.print(f"loaded {len(ok)}/{len(own)} tensors from {path}" + (f"; re-initialized {skipped}" if skipped else ""))
        return model

    # -------------------------------------------------------------- training
    def train(self, data, ctx):
        import torch
        import torch.nn as nn

        p, task = ctx.params, data.task
        info, device, dl = self._prepare(data, ctx, p)
        train_dl, val_dl, test_dl = dl["train"], dl["val"], dl["test"]

        model = self.build(p, info).to(device)
        if ctx.init_path and not ctx.resume:
            self.load_weights(model, ctx.init_path / "model.pt", device, ctx)
        n_params = sum(x.numel() for x in model.parameters())
        ctx.print(model)
        ctx.print(f"parameters: {n_params:,}")
        opt = self.make_optimizer(model, p)
        sched = self.make_scheduler(opt, p)
        loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        # weighted training: per-example losses scaled by the sample weights
        loss_each = (nn.CrossEntropyLoss(reduction="none") if task == "classification" else nn.MSELoss(reduction="none"))
        use_amp = p["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        monitor = p["early_stopping_metric"] if val_dl is not None else "train/loss"
        stopper = EarlyStopping(p["early_stopping_patience"], metric_mode(monitor), p["early_stopping_min_delta"])
        best_path = ctx.checkpoint_path("best.pt")
        last_path = ctx.checkpoint_path("last.pt")
        start_epoch, train_time_before = 1, 0.0
        if ctx.resume and last_path.exists():
            ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
            if sched is not None and ck.get("sched"):
                sched.load_state_dict(ck["sched"])
            stopper.__dict__.update(ck["stopper"])
            start_epoch, train_time_before = ck["epoch"] + 1, ck.get("train_time", 0.0)
            ctx.truncate_after(ck["epoch"])
            ctx.print(f"resuming after epoch {ck['epoch']}")

        snap = self._snapshot_inputs(data, ctx) if ctx.snapshots else None
        snapshot_due = ctx.snapshot_schedule(p["epochs"])
        if start_epoch == 1 and snap is not None:
            self._take_snapshot(model, data, ctx, 0, snap, device, info)   # untrained baseline
        t0, epoch = time.time(), start_epoch - 1

        for epoch in range(start_epoch, p["epochs"] + 1):
            model.train()
            tot, n, preds, ys = 0.0, 0, [], []
            for i, batch in enumerate(train_dl):
                xb, yb = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=use_amp):
                    out = model(xb)
                    out = out.squeeze(-1) if task == "regression" else out
                    if len(batch) == 3:
                        wb = batch[2].to(device, non_blocking=True)
                        loss = (loss_each(out, yb) * wb).sum() / wb.sum()
                    else:
                        loss = loss_fn(out, yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if p["grad_clip"] > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(yb)
                n += len(yb)
                preds.append((out.argmax(1) if task == "classification" else out).detach().float().cpu())
                ys.append(yb.cpu())
                if i % 20 == 0:
                    ctx.set_progress((epoch - 1 + i / len(train_dl)) / p["epochs"], f"epoch {epoch}/{p['epochs']}")
                    if ctx.should_stop():
                        break
            if ctx.should_stop() and n < len(train_dl.dataset) * 0.5:
                epoch -= 1   # discard a mostly-incomplete epoch
                break
            row = {"train/loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"]}
            row.update(task_metrics(task, torch.cat(ys).numpy(), torch.cat(preds).numpy(), "train"))
            val_pred = None
            if val_dl is not None:
                vl, vy, vp = self._eval_loader(model, val_dl, loss_fn, task, device, use_amp)
                row["val/loss"] = vl
                row.update(task_metrics(task, vy, vp, "val"))
                val_pred = (vy, vp)
            ctx.log(epoch, row)
            ctx.print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4g}" for k, v in row.items()))
            if snapshot_due(epoch):
                self._take_snapshot(model, data, ctx, epoch, snap, device, info, val_pred)

            value = row.get(monitor)
            if value is None:
                raise ValueError(f"early_stopping_metric '{monitor}' not found; available: {sorted(row)}")
            if stopper.improved(value):
                torch.save(model.state_dict(), best_path)
            stop_early = stopper.step(value, epoch)
            if sched is not None:
                sched.step(value) if p["scheduler"] == "plateau" else sched.step()
            if ctx.opt("keep_checkpoint", True):
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                            "sched": sched.state_dict() if sched is not None else None, "stopper": dict(stopper.__dict__),
                            "epoch": epoch, "train_time": train_time_before + time.time() - t0}, last_path)
            if stop_early:
                ctx.print(f"early stopping at epoch {epoch} (best {monitor}={stopper.best:.4g} @ epoch {stopper.best_step})")
                break
            if ctx.should_stop():
                break

        train_time = train_time_before + time.time() - t0
        summary = {"epochs_trained": epoch, "best_epoch": stopper.best_step, f"best_{monitor}": stopper.best,
                   "n_params": n_params, "train_time_s": train_time}
        keep = ctx.keep
        restore = p["restore_best"] and best_path.exists() and keep != "final"
        if keep == "both" or (keep == "none" and restore):
            # score the last-epoch weights too, so best-vs-final (overfitting) is visible
            fin = self._final_eval(model, dl, loss_fn, task, device, use_amp, info, ctx, ("val", "test"), artifacts=False)
            summary.update({f"final_model/{k}": v for k, v in fin.items()})
            if keep == "both":
                torch.save(model.state_dict(), ctx.final_path("model.pt"))
        if restore:
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        summary.update(self._final_eval(model, dl, loss_fn, task, device, use_amp, info, ctx, ("val", "test")))
        if ctx.save_model:
            torch.save(model.state_dict(), ctx.model_path("model.pt"))
        return summary

    def _final_eval(self, model, dl, loss_fn, task, device, use_amp, info, ctx, splits, artifacts=True):
        out = {}
        for split in splits:
            if dl.get(split) is None:
                continue
            l, y, pr = self._eval_loader(model, dl[split], loss_fn, task, device, use_amp)
            out[f"{split}/loss"] = l
            out.update(task_metrics(task, y, pr, split))
            if split == "test" and task == "classification" and artifacts:
                ctx.log_artifact("confusion_matrix_test", "confusion_matrix", confusion(y, pr, info.get("class_names")))
        return out

    # -------------------------------------------------------------- evaluation only
    def evaluate(self, data, ctx):
        import torch.nn as nn
        meta = json.loads((ctx.init_path / "meta.json").read_text())
        p = {**meta["model_params"], **{k: v for k, v in ctx.params.items() if k in ("batch_size", "num_workers")}}
        info, device, dl = self._prepare(data, ctx, p)
        model = self.build(meta["model_params"], {**info, **{k: v for k, v in meta["info"].items() if k == "input_shape"}}).to(device)
        self.load_weights(model, ctx.init_path / "model.pt", device, ctx)
        loss_fn = nn.CrossEntropyLoss() if data.task == "classification" else nn.MSELoss()
        splits = ctx.opt("eval_splits", "val+test").split("+")
        ctx.set_progress(0.3, "evaluating")
        summary = self._final_eval(model, dl, loss_fn, data.task, device, False, info, ctx, splits)
        if ctx.snapshots:
            self._take_snapshot(model, data, ctx, 0, self._snapshot_inputs(data, ctx), device, info)
        summary["n_params"] = sum(x.numel() for x in model.parameters())
        return summary

    @staticmethod
    def _eval_loader(model, dl, loss_fn, task, device, use_amp):
        import torch
        model.eval()
        tot, n, preds, ys = 0.0, 0, [], []
        with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                out = out.squeeze(-1) if task == "regression" else out
                tot += loss_fn(out, yb).item() * len(yb)
                n += len(yb)
                preds.append((out.argmax(1) if task == "classification" else out).float().cpu())
                ys.append(yb.cpu())
        return tot / max(n, 1), torch.cat(ys).numpy(), torch.cat(preds).numpy()

    @staticmethod
    def make_optimizer(model, p):
        import torch
        params, lr, wd = model.parameters(), p["lr"], p["weight_decay"]
        if p["optimizer"] == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=p["momentum"], weight_decay=wd)
        if p["optimizer"] == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, momentum=p["momentum"], weight_decay=wd)
        if p["optimizer"] == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    @staticmethod
    def make_scheduler(opt, p):
        import torch
        s = p["scheduler"]
        if s == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
        if s == "step":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, p["epochs"] // 3), gamma=0.1)
        if s == "plateau":
            mode = metric_mode(p["early_stopping_metric"])
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.5, patience=3)
        return None


def _act(name):
    import torch.nn as nn
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU, "leaky_relu": nn.LeakyReLU}[name]()


ACTS = ["relu", "gelu", "silu", "tanh", "leaky_relu"]


@register
class MLP(TorchModel):
    id = "torch.mlp"
    name = "MLP (PyTorch)"
    description = "Fully connected network; images are flattened"

    def model_params(self):
        return [Param("hidden_sizes", "int_list", [256, 128], "Units per hidden layer, e.g. 256,128"),
                Param("activation", "choice", "relu", choices=ACTS),
                Param("dropout", "float", 0.1, min=0.0, max=0.95),
                Param("batch_norm", "bool", False)]

    def build(self, p, info):
        import torch.nn as nn
        layers, d = [nn.Flatten()], info["input_dim"]
        for h in p["hidden_sizes"]:
            layers += [nn.Linear(d, h)] + ([nn.BatchNorm1d(h)] if p["batch_norm"] else []) + [_act(p["activation"])]
            if p["dropout"] > 0:
                layers.append(nn.Dropout(p["dropout"]))
            d = h
        layers.append(nn.Linear(d, info["output_dim"]))
        return nn.Sequential(*layers)


@register
class CNN(TorchModel):
    id = "torch.cnn"
    name = "CNN (PyTorch)"
    description = "Conv blocks (conv-[bn]-act-pool) followed by an MLP head"
    modalities = ("image",)

    def model_params(self):
        return [Param("conv_channels", "int_list", [32, 64, 128], "Channels per conv block"),
                Param("kernel_size", "int", 3, min=1),
                Param("fc_sizes", "int_list", [256], "Hidden units in the classifier head"),
                Param("activation", "choice", "relu", choices=ACTS),
                Param("dropout", "float", 0.25, min=0.0, max=0.95),
                Param("batch_norm", "bool", True)]

    def build(self, p, info):
        import torch.nn as nn
        c, h, w = info["input_shape"]
        layers, k = [], p["kernel_size"]
        for ch in p["conv_channels"]:
            layers += [nn.Conv2d(c, ch, k, padding=k // 2)] + ([nn.BatchNorm2d(ch)] if p["batch_norm"] else [])
            layers += [_act(p["activation"])]
            if min(h, w) >= 2:
                layers.append(nn.MaxPool2d(2))
                h, w = h // 2, w // 2
            c = ch
        layers += [nn.Flatten()]
        d = c * h * w
        for f in p["fc_sizes"]:
            layers += [nn.Linear(d, f), _act(p["activation"]), nn.Dropout(p["dropout"])]
            d = f
        layers.append(nn.Linear(d, info["output_dim"]))
        return nn.Sequential(*layers)


@register
class TorchvisionModel(TorchModel):
    id = "torch.torchvision"
    name = "Torchvision architecture"
    description = "ResNet / MobileNet / EfficientNet / ConvNeXt from torchvision, optional ImageNet weights"
    modalities = ("image",)

    def model_params(self):
        return [Param("arch", "choice", "resnet18", choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_small",
                                                               "efficientnet_b0", "convnext_tiny"]),
                Param("pretrained", "bool", False, "Load ImageNet weights (downloads once)"),
                Param("small_input_stem", "bool", True, "ResNet only: 3x3 stem without max-pool, for 32x32-ish images"),
                Param("freeze_backbone", "bool", False, "Train only the final layer")]

    def build(self, p, info):
        import torch.nn as nn
        import torchvision.models as tvm
        weights = "DEFAULT" if p["pretrained"] else None
        m = getattr(tvm, p["arch"])(weights=weights)
        if p["freeze_backbone"]:
            for prm in m.parameters():
                prm.requires_grad = False
        out = info["output_dim"]
        if p["arch"].startswith("resnet"):
            if p["small_input_stem"]:
                m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                m.maxpool = nn.Identity()
            m.fc = nn.Linear(m.fc.in_features, out)
        else:
            last = m.classifier[-1]
            m.classifier[-1] = nn.Linear(last.in_features, out)
        if info["input_shape"][0] != 3:   # grayscale -> repeat to 3 channels
            return nn.Sequential(_Repeat3(), m)
        return m


def _Repeat3():
    import torch.nn as nn

    class Repeat3(nn.Module):
        def forward(self, x):
            return x.expand(-1, 3, -1, -1)

    return Repeat3()


def wrap_module(module_cls, path: Path) -> TorchModel:
    """Wrap a plain user nn.Module so it trains with TorchModel's loop."""
    from ..plugins.introspect import assemble, extract, own_doc
    mparams = extract(module_cls)

    class Wrapped(TorchModel):
        id = f"custom.{path.stem}.{module_cls.__name__}"
        name = module_cls.__name__
        description = (own_doc(module_cls) or f"Custom nn.Module from {path.name}").split("\n")[0]
        module_class = module_cls
        tasks = tuple(getattr(module_cls, "TASKS", ("classification", "regression")))
        modalities = tuple(getattr(module_cls, "MODALITIES", ("tabular", "image")))

        def model_params(self):
            return mparams

        def build(self, p, info):
            inject = {"input_dim": info["input_dim"], "input_shape": tuple(info["input_shape"]),
                      "in_channels": info["input_shape"][0], "in_ch": info["input_shape"][0],
                      "num_classes": info["output_dim"], "n_classes": info["output_dim"], "output_dim": info["output_dim"]}
            return module_cls(**assemble(module_cls, p, inject))

    return Wrapped()

