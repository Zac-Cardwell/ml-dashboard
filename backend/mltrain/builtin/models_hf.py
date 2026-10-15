"""Hugging Face transformers: text classification fine-tuning and seq2seq (translation/summarization)."""
from __future__ import annotations

import math
import time

import numpy as np

from .. import config
from ..plugins.base import ModelPlugin, Param
from ..plugins.registry import register
from ..training.utils import EarlyStopping, confusion, metric_mode, task_metrics

HF_TRAIN = [
    Param("epochs", "int", 3, min=1, group="Training"),
    Param("batch_size", "int", 16, min=1, group="Training"),
    Param("grad_accum_steps", "int", 1, "Gradient accumulation steps", 1, group="Training"),
    Param("lr", "float", 5e-5, "Learning rate", 1e-8, log=True, group="Training"),
    Param("weight_decay", "float", 0.01, min=0.0, group="Training"),
    Param("warmup_ratio", "float", 0.06, "Fraction of steps for linear warmup", 0.0, 0.5, group="Training"),
    Param("amp", "bool", True, "bf16/fp16 mixed precision on GPU", group="Training"),
    Param("eval_every_steps", "int", 0, "Evaluate every N optimizer steps (0 = once per epoch)", 0, group="Training"),
    Param("log_every_steps", "int", 20, min=1, group="Training"),
    Param("early_stopping_patience", "int", 0, "Evaluations without improvement before stopping (0 = off)", 0, group="Early stopping"),
]


class HFBase(ModelPlugin):
    framework = "hf"
    modalities = ("text",)
    uses_gpu = True
    supports_init = True
    supports_eval = True

    # subclasses implement
    def load_model(self, p, info, source: str): ...
    def collate(self, tok, batch, p): ...
    def run_eval(self, model, tok, rows, p, device, info, split) -> tuple[dict, dict]: ...
    def samples(self, model, tok, rows, p, device, info) -> dict: ...

    def _snapshot(self, model, tok, data, p, device, info, ctx, step):
        if ctx.snapshots and len(data.val):
            ctx.log_media(step, "samples", "text_samples", self.samples(model, tok, data.val[:8], p, device, info))
            model.train()

    def evaluate(self, data, ctx):
        import torch
        p, info = ctx.params, data.info
        device = torch.device(ctx.device)
        source = str(ctx.init_path) if ctx.init_path else p["model_name"]
        tok, model = self.load_model(p, info, source)
        model.to(device)
        ctx.print(f"evaluating {source}")
        summary = {"n_params": sum(x.numel() for x in model.parameters())}
        splits = ctx.opt("eval_splits", "val+test").split("+")
        for i, split in enumerate(splits):
            rows = getattr(data, split)
            if len(rows):
                ctx.set_progress(i / len(splits), f"evaluating {split}")
                m, artifacts = self.run_eval(model, tok, rows, p, device, info, split)
                summary.update(m)
                for name, (kind, payload) in artifacts.items():
                    ctx.log_artifact(name, kind, payload)
        self._snapshot(model, tok, data, p, device, info, ctx, 0)
        return summary

    monitor = "val/loss"

    def train(self, data, ctx):
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        p, info = ctx.params, data.info
        device = torch.device(ctx.device)
        source = str(ctx.init_path) if ctx.init_path else p["model_name"]
        tok, model = self.load_model(p, info, source)
        model.to(device)
        self._snapshot(model, tok, data, p, device, info, ctx, 0)   # what the model does before training
        n_params = sum(x.numel() for x in model.parameters())
        ctx.print(f"{p['model_name']}: {n_params:,} parameters; train={len(data.train)} val={len(data.val)} test={len(data.test)}")

        g = torch.Generator().manual_seed(ctx.seed)
        dl = DataLoader(data.train, batch_size=p["batch_size"], shuffle=True, generator=g,
                        collate_fn=lambda b: self.collate(tok, b, p))
        steps_per_epoch = math.ceil(len(dl) / p["grad_accum_steps"])
        total_steps = steps_per_epoch * p["epochs"]
        opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * p["warmup_ratio"]), total_steps)
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        use_amp = p["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

        monitor = p.get("monitor_metric") or self.monitor
        stopper = EarlyStopping(p["early_stopping_patience"], metric_mode(monitor))
        best_dir = ctx.checkpoint_dir / "best"
        step, running, t0, stop = 0, [], time.time(), False

        def do_eval():
            nonlocal stop
            m, _ = self.run_eval(model, tok, data.val, p, device, info, "val")
            ctx.log(step, m)
            ctx.print(f"step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
            self._snapshot(model, tok, data, p, device, info, ctx, step)
            if monitor in m:
                if stopper.improved(m[monitor]):
                    model.save_pretrained(best_dir)
                    tok.save_pretrained(best_dir)
                if stopper.step(m[monitor], step):
                    ctx.print("early stopping")
                    stop = True
            model.train()

        model.train()
        for epoch in range(1, p["epochs"] + 1):
            for i, batch in enumerate(dl):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    loss = model(**batch).loss / p["grad_accum_steps"]
                scaler.scale(loss).backward()
                running.append(loss.item() * p["grad_accum_steps"])
                if (i + 1) % p["grad_accum_steps"] and i + 1 != len(dl):
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % p["log_every_steps"] == 0:
                    ctx.log(step, {"train/loss": float(np.mean(running)), "lr": sched.get_last_lr()[0], "epoch": epoch})
                    running = []
                    ctx.set_progress(step / total_steps, f"epoch {epoch}/{p['epochs']} · step {step}/{total_steps}")
                if p["eval_every_steps"] and step % p["eval_every_steps"] == 0 and len(data.val):
                    do_eval()
                if stop or ctx.should_stop():
                    break
            if not p["eval_every_steps"] and len(data.val) and not ctx.should_stop():
                do_eval()
            if stop or ctx.should_stop():
                break

        summary = {"steps_trained": step, "epochs_trained": epoch, "n_params": n_params, "train_time_s": time.time() - t0,
                   f"best_{monitor}": stopper.best}
        keep = ctx.keep
        if best_dir.exists() and keep != "final":
            if keep == "both":
                for split, rows in (("test", data.test),):
                    if len(rows):
                        m, _ = self.run_eval(model, tok, rows, p, device, info, split)
                        summary.update({f"final_model/{k}": v for k, v in m.items()})
                model.save_pretrained(ctx.final_dir)
                tok.save_pretrained(ctx.final_dir)
            model = type(model).from_pretrained(best_dir).to(device)
        for split, rows in (("val", data.val), ("test", data.test)):
            if len(rows):
                m, artifacts = self.run_eval(model, tok, rows, p, device, info, split)
                summary.update(m)
                for name, (kind, payload) in artifacts.items():
                    ctx.log_artifact(name, kind, payload)
        if ctx.save_model:
            model.save_pretrained(ctx.model_dir)
            tok.save_pretrained(ctx.model_dir)
        return summary


@register
class HFTextClassifier(HFBase):
    id = "hf.text_classifier"
    name = "Transformer text classifier"
    description = "Fine-tune any Hugging Face encoder (DistilBERT, BERT, RoBERTa, ...) for classification"
    tasks = ("classification",)
    monitor = "val/loss"

    def params(self):
        return [Param("model_name", "str", "distilbert-base-uncased", "Hugging Face model id",
                      suggestions=["distilbert-base-uncased", "bert-base-uncased", "roberta-base",
                                   "microsoft/deberta-v3-small", "google/bert_uncased_L-2_H-128_A-2"]),
                Param("max_length", "int", 128, "Max tokens per example", 8, 4096),
                Param("dropout", "float", 0.1, "Classifier dropout", 0.0, 0.9),
                Param("monitor_metric", "str", "val/loss", "Metric for best checkpoint / early stopping",
                      suggestions=["val/loss", "val/acc", "val/f1"])] + HF_TRAIN

    def load_model(self, p, info, source):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        cache = str(config.CACHE_DIR / "hf_models")
        tok = AutoTokenizer.from_pretrained(source, cache_dir=cache)
        names = info["class_names"]
        cfg = AutoConfig.from_pretrained(source, num_labels=len(names), cache_dir=cache,
                                         id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)})
        for attr in ("classifier_dropout", "seq_classif_dropout"):  # name differs per architecture
            if hasattr(cfg, attr):
                setattr(cfg, attr, p["dropout"])
        # ignore_mismatched_sizes: a saved model with a different number of classes gets a fresh head
        model = AutoModelForSequenceClassification.from_pretrained(source, config=cfg, cache_dir=cache,
                                                                   ignore_mismatched_sizes=True)
        return tok, model

    def collate(self, tok, batch, p):
        import torch
        enc = tok([b["text"] for b in batch], truncation=True, max_length=p["max_length"], padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return dict(enc)

    def samples(self, model, tok, rows, p, device, info):
        import torch
        model.eval()
        batch = {k: v.to(device) for k, v in self.collate(tok, rows, p).items()}
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits.float(), -1).cpu().numpy()
        names = info["class_names"]
        return {"columns": ["text", "label", "prediction", "confidence"],
                "rows": [[r["text"][:300], names[r["label"]], names[int(pr.argmax())], round(float(pr.max()), 3)]
                         for r, pr in zip(rows, probs)]}

    def run_eval(self, model, tok, rows, p, device, info, split):
        import torch
        model.eval()
        losses, preds, ys = [], [], []
        with torch.no_grad():
            for i in range(0, len(rows), p["batch_size"] * 2):
                batch = {k: v.to(device) for k, v in self.collate(tok, rows[i:i + p["batch_size"] * 2], p).items()}
                out = model(**batch)
                losses.append(out.loss.item() * len(batch["labels"]))
                preds.append(out.logits.argmax(-1).cpu())
                ys.append(batch["labels"].cpu())
        y, pr = torch.cat(ys).numpy(), torch.cat(preds).numpy()
        m = {f"{split}/loss": sum(losses) / len(y), **task_metrics("classification", y, pr, split)}
        arts = {}
        if split == "test":
            arts["confusion_matrix_test"] = ("confusion_matrix", confusion(y, pr, info.get("class_names")))
        return m, arts


@register
class HFSeq2Seq(HFBase):
    id = "hf.seq2seq"
    name = "Transformer seq2seq"
    description = "Fine-tune T5 / BART / Marian etc. for translation or summarization; reports BLEU, chrF, ROUGE"
    tasks = ("seq2seq",)
    monitor = "val/bleu"
    eval_without_init = True   # pretrained checkpoints (T5, Marian, ...) can be scored zero-shot

    def params(self):
        return [Param("model_name", "str", "google-t5/t5-small", "Hugging Face model id",
                      suggestions=["google-t5/t5-small", "google-t5/t5-base", "facebook/bart-base",
                                   "Helsinki-NLP/opus-mt-en-fr", "google/flan-t5-small"]),
                Param("max_source_length", "int", 128, min=8),
                Param("max_target_length", "int", 128, min=8),
                Param("num_beams", "int", 4, "Beam size for generation during evaluation", 1),
                Param("max_eval_generate", "int", 300, "Examples to generate for BLEU/ROUGE per evaluation", 1),
                Param("monitor_metric", "str", "val/bleu", "Metric for best checkpoint / early stopping",
                      suggestions=["val/bleu", "val/loss", "val/rougeL", "val/chrf"])] + HF_TRAIN

    def load_model(self, p, info, source):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        cache = str(config.CACHE_DIR / "hf_models")
        return (AutoTokenizer.from_pretrained(source, cache_dir=cache),
                AutoModelForSeq2SeqLM.from_pretrained(source, cache_dir=cache))

    def samples(self, model, tok, rows, p, device, info):
        import torch
        model.eval()
        enc = tok([r["source"] for r in rows], truncation=True, max_length=p["max_source_length"], padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=p["num_beams"], max_new_tokens=p["max_target_length"])
        hyps = tok.batch_decode(out, skip_special_tokens=True)
        return {"columns": ["source", "reference", "prediction"], "rows": [[r["source"], r["target"], h] for r, h in zip(rows, hyps)]}

    def collate(self, tok, batch, p):
        enc = tok([b["source"] for b in batch], truncation=True, max_length=p["max_source_length"], padding=True, return_tensors="pt")
        labels = tok(text_target=[b["target"] for b in batch], truncation=True, max_length=p["max_target_length"],
                     padding=True, return_tensors="pt")["input_ids"]
        labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return dict(enc)

    def run_eval(self, model, tok, rows, p, device, info, split):
        import sacrebleu
        import torch
        from rouge_score import rouge_scorer
        model.eval()
        bs = p["batch_size"] * 2
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                batch = {k: v.to(device) for k, v in self.collate(tok, rows[i:i + bs], p).items()}
                loss_sum += model(**batch).loss.item() * len(batch["labels"])
                n += len(batch["labels"])
            gen_rows = rows[: p["max_eval_generate"]]
            hyps = []
            for i in range(0, len(gen_rows), bs):
                enc = tok([r["source"] for r in gen_rows[i:i + bs]], truncation=True, max_length=p["max_source_length"],
                          padding=True, return_tensors="pt").to(device)
                out = model.generate(**enc, num_beams=p["num_beams"], max_new_tokens=p["max_target_length"])
                hyps += tok.batch_decode(out, skip_special_tokens=True)
        refs = [r["target"] for r in gen_rows]
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rs = [scorer.score(r, h) for r, h in zip(refs, hyps)]
        m = {f"{split}/loss": loss_sum / max(n, 1),
             f"{split}/bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
             f"{split}/chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
             **{f"{split}/{k}": float(np.mean([s[k].fmeasure for s in rs])) * 100 for k in ("rouge1", "rouge2", "rougeL")}}
        arts = {f"samples_{split}": ("table", {"columns": ["source", "reference", "prediction"],
                                              "rows": [[r["source"], r["target"], h] for r, h in zip(gen_rows[:25], hyps[:25])]})}
        return m, arts
