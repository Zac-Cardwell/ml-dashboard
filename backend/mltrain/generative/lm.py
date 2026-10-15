"""Text generation: autoregressive language models (built-in GPT, your own `forward(idx) -> logits` module, or a
pretrained Hugging Face causal LM to fine-tune). Character-level or GPT-2 BPE tokenization."""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..plugins.base import Param
from ..plugins.registry import register
from . import losses as L
from ..plugins.introspect import assemble, own_doc
from .core import GenerativeModel
from .nets import GPT


# ============================================================================= tokenizers
class CharTokenizer:
    def __init__(self, vocab: str):
        self.vocab = vocab
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.size = len(vocab)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.vocab[i] for i in ids if 0 <= i < self.size)

    def spec(self):
        return {"kind": "char", "vocab": self.vocab}


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        from .. import config
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=str(config.CACHE_DIR / "hf_models"))
        self.size = len(self.tok)

    def encode(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def spec(self):
        return {"kind": "hf", "name": self.name}


def tokenizer_from_spec(spec: dict):
    return CharTokenizer(spec["vocab"]) if spec["kind"] == "char" else HFTokenizer(spec["name"])


def as_text(split) -> str:
    """Dataset split -> one string (lists of {'text': ...} / strings are joined with blank lines)."""
    if split is None:
        return ""
    if isinstance(split, str):
        return split
    parts = []
    for r in split:
        parts.append(r["text"] if isinstance(r, dict) else str(r))
    return "\n\n".join(parts)


class Windows(torch.utils.data.Dataset):
    """Random (train) or tiled (eval) windows of block_size + 1 tokens."""

    def __init__(self, ids: torch.Tensor, block: int, random_windows: bool, n: int | None = None):
        self.ids, self.block, self.random = ids, block, random_windows
        self.n = n or max(1, (len(ids) - 1) // block)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.random:
            j = int(torch.randint(0, max(1, len(self.ids) - self.block - 1), (1,)))
        else:
            j = min(i * self.block, max(0, len(self.ids) - self.block - 1))
        return (self.ids[j:j + self.block + 1],)


# ============================================================================= process
class LMProcess(nn.Module):
    def __init__(self, net: nn.Module, block_size: int, loss="cross_entropy", hf=False, temperature=0.8, top_k=40,
                 sample_len=200, prompts=("",), tokenizer=None):
        super().__init__()
        self.net, self.block, self.hf = net, block_size, hf
        self.loss_name = loss
        self.loss_fn = None if loss == "cross_entropy" else L.get(loss)
        self.temperature, self.top_k, self.sample_len, self.prompts = temperature, top_k, sample_len, list(prompts) or [""]
        self.tok = tokenizer

    def logits(self, idx):
        out = self.net(input_ids=idx) if self.hf else self.net(idx)
        return out.logits if hasattr(out, "logits") else out

    def losses(self, x, y=None, generator=None):
        logits = self.logits(x[:, :-1]).float()
        tgt = x[:, 1:]
        if self.loss_fn is None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        else:
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, prompt: str, length: int, temperature=None, top_k=None, generator=None) -> str:
        dev = next(self.parameters()).device
        ids = self.tok.encode(prompt) or [0]
        x = torch.tensor([ids], device=dev)
        temperature = self.temperature if temperature is None else temperature
        top_k = self.top_k if top_k is None else top_k
        for _ in range(length):
            logits = self.logits(x[:, -self.block:])[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            x = torch.cat([x, nxt], 1)
        return self.tok.decode(x[0, len(ids):].tolist())

    def sample(self, n, y=None, generator=None):
        return [self.generate(self.prompts[i % len(self.prompts)], self.sample_len, generator=generator) for i in range(n)]


LM_PARAMS = [
    Param("tokenizer", "choice", "char", "char = one token per character (vocabulary from the training text); "
          "gpt2 = GPT-2 BPE (ignored for Hugging Face models, which use their own)", choices=["char", "gpt2"], group="Text"),
    Param("block_size", "int", 128, "Context length in tokens", 8, 4096, group="Text"),
    Param("loss", "str", "cross_entropy", "cross_entropy, or your own loss(logits (N,V), targets (N,))",
          suggestions=["cross_entropy"], group="Text"),
    Param("prompts", "str", "", "Prompts for sample snapshots, separated by |  (empty = unconditional)", group="Sampling"),
    Param("sample_length", "int", 200, "Tokens generated per sample", 1, 4096, group="Sampling"),
    Param("temperature", "float", 0.8, "Sampling temperature (0 = greedy)", 0.0, 5.0, group="Sampling"),
    Param("top_k", "int", 40, "Sample from the k most likely tokens (0 = all)", 0, group="Sampling"),
    Param("max_chars", "int", 0, "Use only the first N characters of the training text (0 = all)", 0, group="Text"),
]
TEXT_TRAIN_OVERRIDES = {"lr": 3e-4, "batch_size": 32, "epochs": 10, "ema_decay": 0.0, "n_samples": 4, "eval_samples": 0}


class LanguageModel(GenerativeModel):
    modalities = ("text",)

    def method_params(self):
        return LM_PARAMS + [Param("windows_per_epoch", "int", 2000, "Random training windows per epoch", 1, group="Text")]

    def params(self):
        from dataclasses import replace
        ps = [q for q in super().params() if q.name != "image_size"]
        return [replace(q, default=TEXT_TRAIN_OVERRIDES[q.name]) if q.name in TEXT_TRAIN_OVERRIDES else q for q in ps]

    hf_name: str | None = None

    def make_net(self, p, info, vocab_size) -> nn.Module:
        raise NotImplementedError

    def prepare(self, data, p, ctx):
        from torch.utils.data import DataLoader
        info = dict(data.info)
        info["modality"] = "text"
        train_text = as_text(data.train)
        if p.get("max_chars"):
            train_text = train_text[: p["max_chars"]]
        val_text, test_text = as_text(data.val), as_text(data.test)
        if self.hf_name or (p.get("network") or "").startswith("hf:"):
            tok = HFTokenizer(self.hf_name or p["network"][3:])
        elif ctx.init_path and (ctx.init_path / "generation.json").exists():
            import json
            tok = tokenizer_from_spec(json.loads((ctx.init_path / "generation.json").read_text())["tokenizer"])
        elif p["tokenizer"] == "gpt2":
            tok = HFTokenizer("gpt2")
        else:
            tok = CharTokenizer("".join(sorted(set(train_text + val_text + test_text))))
        enc = lambda s: torch.tensor(tok.encode(s), dtype=torch.long)
        ids = {"train": enc(train_text), "val": enc(val_text), "test": enc(test_text)}
        B = p["block_size"]
        info.update(tokenizer=tok.spec(), vocab_size=tok.size, block_size=B, data_shape=[B], n_train_tokens=len(ids["train"]))
        self._tok = tok
        kw = dict(batch_size=p["batch_size"], num_workers=0)
        dl = {"train": DataLoader(Windows(ids["train"], B, True, p["windows_per_epoch"]), **kw)}
        for s in ("val", "test"):
            dl[s] = DataLoader(Windows(ids[s], B, False), shuffle=False, **kw) if len(ids[s]) > B + 1 else None
        to_model = lambda batch, device: (batch[0].to(device), None)
        return info, dl, to_model

    def build(self, p, info):
        prompts = [s for s in (p.get("prompts") or "").split("|")] or [""]
        net = self.make_net(p, info, info["vocab_size"])
        hf = bool(self.hf_name or (p.get("network") or "").startswith("hf:"))
        return LMProcess(net, info["block_size"], p["loss"], hf, p["temperature"], p["top_k"], p["sample_length"], prompts, self._tok)

    def log_samples(self, net, ctx, step, info, device, name="generated", trajectory=False, x_real=None):
        g = torch.Generator(device=device).manual_seed(int(ctx.seed) + 9973)
        n = ctx.params.get("n_samples", 4)
        texts = net.sample(n, generator=g)
        prompts = net.prompts
        ctx.log_media(step, name, "text_samples", {"columns": ["prompt", "prediction"],
                                                   "rows": [[prompts[i % len(prompts)] or "(none)", t] for i, t in enumerate(texts)]})

    def sample_quality(self, net, info, loader, to_model, device, n_eval, ctx, prefix="test"):
        return {}

    def _finish(self, model, info, dl, to_model, device, ctx, use_amp, x_real):
        out = super()._finish(model, info, dl, to_model, device, ctx, use_amp, x_real)
        if "test/loss" in out:
            out["test/perplexity"] = math.exp(min(out["test/loss"], 50))
        return out

    def _eval_losses(self, net, loader, to_model, device, max_batches=50, use_amp=False):
        out = super()._eval_losses(net, loader, to_model, device, max_batches, use_amp)
        if "loss" in out:
            out["perplexity"] = math.exp(min(out["loss"], 50))
        return out


@register
class BuiltinLM(LanguageModel):
    id = "gen.lm"
    name = "Language model · GPT (built-in)"
    description = "Decoder-only transformer trained from scratch (char-level or GPT-2 BPE), or fine-tune a Hugging Face model"

    def network_params(self):
        return [Param("network", "str", "gpt", "gpt = built-in transformer; hf:<model> fine-tunes a pretrained causal LM",
                      suggestions=["gpt", "hf:distilgpt2", "hf:gpt2", "hf:EleutherAI/pythia-70m", "hf:HuggingFaceTB/SmolLM2-135M"],
                      group="Network"),
                Param("n_layer", "int", 4, "Transformer blocks (built-in)", 1, 48, group="Network"),
                Param("n_head", "int", 4, "Attention heads (built-in)", 1, 64, group="Network"),
                Param("n_embd", "int", 256, "Embedding width (built-in; divisible by n_head)", 16, 4096, group="Network"),
                Param("dropout", "float", 0.1, min=0.0, max=0.9, group="Network")]

    def make_net(self, p, info, vocab_size):
        if (p.get("network") or "gpt").startswith("hf:"):
            from transformers import AutoModelForCausalLM

            from .. import config
            return AutoModelForCausalLM.from_pretrained(p["network"][3:], cache_dir=str(config.CACHE_DIR / "hf_models"))
        return GPT(vocab_size, p["block_size"], p["n_layer"], p["n_head"], p["n_embd"], p["dropout"])


def lm_for(module_cls, path, net_params):
    """A language-model plugin around a user module with forward(idx) -> logits (B, L, vocab)."""
    sig = inspect.signature(module_cls.__init__).parameters

    class _L(LanguageModel):
        id = f"gen.lm.{path.stem}.{module_cls.__name__}"
        name = f"Language model · {module_cls.__name__}"
        description = (own_doc(module_cls) or f"Your language model from {path.name}").split("\n")[0]
        network_class = module_cls

        def network_params(self):
            return net_params

        def make_net(self, p, info, vocab_size):
            inj = {n: vocab_size for n in ("vocab_size", "n_vocab", "num_tokens")}
            inj.update({n: p["block_size"] for n in ("block_size", "max_len", "context_length", "seq_len")})
            return module_cls(**assemble(module_cls, p, inj))

    _L.__name__ = f"LM_{module_cls.__name__}"
    return _L()
