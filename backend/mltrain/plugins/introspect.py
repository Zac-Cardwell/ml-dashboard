"""Turn a model class's constructor into form fields, and form values back into constructor arguments.

What is read (no changes to your code needed):
  * defaults                        -> pre-filled values
  * type hints                      -> field types (int, float, bool, str, list/tuple of ints, Literal[...] -> dropdown)
  * docstring "Args:" / "Parameters" sections (Google / NumPy / reST) -> help text
  * trailing "# comments" on the constructor's argument lines -> help text
  * dataclass arguments (a config object) -> one field per dataclass field (named cfg.field)
  * arguments without defaults      -> filled from the dataset when the name is recognized (in_channels, num_classes,
                                       vocab_size, image size...), otherwise shown as required fields
"""
from __future__ import annotations

import dataclasses
import inspect
import io
import re
import tokenize
import typing

from .base import Param

INJECTED = {"in_ch", "in_channels", "input_channels", "channels", "out_ch", "out_channels", "output_channels", "n_channels",
            "img_size", "image_size", "resolution", "dims", "length", "seq_len", "audio_length", "num_classes",
            "n_classes", "vocab_size", "n_vocab", "num_tokens", "block_size", "max_len", "context_length",
            "input_dim", "input_shape", "output_dim", "data_shape"}
SKIP = {"self", "args", "kwargs"}


def own_doc(cls) -> str:
    """The class's own docstring (not one inherited from nn.Module)."""
    d = vars(cls).get("__doc__") if isinstance(cls, type) else getattr(cls, "__doc__", None)
    return inspect.cleandoc(d) if d else ""


def doc_args(*objs) -> dict[str, str]:
    """{arg: description} from Google ('Args:'), NumPy ('Parameters\\n----------') or reST (':param x:') docstrings."""
    out: dict[str, str] = {}
    for obj in objs:
        doc = inspect.getdoc(obj) or ""
        for m in re.finditer(r":param\s+(?:\w+\s+)?(\w+):\s*(.+)", doc):
            out.setdefault(m.group(1), m.group(2).strip())
        lines = doc.splitlines()
        mode = None
        for i, line in enumerate(lines):
            s = line.strip()
            if re.match(r"^(Args|Arguments|Parameters|Params|Keyword Args):?$", s):
                mode = "google" if s.endswith(":") else "numpy"
                continue
            if mode and re.match(r"^(Returns|Raises|Yields|Examples?|Notes?|Attributes|See Also)\b", s):
                mode = None
            if mode == "google":
                m = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", line)
                if m:
                    out.setdefault(m.group(1), m.group(2).strip())
            elif mode == "numpy":
                m = re.match(r"^(\w+)\s*:\s*\S.*$", s)
                if m and i + 1 < len(lines) and lines[i + 1].startswith((" ", "\t")):
                    out.setdefault(m.group(1), lines[i + 1].strip())
    return out


def comment_args(fn) -> dict[str, str]:
    """{arg: comment} for arguments followed by `# comment` on the same line of the def."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return {}
    out, last_name, depth = {}, None, 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.OP and tok.string == "(":
                depth += 1
            elif tok.type == tokenize.OP and tok.string == ")":
                depth -= 1
                if depth == 0:
                    break
            elif tok.type == tokenize.NAME and depth == 1 and tok.string not in ("self", "None", "True", "False"):
                if last_name is None or tok.start[0] != last_name[1]:
                    last_name = (tok.string, tok.start[0])
            elif tok.type == tokenize.COMMENT and last_name and tok.start[0] == last_name[1]:
                out[last_name[0]] = tok.string.lstrip("# ").strip()
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def _hints(fn) -> dict:
    try:
        return typing.get_type_hints(fn)
    except Exception:
        return getattr(fn, "__annotations__", {}) or {}


def _param_for(name: str, ann, default, help_: str, group: str) -> Param | None:
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if origin is typing.Union and type(None) in args:            # Optional[X]
        rest = [a for a in args if a is not type(None)]
        ann = rest[0] if len(rest) == 1 else ann
        origin, args = typing.get_origin(ann), typing.get_args(ann)
    required = default is inspect.Parameter.empty
    d = None if required else default
    if required:
        help_ = (help_ + " · " if help_ else "") + "required"
    if origin is typing.Literal:
        return Param(name, "choice", d if d is not None else args[0], help_, choices=list(args), group=group)
    if ann is bool or isinstance(d, bool):
        return Param(name, "bool", bool(d) if d is not None else False, help_, group=group)
    if ann is int or (isinstance(d, int) and not isinstance(d, bool)):
        return Param(name, "int", d, help_, group=group)
    if ann is float or isinstance(d, float):
        return Param(name, "float", d, help_, group=group)
    seq = origin in (list, tuple, typing.List, typing.Tuple) or isinstance(d, (list, tuple))
    if seq:
        inner = [a for a in args if a is not Ellipsis]
        if (inner and all(a is int for a in inner)) or (isinstance(d, (list, tuple)) and d and all(isinstance(x, int) for x in d)):
            return Param(name, "int_list", list(d) if d is not None else None, help_, group=group)
        return Param(name, "json", list(d) if d is not None else [], help_, group=group)
    if isinstance(d, dict) or origin is dict:
        return Param(name, "json", d or {}, help_, group=group)
    if ann is str or isinstance(d, str) or d is None:
        return Param(name, "str", d, help_, group=group)
    if callable(d):           # e.g. activation=nn.ReLU -> not editable as text
        return None
    return Param(name, "str", str(d), help_, group=group)


def extract(cls, group: str = "Model", skip: set | None = None) -> list[Param]:
    """Form fields for `cls.__init__` (see module docstring)."""
    fn = cls.__init__ if inspect.isclass(cls) else cls
    skip = SKIP | (skip or set())
    hints, docs, comments = _hints(fn), doc_args(cls, fn) if inspect.isclass(cls) else doc_args(fn), comment_args(fn)
    out = []
    for name, prm in inspect.signature(fn).parameters.items():
        if name in skip or prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            continue
        if prm.default is inspect.Parameter.empty and name in INJECTED:
            continue                                     # filled from the dataset
        ann = hints.get(name, prm.annotation if prm.annotation is not inspect.Parameter.empty else None)
        dc = ann if dataclasses.is_dataclass(ann) else (type(prm.default) if dataclasses.is_dataclass(prm.default) else None)
        if dc is not None:                               # config object -> one field per dataclass field
            base = prm.default if dataclasses.is_dataclass(prm.default) else None
            fh = _hints(dc)
            for f in dataclasses.fields(dc):
                fd = getattr(base, f.name) if base is not None else (
                    f.default if f.default is not dataclasses.MISSING else
                    f.default_factory() if f.default_factory is not dataclasses.MISSING else inspect.Parameter.empty)
                p = _param_for(f"{name}.{f.name}", fh.get(f.name, f.type), fd, comments.get(f.name, ""), group)
                if p:
                    out.append(p)
            continue
        if name in INJECTED and prm.default is not inspect.Parameter.empty:
            # has a default but the dataset determines it: still filled automatically
            continue
        p = _param_for(name, ann, prm.default, docs.get(name) or comments.get(name, ""), group)
        if p:
            out.append(p)
    return out


def assemble(cls, values: dict, injected: dict | None = None, prefix: str = "") -> dict:
    """Constructor kwargs from form values (+ dataset-injected values), rebuilding dataclass config objects."""
    fn = cls.__init__
    sig = inspect.signature(fn).parameters
    hints = _hints(fn)
    kw = {k: v for k, v in (injected or {}).items() if k in sig}
    vals = {k[len(prefix):]: v for k, v in values.items() if k.startswith(prefix)} if prefix else dict(values)
    for name, prm in sig.items():
        if name in SKIP:
            continue
        ann = hints.get(name)
        dc = ann if dataclasses.is_dataclass(ann) else (type(prm.default) if dataclasses.is_dataclass(prm.default) else None)
        if dc is not None:
            fields = {k.split(".", 1)[1]: v for k, v in vals.items() if k.startswith(name + ".")}
            if fields:
                base = prm.default if dataclasses.is_dataclass(prm.default) else None
                kw[name] = dataclasses.replace(base, **fields) if base is not None else dc(**fields)
            continue
        if name in vals and name not in kw and vals[name] is not None:
            v = vals[name]
            if isinstance(prm.default, tuple) and isinstance(v, list):
                v = tuple(v)
            kw[name] = v
    missing = [n for n, prm in sig.items() if n not in SKIP and prm.default is inspect.Parameter.empty and n not in kw
               and prm.kind not in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD)]
    if missing:
        raise ValueError(f"{cls.__name__}: no value for required argument(s) {missing}")
    return kw
