"""Plugin interface. Every model, dataset and environment is a subclass of one of these.

A plugin declares its configurable parameters as a list of `Param`; the dashboard renders
a form from them automatically, so new plugins need no frontend code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..training.context import RunContext

# Task / modality vocabulary used to match models to problems.
TASKS = ("classification", "regression", "rl", "seq2seq", "generation")
MODALITIES = ("tabular", "image", "text", "env", "audio", "video")


@dataclass
class Param:
    name: str
    type: str                      # float | int | bool | str | choice | int_list | json
    default: Any = None
    help: str = ""
    min: float | None = None
    max: float | None = None
    choices: list | None = None    # for 'choice'
    suggestions: list | None = None  # free-text 'str' with autocomplete
    log: bool = False              # hint: log-scale value (lr etc.)
    group: str = "Model"           # form section
    labels: dict | None = None     # 'choice': display label per value
    option_groups: dict | None = None  # 'choice': value -> heading it's listed under
    show_if: dict | None = None    # only shown when another field matches: {"param": p, "in": [...]} | {"param": p, "not_in": [...]}

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def coerce_params(spec: list[Param], values: dict | None) -> dict:
    """Fill defaults and coerce user values to declared types. Unknown keys are kept."""
    values = dict(values or {})
    out: dict[str, Any] = {}
    for p in spec:
        v = values.pop(p.name, p.default)
        if v is None or v == "":
            out[p.name] = p.default if v == "" else v
            continue
        try:
            if p.type == "float":
                v = float(v)
            elif p.type == "int":
                v = int(float(v))
            elif p.type == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif p.type == "int_list":
                if isinstance(v, str):
                    v = [int(x) for x in v.replace(" ", "").split(",") if x]
                else:
                    v = [int(x) for x in v]
            elif p.type == "json" and isinstance(v, str):
                import json
                v = json.loads(v) if v.strip() else {}
            elif p.type == "choice" and p.choices and v not in p.choices:
                raise ValueError(f"must be one of {p.choices}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Parameter '{p.name}': invalid value {v!r} ({e})") from None
        if p.type in ("float", "int"):
            if p.min is not None and v < p.min:
                raise ValueError(f"Parameter '{p.name}' must be >= {p.min}")
            if p.max is not None and v > p.max:
                raise ValueError(f"Parameter '{p.name}' must be <= {p.max}")
        out[p.name] = v
    out.update(values)
    return out


class Plugin:
    kind: ClassVar[str] = ""
    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    source: str = "builtin"        # set by the registry: 'builtin' or the plugin file path

    def params(self) -> list[Param]:
        return []

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "id": self.id,
            "name": self.name or self.id,
            "description": self.description,
            "source": self.source,
            "params": [p.to_dict() for p in self.params()],
        }


@dataclass
class DataBundle:
    """What a dataset hands to a model.

    tabular: train/val/test are (X, y) numpy tuples
    image:   train/val/test are torch Datasets yielding (tensor, label)
    text:    train/val/test are lists of dicts: {"text", "label"} or {"source", "target"}
    `info` carries shape metadata: n_features, n_classes, class_names, input_shape, ...
    """
    task: str
    modality: str
    train: Any
    val: Any
    test: Any
    info: dict = field(default_factory=dict)
    raw: dict | None = None   # tabular: {"train"|"val"|"test": (raw DataFrame, y)} before preprocessing
    weights: dict | None = None   # optional per-example training weights {"train": np.ndarray} (e.g. recency weighting)


class DatasetPlugin(Plugin):
    kind = "dataset"
    task: ClassVar[str] = "classification"
    modality: ClassVar[str] = "tabular"

    def load(self, params: dict, seed: int) -> DataBundle:
        raise NotImplementedError

    def meta(self) -> dict:
        return {**super().meta(), "task": self.task, "modality": self.modality}


class EnvPlugin(Plugin):
    kind = "env"
    task: ClassVar[str] = "rl"
    modality: ClassVar[str] = "env"

    def make(self, params: dict, seed: int, render_mode: str | None = None):
        """Return a gymnasium.Env instance. render_mode is None, 'rgb_array' (videos/live view) or 'human' (GUI)."""
        raise NotImplementedError

    def create(self, params: dict, seed: int, render_mode: str | None = None):
        """Call make(), tolerating older plugins whose make() has no render_mode argument."""
        import inspect
        if "render_mode" in inspect.signature(self.make).parameters:
            return self.make(params, seed, render_mode=render_mode)
        return self.make(params, seed)

    def meta(self) -> dict:
        return {**super().meta(), "task": self.task, "modality": self.modality}


class ModelPlugin(Plugin):
    kind = "model"
    framework: ClassVar[str] = "custom"      # sklearn | torch | sb3 | hf | custom
    tasks: ClassVar[tuple[str, ...]] = ()
    modalities: ClassVar[tuple[str, ...]] = ()
    uses_gpu: ClassVar[bool] = False
    supports_init: ClassVar[bool] = False      # can start from a saved model (fine-tune / continue training)
    supports_eval: ClassVar[bool] = False      # can evaluate a saved model without training
    supports_resume: ClassVar[bool] = False    # can continue a stopped/failed run from its checkpoint
    eval_without_init: ClassVar[bool] = False  # can evaluate pretrained weights named in its params (e.g. HF hub)

    def evaluate(self, problem: Any, ctx: "RunContext") -> dict:
        """Evaluate the saved model at ctx.init_path (or pretrained weights) on `problem`; return final metrics."""
        raise NotImplementedError

    def train(self, problem: Any, ctx: "RunContext") -> dict:
        """Train on `problem` (a DataBundle, or an EnvPlugin for RL).

        Log curves with ctx.log(step, {...}); return a dict of final metrics.
        """
        raise NotImplementedError

    def compatible(self, task: str, modality: str) -> bool:
        return task in self.tasks and modality in self.modalities

    def meta(self) -> dict:
        return {**super().meta(), "framework": self.framework, "tasks": list(self.tasks),
                "modalities": list(self.modalities), "uses_gpu": self.uses_gpu, "supports_init": self.supports_init,
                "supports_eval": self.supports_eval, "supports_resume": self.supports_resume,
                "eval_without_init": self.eval_without_init}
