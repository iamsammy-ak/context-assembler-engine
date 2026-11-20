"""Load an assembly spec (YAML or JSON) into an :class:`AssemblyRequest`.

A spec holds the slot contents plus either ``model`` (looked up in the registry) or an explicit
``budget`` block. ``encoding`` selects the tokenizer; it defaults to the model's encoding::

    model: generic-8k            # or: budget: {context_window: 8192, output_reserve: 1024}
    encoding: o200k_base
    system: You are ...
    instructions: [{id: cite, text: Cite sources by id., required: true}]
    retrieved: [{id: kb-12, text: ..., score: 0.82}]
    history: [{role: user, content: ...}]
    query: ...
    policies: {retrieved: {weight: 4}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import ModelRegistry, default_registry
from .schema import AssemblyRequest, Budget
from .tokenizer import DEFAULT_ENCODING

DEMO_SCENARIO = "support_copilot.yaml"


class SpecError(ValueError):
    """The spec file is missing, malformed or invalid."""


@dataclass(frozen=True)
class LoadedSpec:
    """A validated request plus the tokenizer encoding and model name it asked for."""

    request: AssemblyRequest
    encoding: str
    model: str | None
    title: str


def parse_spec(data: Any, *, registry: ModelRegistry | None = None, budget: Budget | None = None) -> LoadedSpec:
    """Validate spec data. ``budget`` (if given) overrides the spec's model and budget."""
    if not isinstance(data, dict):
        raise SpecError("a spec must be a mapping of slot names to content")
    data = dict(data)
    registry = registry or default_registry()
    model = data.pop("model", None)
    encoding = data.pop("encoding", None)
    title = str(data.pop("title", "") or "")
    raw_budget = data.pop("budget", None)
    spec_model = None
    if model is not None:
        try:
            spec_model = registry.get(str(model))
        except KeyError as exc:
            raise SpecError(str(exc.args[0])) from None
    try:
        if budget is None:
            if raw_budget is not None:
                budget = Budget.model_validate(raw_budget)
            elif spec_model is not None:
                budget = spec_model.budget()
            else:
                raise SpecError("the spec needs a 'model' or a 'budget' (or pass one on the command line)")
        request = AssemblyRequest.model_validate({**data, "budget": budget})
    except ValidationError as exc:
        raise SpecError(f"invalid spec: {exc}") from None
    resolved_encoding = str(encoding or (spec_model.encoding if spec_model else DEFAULT_ENCODING))
    return LoadedSpec(request=request, encoding=resolved_encoding, model=spec_model.name if spec_model else None,
                      title=title)  # fmt: skip


def load_spec(path: str | Path, **kwargs: Any) -> LoadedSpec:
    """Read and validate a ``.yaml``/``.yml`` or ``.json`` spec file."""
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"spec file not found: {path.name}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SpecError(f"could not parse {path.name}: {exc}") from None
    return parse_spec(data, **kwargs)


def demo_spec(**kwargs: Any) -> LoadedSpec:
    """The bundled demo scenario: a billing-support copilot with every slot populated."""
    text = resources.files("context_assembler.data").joinpath(DEMO_SCENARIO).read_text(encoding="utf-8")
    return parse_spec(yaml.safe_load(text), **kwargs)
