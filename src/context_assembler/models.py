"""A configurable registry of model profiles: context window, default output reserve and encoding.

The built-in entries are **illustrative profiles, not vendor limits**. Context windows and output
caps change between model versions and providers, so register the models you actually call with the
figures from your provider's documentation, either in code or from a YAML/JSON file::

    models:
      - name: my-model
        context_window: 32768
        output_reserve: 2048
        encoding: o200k_base
        notes: "Figures from the provider's model page, checked 2026-11."
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import Budget
from .tokenizer import DEFAULT_ENCODING


class ModelSpec(BaseModel):
    """One model profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    context_window: int = Field(gt=0)
    output_reserve: int = Field(ge=0)
    encoding: str = DEFAULT_ENCODING
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> ModelSpec:
        if self.output_reserve >= self.context_window:
            raise ValueError(f"{self.name}: output_reserve must be smaller than context_window")
        return self

    def budget(self, *, output_reserve: int | None = None, safety_margin: int = 0) -> Budget:
        """A :class:`Budget` for this model, optionally with a different output reserve."""
        reserve = self.output_reserve if output_reserve is None else output_reserve
        return Budget(context_window=self.context_window, output_reserve=reserve, safety_margin=safety_margin)


BUILTIN_PROFILES: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="generic-4k",
        context_window=4_096,
        output_reserve=512,
        encoding="cl100k_base",
        notes="Illustrative, not a vendor limit.",
    ),
    ModelSpec(
        name="generic-8k",
        context_window=8_192,
        output_reserve=1_024,
        notes="Illustrative, not a vendor limit.",
    ),
    ModelSpec(
        name="generic-32k",
        context_window=32_768,
        output_reserve=2_048,
        notes="Illustrative, not a vendor limit.",
    ),
    ModelSpec(
        name="generic-128k",
        context_window=131_072,
        output_reserve=4_096,
        notes="Illustrative, not a vendor limit.",
    ),
)


class ModelRegistry:
    """Name -> :class:`ModelSpec` lookup that users extend with their own models."""

    def __init__(self, specs: Iterable[ModelSpec] = ()) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        """Add a model. Re-registering a name raises unless ``replace`` is true."""
        if spec.name in self._specs and not replace:
            raise ValueError(f"model {spec.name!r} is already registered; pass replace=True to update it")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown model {name!r}; registered: {', '.join(self.names()) or 'none'}") from None

    def names(self) -> list[str]:
        return list(self._specs)

    def load_file(self, path: str | Path, *, replace: bool = True) -> int:
        """Register every model in a YAML or JSON file; returns how many were loaded."""
        data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        entries = data.get("models", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError(f"{path}: expected a list of models or a mapping with a 'models' list")
        for entry in entries:
            self.register(ModelSpec.model_validate(entry), replace=replace)
        return len(entries)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


def default_registry() -> ModelRegistry:
    """A new registry holding the illustrative built-in profiles."""
    return ModelRegistry(BUILTIN_PROFILES)
