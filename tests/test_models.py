from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from context_assembler import Budget, ModelRegistry, ModelSpec, default_registry


def test_builtin_profiles_are_labelled_illustrative() -> None:
    registry = default_registry()
    assert len(registry) >= 3
    assert all("not a vendor limit" in spec.notes for spec in registry)


def test_register_and_get_custom_model() -> None:
    registry = default_registry()
    registry.register(ModelSpec(name="mine", context_window=10_000, output_reserve=1_000))
    assert registry.get("mine").budget().prompt_budget == 9_000
    assert registry.get("mine").budget(output_reserve=0, safety_margin=50).prompt_budget == 9_950


def test_duplicate_registration_needs_replace() -> None:
    registry = ModelRegistry([ModelSpec(name="m", context_window=100, output_reserve=10)])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ModelSpec(name="m", context_window=200, output_reserve=10))
    registry.register(ModelSpec(name="m", context_window=200, output_reserve=10), replace=True)
    assert registry.get("m").context_window == 200


def test_unknown_model_lists_known_names() -> None:
    with pytest.raises(KeyError, match="generic-8k"):
        default_registry().get("nope")


def test_reserve_must_be_smaller_than_window() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(name="bad", context_window=100, output_reserve=100)


def test_load_models_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - {name: a, context_window: 2048, output_reserve: 256, encoding: cl100k_base}\n")
    registry = default_registry()
    assert registry.load_file(path) == 1
    assert registry.get("a").encoding == "cl100k_base"


def test_budget_requires_room_for_the_prompt() -> None:
    with pytest.raises(ValidationError, match="no room"):
        Budget(context_window=1000, output_reserve=900, safety_margin=100)
    assert Budget.prompt_only(512).prompt_budget == 512
