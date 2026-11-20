"""Data contracts for an assembly request: the budget and the content of every slot."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .policy import PolicyOverride
from .slots import SlotKind


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextItem(_Strict):
    """A text unit for the instructions, memory or retrieved slots.

    ``score`` means relevance for retrieved chunks, importance for memory items and priority for
    instructions. Higher is better; any finite value is accepted.
    """

    id: str = Field(min_length=1, max_length=128)
    text: str
    score: float = Field(default=1.0, allow_inf_nan=False)
    required: bool = False
    source: str | None = None


class Turn(_Strict):
    """One conversation turn. ``pinned`` turns are required and are never summarised or dropped."""

    role: Literal["user", "assistant"]
    content: str
    id: str | None = Field(default=None, min_length=1, max_length=128)
    pinned: bool = False


class ToolSpec(_Strict):
    """A callable tool in OpenAI function format; ``parameters`` is a JSON Schema object."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    required: bool = False


class FewShotExample(_Strict):
    """A worked example rendered as a user/assistant message pair."""

    id: str = Field(min_length=1, max_length=128)
    user: str
    assistant: str
    score: float = Field(default=1.0, allow_inf_nan=False)


class Budget(_Strict):
    """Token budget for one call: ``prompt_budget = context_window - output_reserve - safety_margin``."""

    context_window: int = Field(gt=0)
    output_reserve: int = Field(default=0, ge=0)
    safety_margin: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Budget:
        if self.prompt_budget <= 0:
            raise ValueError(
                f"no room for the prompt: context window {self.context_window} minus output reserve "
                f"{self.output_reserve} and safety margin {self.safety_margin} leaves {self.prompt_budget}"
            )
        return self

    @property
    def prompt_budget(self) -> int:
        """Maximum number of tokens the rendered prompt may use."""
        return self.context_window - self.output_reserve - self.safety_margin

    @classmethod
    def prompt_only(cls, tokens: int) -> Budget:
        """A budget expressed directly as a prompt-token limit (no output reserve)."""
        return cls(context_window=tokens)


class AssemblyRequest(_Strict):
    """Everything the assembler may put in the prompt, plus the budget it must respect."""

    budget: Budget
    system: str = ""
    instructions: list[ContextItem] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    memory: list[ContextItem] = Field(default_factory=list)
    retrieved: list[ContextItem] = Field(default_factory=list)
    history: list[Turn] = Field(default_factory=list)
    few_shot: list[FewShotExample] = Field(default_factory=list)
    query: str = Field(min_length=1)
    policies: dict[SlotKind, PolicyOverride] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the query must not be blank")
        return value

    @model_validator(mode="after")
    def _ids(self) -> AssemblyRequest:
        for index, turn in enumerate(self.history):
            if turn.id is None:
                turn.id = f"turn-{index + 1}"
        groups: dict[str, list[str]] = {
            "instructions": [item.id for item in self.instructions],
            "memory": [item.id for item in self.memory],
            "retrieved": [item.id for item in self.retrieved],
            "history": [turn.id or "" for turn in self.history],
            "few_shot": [example.id for example in self.few_shot],
            "tools": [tool.name for tool in self.tools],
        }
        for slot, ids in groups.items():
            duplicates = sorted(name for name, count in Counter(ids).items() if count > 1)
            if duplicates:
                raise ValueError(f"duplicate ids in {slot}: {duplicates}")
        return self

    def with_budget(self, budget: Budget) -> AssemblyRequest:
        """A copy of this request with a different budget."""
        return self.model_copy(update={"budget": budget})
