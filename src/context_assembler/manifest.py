"""The explain report: what each slot asked for, what it got, and what happened to every item."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .slots import SlotKind

Action = Literal["kept", "compressed", "trimmed", "summarised", "dropped", "deduplicated"]
ACTIONS: tuple[str, ...] = ("kept", "compressed", "trimmed", "summarised", "dropped", "deduplicated")


class ItemDecision(BaseModel):
    """The fate of one input item."""

    model_config = ConfigDict(extra="forbid")

    slot: SlotKind
    item_id: str
    action: Action
    tokens_before: int = Field(ge=0)
    tokens_after: int = Field(ge=0)
    reason: str = ""


class SlotReport(BaseModel):
    """Budget accounting for one slot.

    ``requested_tokens`` is what the slot needs to include everything at full fidelity (after
    near-duplicate removal); ``floor_tokens`` is what its required items need; ``granted_tokens`` is
    the allocator's grant; ``used_tokens`` is what the kept content costs. ``used`` can exceed
    ``granted`` only when spare tokens left by other slots were reclaimed.
    """

    model_config = ConfigDict(extra="forbid")

    slot: SlotKind
    tier: int
    required: bool
    items_in: int
    items_kept: int
    requested_tokens: int
    floor_tokens: int
    granted_tokens: int
    used_tokens: int
    steps: list[str] = Field(default_factory=list)
    actions: dict[str, int] = Field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """True if any item was dropped, compressed, trimmed or summarised."""
        return any(self.actions.get(action, 0) for action in ("compressed", "trimmed", "summarised", "dropped"))


class Manifest(BaseModel):
    """Everything needed to audit one assembly."""

    model_config = ConfigDict(extra="forbid")

    tokenizer: str
    summariser: str
    context_window: int
    output_reserve: int
    safety_margin: int
    prompt_budget: int
    fixed_overhead_tokens: int = Field(description="Reply priming plus the system and final user message wrappers.")
    estimated_tokens: int = Field(description="Fixed overhead plus the per-slot estimates.")
    verified_tokens: int = Field(description="Exact count of the final rendered prompt.")
    headroom: int
    within_budget: bool
    overflow_repairs: int = Field(description="Items removed after verification found the estimate too low.")
    required_only: bool = Field(description="True when only required items fitted.")
    latency_ms: float
    slots: list[SlotReport]
    decisions: list[ItemDecision]

    def slot(self, kind: SlotKind) -> SlotReport:
        """The report for ``kind``."""
        return next(report for report in self.slots if report.slot == kind)

    def decisions_for(self, kind: SlotKind) -> list[ItemDecision]:
        return [decision for decision in self.decisions if decision.slot == kind]
