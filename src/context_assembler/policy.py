"""Per-slot budgeting policy: priority tier, token shares, weight and the degradation chain.

A policy answers three questions for one slot:

* **How important is it?** ``tier`` (0 is most important) decides who is filled first when tokens are
  scarce; ``weight`` decides how slots in the same tier share what is left.
* **How much may it take?** ``min_share`` is a guaranteed slice of the allocatable budget (if the slot
  has that much content), ``max_share`` caps it while other slots still want tokens.
* **How does it shrink?** ``chain`` is the ordered list of degradation steps applied when the slot's
  content is larger than its grant, for example "summarise the oldest turns, then drop the oldest".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .slots import SlotKind

Ordering = Literal["edges", "score", "original"]

_ITEM_SLOTS = frozenset({SlotKind.INSTRUCTIONS, SlotKind.MEMORY, SlotKind.RETRIEVED, SlotKind.FEW_SHOT, SlotKind.TOOLS})

#: Degradation steps and the slots each one may be used on.
STEP_SLOTS: dict[str, frozenset[SlotKind]] = {
    "keep_by_score": _ITEM_SLOTS,
    "keep_by_density": _ITEM_SLOTS,
    "trim_to_sentence": frozenset({SlotKind.MEMORY, SlotKind.RETRIEVED}),
    "summarise_oldest": frozenset({SlotKind.HISTORY}),
    "drop_oldest": frozenset({SlotKind.HISTORY}),
    "top_k_relevant": frozenset({SlotKind.TOOLS}),
    "compress_schemas": frozenset({SlotKind.TOOLS}),
}

#: Slots whose content is always required in full; their policy cannot be overridden.
FIXED_SLOTS = frozenset({SlotKind.SYSTEM, SlotKind.QUERY})


class SlotPolicy(BaseModel):
    """Budgeting rules for one slot. Shares are fractions of the allocatable prompt budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: int = Field(ge=0, le=9, description="Priority tier; 0 is filled first.")
    required: bool = Field(default=False, description="Every item in the slot is required.")
    min_share: float = Field(default=0.0, ge=0.0, le=1.0)
    max_share: float = Field(default=1.0, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, gt=0.0, description="Share of spare tokens within the tier.")
    chain: tuple[str, ...] = Field(default=(), description="Degradation steps, applied in order.")
    top_k: int = Field(default=8, ge=1, description="tools: how many tools top_k_relevant keeps.")
    summary_share: float = Field(default=0.25, gt=0.0, lt=1.0, description="history: summary size vs grant.")
    dedupe_threshold: float | None = Field(default=None, gt=0.0, le=1.0, description="Jaccard cut-off.")
    dedupe_containment: float | None = Field(default=None, gt=0.0, le=1.0, description="Containment cut-off.")
    value_exponent: float = Field(default=1.0, gt=0.0, description="keep_by_density uses score**exponent.")
    ordering: Ordering = Field(default="original", description="Render order of the kept items.")
    min_trim_tokens: int = Field(default=24, ge=1, description="trim_to_sentence: smallest useful slice.")

    @model_validator(mode="after")
    def _check(self) -> SlotPolicy:
        if self.min_share > self.max_share:
            raise ValueError(f"min_share ({self.min_share}) is larger than max_share ({self.max_share})")
        unknown = [step for step in self.chain if step not in STEP_SLOTS]
        if unknown:
            raise ValueError(f"unknown degradation step(s) {unknown}; choose from {sorted(STEP_SLOTS)}")
        return self


class PolicyOverride(BaseModel):
    """Partial :class:`SlotPolicy`; only the fields that are set replace the defaults."""

    model_config = ConfigDict(extra="forbid")

    tier: int | None = Field(default=None, ge=0, le=9)
    required: bool | None = None
    min_share: float | None = Field(default=None, ge=0.0, le=1.0)
    max_share: float | None = Field(default=None, ge=0.0, le=1.0)
    weight: float | None = Field(default=None, gt=0.0)
    chain: tuple[str, ...] | None = None
    top_k: int | None = Field(default=None, ge=1)
    summary_share: float | None = Field(default=None, gt=0.0, lt=1.0)
    dedupe_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    dedupe_containment: float | None = Field(default=None, gt=0.0, le=1.0)
    value_exponent: float | None = Field(default=None, gt=0.0)
    ordering: Ordering | None = None
    min_trim_tokens: int | None = Field(default=None, ge=1)


def default_policies() -> dict[SlotKind, SlotPolicy]:
    """The engine's default policy for every slot (a fresh copy each call)."""
    return {
        SlotKind.SYSTEM: SlotPolicy(tier=0, required=True),
        SlotKind.QUERY: SlotPolicy(tier=0, required=True),
        SlotKind.INSTRUCTIONS: SlotPolicy(tier=1, max_share=0.15, chain=("keep_by_score",)),
        SlotKind.TOOLS: SlotPolicy(
            tier=2,
            min_share=0.05,
            max_share=0.25,
            weight=1.0,
            chain=("top_k_relevant", "compress_schemas", "keep_by_score"),
            top_k=8,
        ),
        SlotKind.MEMORY: SlotPolicy(
            tier=2,
            min_share=0.05,
            max_share=0.20,
            weight=1.0,
            chain=("keep_by_score",),
            dedupe_threshold=0.8,
            dedupe_containment=0.9,
            ordering="score",
        ),
        SlotKind.RETRIEVED: SlotPolicy(
            tier=2,
            min_share=0.10,
            max_share=0.50,
            weight=3.0,
            chain=("keep_by_density", "trim_to_sentence"),
            dedupe_threshold=0.8,
            dedupe_containment=0.9,
            value_exponent=2.0,
            ordering="edges",
        ),
        SlotKind.HISTORY: SlotPolicy(
            tier=2,
            min_share=0.10,
            max_share=0.40,
            weight=2.0,
            chain=("summarise_oldest", "drop_oldest"),
            summary_share=0.25,
        ),
        SlotKind.FEW_SHOT: SlotPolicy(tier=3, max_share=0.15, chain=("keep_by_score",)),
    }


def resolve_policies(overrides: Mapping[SlotKind, PolicyOverride] | None = None) -> dict[SlotKind, SlotPolicy]:
    """Merge user overrides into the defaults and check every chain step suits its slot."""
    policies = default_policies()
    for slot, override in (overrides or {}).items():
        slot = SlotKind(slot)
        changes = override.model_dump(exclude_none=True)
        if not changes:
            continue
        if slot in FIXED_SLOTS:
            raise ValueError(f"the {slot.value} slot is always required; its policy cannot be overridden")
        policies[slot] = SlotPolicy.model_validate({**policies[slot].model_dump(), **changes})
    for slot, policy in policies.items():
        misplaced = [step for step in policy.chain if slot not in STEP_SLOTS[step]]
        if misplaced:
            raise ValueError(f"step(s) {misplaced} cannot be used on the {slot.value} slot")
    return policies
