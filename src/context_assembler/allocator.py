"""Split the allocatable prompt budget across slots: floors, minimums, then priority water-filling.

Water-filling: pour the remaining tokens into a tier's slots in proportion to their weights. A slot
whose level reaches its cap (its demand, or its ``max_share``) stops filling and its unused portion
flows to the others, like water finding its level. Tiers are filled strictly in order.

Phases:

1. **Floors.** Every slot gets the tokens its required items need. If the floors alone exceed the
   budget, allocation is impossible (the assembler then checks whether the required-only prompt fits).
2. **Minimum shares.** Tier by tier, each slot is raised towards ``min(demand, min_share × budget)``.
3. **Priority water-filling** up to ``min(demand, max_share × budget)``, tier by tier.
4. **Spill-over** (unless ``strict_max_share``): if every slot that wants more is capped while tokens
   are still free, the caps are relaxed to the full demand, again tier by tier.

All arithmetic is in whole tokens with a largest-remainder rounding rule and deterministic
tie-breaks, so the same input always gives the same grants.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .slots import SlotKind, slot_index


@dataclass(frozen=True)
class SlotDemand:
    """Allocator input for one slot."""

    slot: SlotKind
    tier: int
    weight: float
    floor: int
    demand: int
    min_share: float = 0.0
    max_share: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.floor <= self.demand:
            raise ValueError(f"{self.slot.value}: need 0 <= floor <= demand, got {self.floor} and {self.demand}")
        if self.weight <= 0:
            raise ValueError(f"{self.slot.value}: weight must be positive")


class InsufficientBudgetError(ValueError):
    """The required floors alone exceed the budget."""


def _water_fill(
    grants: dict[SlotKind, int], slots: Sequence[SlotDemand], caps: dict[SlotKind, int], amount: int
) -> int:
    """Distribute up to ``amount`` tokens over ``slots`` by weight without exceeding ``caps``; return the rest."""
    active = [d for d in slots if grants[d.slot] < caps[d.slot]]
    while amount > 0 and active:
        total_weight = sum(d.weight for d in active)
        shares = {d.slot: amount * d.weight / total_weight for d in active}
        capped = [d for d in active if caps[d.slot] - grants[d.slot] <= shares[d.slot]]
        if capped:
            for d in capped:
                gain = caps[d.slot] - grants[d.slot]
                grants[d.slot] += gain
                amount -= gain
            active = [d for d in active if d not in capped]
            continue
        whole = {slot: math.floor(share) for slot, share in shares.items()}
        for slot, gain in whole.items():
            grants[slot] += gain
        leftover = amount - sum(whole.values())
        by_remainder = sorted(active, key=lambda d: (-(shares[d.slot] - whole[d.slot]), slot_index(d.slot)))
        for d in by_remainder[:leftover]:
            grants[d.slot] += 1  # share < cap - grant, so floor(share) + 1 still respects the cap
        return 0
    return amount


def allocate(demands: Sequence[SlotDemand], budget: int, *, strict_max_share: bool = False) -> dict[SlotKind, int]:
    """Token grant per slot. Guarantees ``floor <= grant <= demand`` and ``sum(grants) <= budget``."""
    if budget < 0:
        raise ValueError("budget must be >= 0")
    grants = {d.slot: d.floor for d in demands}
    remaining = budget - sum(grants.values())
    if remaining < 0:
        raise InsufficientBudgetError(f"required floors need {sum(grants.values())} tokens; budget is {budget}")
    tiers = sorted({d.tier for d in demands})
    by_tier = {tier: sorted((d for d in demands if d.tier == tier), key=lambda d: slot_index(d.slot)) for tier in tiers}

    for tier in tiers:  # 2. minimum shares
        caps = {d.slot: min(d.demand, max(d.floor, math.ceil(d.min_share * budget))) for d in by_tier[tier]}
        remaining = _water_fill(grants, by_tier[tier], caps, remaining)
    for tier in tiers:  # 3. priority water-filling under max_share caps
        caps = {d.slot: max(grants[d.slot], min(d.demand, math.floor(d.max_share * budget))) for d in by_tier[tier]}
        remaining = _water_fill(grants, by_tier[tier], caps, remaining)
    if not strict_max_share:  # 4. spill-over: relax caps rather than leave tokens unused
        for tier in tiers:
            caps = {d.slot: d.demand for d in by_tier[tier]}
            remaining = _water_fill(grants, by_tier[tier], caps, remaining)
    return grants
