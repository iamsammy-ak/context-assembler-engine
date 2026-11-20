"""The context sources ("slots") a prompt is assembled from."""

from __future__ import annotations

from enum import Enum


class SlotKind(str, Enum):
    """One source of prompt content. The declaration order is the canonical slot order."""

    SYSTEM = "system"
    INSTRUCTIONS = "instructions"
    TOOLS = "tools"
    MEMORY = "memory"
    RETRIEVED = "retrieved"
    HISTORY = "history"
    FEW_SHOT = "few_shot"
    QUERY = "query"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        """Human-readable name, e.g. ``few-shot``."""
        return self.value.replace("_", "-")


SLOT_ORDER: tuple[SlotKind, ...] = tuple(SlotKind)


def slot_index(slot: SlotKind) -> int:
    """Position of ``slot`` in the canonical order (used for deterministic tie-breaks)."""
    return SLOT_ORDER.index(slot)
