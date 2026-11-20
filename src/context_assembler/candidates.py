"""The internal unit of selection: one renderable piece of one slot, with its token cost."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from .slots import SlotKind

CandidateState = Literal["full", "compressed", "trimmed", "summary"]
SUMMARY_ID = "history-summary"


@dataclass(frozen=True)
class Candidate:
    """A piece of content the assembler may keep, degrade or drop.

    ``tokens`` is the estimated cost of rendering this piece, including its line break, bullet or
    citation prefix and (for turns and examples) the per-message chat overhead. ``score`` is the value
    used for selection: relevance, importance, recency or priority depending on the slot.
    """

    slot: SlotKind
    id: str
    text: str
    tokens: int
    score: float
    required: bool
    order: int
    original_tokens: int
    state: CandidateState = "full"
    role: str | None = None
    tool: dict[str, Any] | None = None
    example: tuple[str, str] | None = None
    covers: tuple[str, ...] = ()

    def replaced(self, **changes: Any) -> Candidate:
        """A copy with some fields changed (candidates are immutable)."""
        return dataclasses.replace(self, **changes)
