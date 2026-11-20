"""Ordering policies for the kept items of a slot.

Long-context models use information at the start and end of their input more reliably than
information buried in the middle ("lost in the middle", Liu et al., 2023). The ``edges`` policy
therefore places the strongest evidence at both ends of the retrieved block and the weakest in the
middle: ranks 1, 3, 5, … from the front and ranks 2, 4, 6, … from the back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from .policy import Ordering

T = TypeVar("T")


def edges_first(items: Sequence[T], score: Callable[[T], float]) -> list[T]:
    """Best item first, second-best last, and so on inwards (ties keep the input order)."""
    ranked = sorted(range(len(items)), key=lambda i: (-score(items[i]), i))
    front = [items[i] for rank, i in enumerate(ranked) if rank % 2 == 0]
    back = [items[i] for rank, i in enumerate(ranked) if rank % 2 == 1]
    return front + back[::-1]


def order_items(
    items: Sequence[T], mode: Ordering, *, score: Callable[[T], float], position: Callable[[T], int]
) -> list[T]:
    """Order ``items`` by the ``mode`` policy: ``edges``, ``score`` (descending) or ``original``."""
    if mode == "edges":
        return edges_first(items, score)
    if mode == "score":
        return sorted(items, key=lambda item: (-score(item), position(item)))
    return sorted(items, key=position)
