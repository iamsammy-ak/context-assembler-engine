"""Near-duplicate detection with word shingles and Jaccard similarity.

Retrieval often returns the same passage twice (a page and its mirror, overlapping chunk windows) and
memories get re-recorded with small wording changes. Every duplicate costs tokens and adds nothing.

A *shingle* is a run of ``size`` consecutive words. Two texts whose shingle sets overlap by at least
``threshold`` (Jaccard similarity = |A ∩ B| / |A ∪ B|) are near-duplicates. Jaccard penalises length
differences, so a truncated copy of a chunk (an overlapping retrieval window, an older shorter version)
can score only ~0.75 against the original. A second test catches those: an item is also redundant when
at least ``containment`` of its own shingles already appear in a kept item (|B ∩ A| / |B|). Items are
visited from the highest score down, so the copy that survives is always the best-scored one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .text import words

Shingle = tuple[str, ...]


def shingles(text: str, size: int = 3) -> frozenset[Shingle]:
    """Word ``size``-grams of ``text`` (as tuples); a text shorter than ``size`` words is one shingle."""
    tokens = words(text)
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset({tuple(tokens)})
    return frozenset(zip(*(tokens[i:] for i in range(size)), strict=False))


def jaccard(a: frozenset[Shingle], b: frozenset[Shingle]) -> float:
    """|A ∩ B| / |A ∪ B|; two empty sets count as identical."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    shared = len(a & b)
    return shared / (len(a) + len(b) - shared)


def containment(part: frozenset[Shingle], whole: frozenset[Shingle]) -> float:
    """|part ∩ whole| / |part|: how much of ``part`` is already covered by ``whole``."""
    if not part:
        return 1.0
    return len(part & whole) / len(part)


@dataclass(frozen=True)
class Duplicate:
    """``dropped`` (an index) is a near-duplicate of the kept item at index ``kept``."""

    dropped: int
    kept: int
    similarity: float
    measure: str = "Jaccard"


def find_near_duplicates(
    texts: Sequence[str],
    scores: Sequence[float],
    *,
    threshold: float = 0.8,
    containment_threshold: float | None = 0.9,
    size: int = 3,
    protected: Sequence[bool] | None = None,
) -> list[Duplicate]:
    """Indices to drop because a better-scored (or protected) near-duplicate is kept.

    An item is a duplicate of a kept item when their Jaccard similarity is at least ``threshold`` or,
    if ``containment_threshold`` is set, when that share of its shingles is covered by the kept item.
    ``protected`` items (for example required ones) are visited first and are never dropped.
    O(n²) set comparisons, which is fine for the tens to low hundreds of items a prompt holds.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if containment_threshold is not None and not 0.0 < containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be in (0, 1]")
    if len(texts) != len(scores):
        raise ValueError("texts and scores must have the same length")
    guard = list(protected) if protected is not None else [False] * len(texts)
    sets = [shingles(text, size) for text in texts]
    order = sorted(range(len(texts)), key=lambda i: (not guard[i], -scores[i], i))
    kept: list[int] = []
    duplicates: list[Duplicate] = []
    for i in order:
        match: Duplicate | None = None
        for j in kept:
            similarity = jaccard(sets[i], sets[j])
            if similarity >= threshold:
                candidate = Duplicate(dropped=i, kept=j, similarity=similarity)
            elif (
                containment_threshold is not None and sets[i] and containment(sets[i], sets[j]) >= containment_threshold
            ):
                candidate = Duplicate(
                    dropped=i, kept=j, similarity=containment(sets[i], sets[j]), measure="containment"
                )
            else:
                continue
            if match is None or candidate.similarity > match.similarity:
                match = candidate
        if match is not None and not guard[i]:
            duplicates.append(match)
        else:
            kept.append(i)
    return duplicates
