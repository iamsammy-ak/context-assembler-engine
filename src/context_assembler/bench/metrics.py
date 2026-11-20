"""Benchmark metrics: fact retention, query retention, wasted tokens and latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .synthetic import Fact


def fact_kept(fact: Fact, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> bool:
    """A tool fact is kept if that tool's schema is sent; any other fact if its marker is in a message."""
    if fact.location == "tool":
        return any(tool["function"]["name"] == fact.marker for tool in tools)
    return any(fact.marker in str(message.get("content", "")) for message in messages)


def query_kept(query: str, messages: Sequence[dict[str, Any]]) -> bool:
    """True if the full query text survives at the end of the final message."""
    return bool(messages) and str(messages[-1].get("content", "")).endswith(query)


def wasted_tokens(budget: int, verified: int, lossless: bool) -> int:
    """Budget left unused while some content was dropped or cut (0 if everything fitted or on a violation)."""
    if lossless or verified > budget:
        return 0
    return budget - verified


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
