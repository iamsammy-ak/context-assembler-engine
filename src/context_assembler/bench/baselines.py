"""Baseline assembly strategies the engine is benchmarked against.

Every strategy receives the same request and renders with the same layout
(:func:`~context_assembler.render.render_prompt`), so only selection and token counting differ:

* ``head``: renders everything in prompt order (system message, tools, few-shot, history, final user
  message) and cuts where the budget runs out: the piece that straddles the limit is cut at a token
  boundary and everything after it is dropped. It counts message text and tool JSON with the real
  tokenizer but ignores the per-message chat overhead, a common shortcut.
* ``recent``: the sliding window many chat apps use. System prompt, instructions, memory, tools,
  few-shot examples, retrieved chunks and the query are always sent in full; the remaining budget is
  filled with the newest history turns. Counting is exact, but the fixed parts are never budgeted.
* ``proportional``: a fixed share of the budget per slot, items in their natural order (retrieved by
  score, memory by importance, history newest first) until the share is used, the straddling item cut
  at a token boundary. Unused share is not passed on. Per-item estimates include chat overhead, but
  the rendered result is not re-verified.
* ``engine``: this library with its default policies and the offline extractive summariser.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..assembler import ContextAssembler
from ..candidates import Candidate
from ..policy import default_policies
from ..pools import SlotPool, build_pools
from ..render import CostModel, render_prompt
from ..schema import AssemblyRequest
from ..slots import SlotKind
from ..summarise import ExtractiveSummariser
from ..tokenizer import ChatFormat, Tokenizer, compact_json, count_prompt

MIN_CUT_TOKENS = 16


@dataclass
class StrategyResult:
    """What a strategy produced. ``lossless`` means every source item was sent verbatim."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    lossless: bool
    repairs: int = 0
    estimated_tokens: int | None = None


class Strategy(Protocol):
    name: str

    def assemble(self, request: AssemblyRequest) -> StrategyResult: ...


def full_pools(request: AssemblyRequest, cost: CostModel) -> dict[SlotKind, SlotPool]:
    """Candidate pools at full fidelity with no near-duplicate removal (baselines do not de-duplicate)."""
    policies = {
        slot: policy.model_copy(update={"dedupe_threshold": None, "dedupe_containment": None})
        for slot, policy in default_policies().items()
    }
    return build_pools(request, policies, cost)


def natural_order(pools: dict[SlotKind, SlotPool]) -> dict[SlotKind, list[Candidate]]:
    """How a naive pipeline lists items: retrieved by score, everything else as given."""
    kept = {slot: list(pool.candidates) for slot, pool in pools.items()}
    kept[SlotKind.RETRIEVED].sort(key=lambda c: (-c.score, c.order))
    return kept


class HeadTruncation:
    name = "head"

    def __init__(self, tokenizer: Tokenizer, chat_format: ChatFormat | None = None) -> None:
        self.tokenizer = tokenizer
        self.cost = CostModel(tokenizer, chat_format)

    def assemble(self, request: AssemblyRequest) -> StrategyResult:
        messages, tools = render_prompt(natural_order(full_pools(request, self.cost)))
        has_system = bool(messages) and messages[0]["role"] == "system"
        sequence: list[tuple[str, dict[str, Any]]] = [("message", messages[0])] if has_system else []
        sequence += [("tool", tool) for tool in tools]
        sequence += [("message", message) for message in messages[1 if has_system else 0 :]]
        remaining = request.budget.prompt_budget
        kept_messages: list[dict[str, Any]] = []
        kept_tools: list[dict[str, Any]] = []
        for kind, piece in sequence:
            text = compact_json(piece) if kind == "tool" else piece["content"]
            cost = self.tokenizer.count(text)
            if cost <= remaining:
                (kept_tools if kind == "tool" else kept_messages).append(piece)
                remaining -= cost
                continue
            if kind == "message" and remaining > 0:
                kept_messages.append({**piece, "content": self.tokenizer.truncate(piece["content"], remaining)})
            return StrategyResult(kept_messages, kept_tools, lossless=False)
        return StrategyResult(kept_messages, kept_tools, lossless=True)


class KeepRecent:
    name = "recent"

    def __init__(self, tokenizer: Tokenizer, chat_format: ChatFormat | None = None) -> None:
        self.tokenizer = tokenizer
        self.format = chat_format or ChatFormat()
        self.cost = CostModel(tokenizer, self.format)

    def assemble(self, request: AssemblyRequest) -> StrategyResult:
        kept = natural_order(full_pools(request, self.cost))
        history = kept[SlotKind.HISTORY]
        kept[SlotKind.HISTORY] = []
        base = count_prompt(self.tokenizer, *render_prompt(kept), self.format)
        remaining = request.budget.prompt_budget - base
        window: list[Candidate] = []
        for turn in reversed(history):
            if turn.tokens > remaining:
                break
            window.insert(0, turn)
            remaining -= turn.tokens
        kept[SlotKind.HISTORY] = window
        messages, tools = render_prompt(kept)
        return StrategyResult(messages, tools, lossless=len(window) == len(history))


class FixedProportional:
    name = "proportional"

    SHARES: dict[SlotKind, float] = {
        SlotKind.SYSTEM: 0.10,
        SlotKind.INSTRUCTIONS: 0.05,
        SlotKind.TOOLS: 0.15,
        SlotKind.MEMORY: 0.10,
        SlotKind.RETRIEVED: 0.30,
        SlotKind.HISTORY: 0.25,
        SlotKind.FEW_SHOT: 0.05,
    }

    def __init__(self, tokenizer: Tokenizer, chat_format: ChatFormat | None = None) -> None:
        self.tokenizer = tokenizer
        self.cost = CostModel(tokenizer, chat_format)

    def _cost(self, candidate: Candidate, text: str) -> int:
        if candidate.slot is SlotKind.HISTORY:
            return self.cost.turn(candidate.role or "user", text)
        if candidate.slot is SlotKind.SYSTEM:
            return self.cost.count(text)
        return self.cost.item(candidate.slot, candidate.id, text)

    def _fill(self, pool: SlotPool, items: Sequence[Candidate], cap: int) -> tuple[list[Candidate], bool]:
        used = pool.section_tokens if items else 0
        chosen: list[Candidate] = []
        for candidate in items:
            if used + candidate.tokens <= cap:
                chosen.append(candidate)
                used += candidate.tokens
                continue
            cuttable = candidate.slot not in (SlotKind.TOOLS, SlotKind.FEW_SHOT)
            room = cap - used - self._cost(candidate, "")
            if cuttable and room >= MIN_CUT_TOKENS:
                text = self.tokenizer.truncate(candidate.text, room)
                tokens = self._cost(candidate, text)
                if text and used + tokens <= cap:
                    chosen.append(candidate.replaced(text=text, tokens=tokens, state="trimmed"))
            return chosen, False
        return chosen, True

    def assemble(self, request: AssemblyRequest) -> StrategyResult:
        pools = full_pools(request, self.cost)
        kept = natural_order(pools)
        available = request.budget.prompt_budget - self.cost.fixed() - pools[SlotKind.QUERY].demand
        lossless = True
        for slot, share in self.SHARES.items():
            items = kept[slot]
            if slot is SlotKind.MEMORY:
                items = sorted(items, key=lambda c: (-c.score, c.order))
            if slot is SlotKind.HISTORY:
                items = list(reversed(items))
            chosen, complete = self._fill(pools[slot], items, int(share * available))
            if slot is SlotKind.HISTORY:
                chosen.sort(key=lambda c: c.order)
            kept[slot] = chosen
            lossless = lossless and complete
        messages, tools = render_prompt(kept)
        return StrategyResult(messages, tools, lossless=lossless)


class EngineStrategy:
    name = "engine"

    def __init__(self, tokenizer: Tokenizer, chat_format: ChatFormat | None = None) -> None:
        self.assembler = ContextAssembler(tokenizer, chat_format=chat_format, summariser=ExtractiveSummariser())

    def assemble(self, request: AssemblyRequest) -> StrategyResult:
        result = self.assembler.assemble(request)
        lossless = all(d.action in ("kept", "deduplicated") for d in result.manifest.decisions)
        manifest = result.manifest
        return StrategyResult(
            result.messages, result.tools, lossless, manifest.overflow_repairs, manifest.estimated_tokens
        )


def all_strategies(tokenizer: Tokenizer, chat_format: ChatFormat | None = None) -> list[Strategy]:
    """The three baselines followed by the engine."""
    return [
        HeadTruncation(tokenizer, chat_format),
        KeepRecent(tokenizer, chat_format),
        FixedProportional(tokenizer, chat_format),
        EngineStrategy(tokenizer, chat_format),
    ]
