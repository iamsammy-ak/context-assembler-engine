"""Turn an :class:`AssemblyRequest` into per-slot candidate pools with token costs.

This is where budget-independent work happens once per assembly: costing every item, removing
near-duplicates, and scoring tools for relevance to the query.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .candidates import Candidate
from .dedupe import find_near_duplicates
from .manifest import ItemDecision
from .policy import SlotPolicy
from .relevance import relevance_scores
from .render import CostModel
from .schema import AssemblyRequest, ContextItem
from .slots import SLOT_ORDER, SlotKind
from .tools import to_openai_tool, tool_document


@dataclass
class SlotPool:
    """All candidates of one slot at full fidelity, after near-duplicate removal."""

    slot: SlotKind
    policy: SlotPolicy
    candidates: list[Candidate]
    section_tokens: int
    items_in: int
    duplicates: list[ItemDecision] = field(default_factory=list)

    @property
    def demand(self) -> int:
        """Tokens needed to include every candidate in full."""
        if not self.candidates:
            return 0
        return self.section_tokens + sum(candidate.tokens for candidate in self.candidates)

    @property
    def floor(self) -> int:
        """Tokens needed for the required candidates alone."""
        required = [candidate for candidate in self.candidates if candidate.required]
        if not required:
            return 0
        return self.section_tokens + sum(candidate.tokens for candidate in required)


def _text_pool(slot: SlotKind, items: Sequence[ContextItem], policy: SlotPolicy, cost: CostModel) -> SlotPool:
    required = [item.required or policy.required for item in items]
    dropped: dict[int, ItemDecision] = {}
    if policy.dedupe_threshold is not None and len(items) > 1:
        duplicates = find_near_duplicates(
            [item.text for item in items],
            [item.score for item in items],
            threshold=policy.dedupe_threshold,
            containment_threshold=policy.dedupe_containment,
            protected=required,
        )
        for duplicate in duplicates:
            item = items[duplicate.dropped]
            dropped[duplicate.dropped] = ItemDecision(
                slot=slot,
                item_id=item.id,
                action="deduplicated",
                tokens_before=cost.item(slot, item.id, item.text),
                tokens_after=0,
                reason=f"near-duplicate of {items[duplicate.kept].id} ({duplicate.measure} {duplicate.similarity:.2f})",
            )
    candidates = []
    for index, item in enumerate(items):
        if index in dropped:
            continue
        tokens = cost.item(slot, item.id, item.text)
        candidates.append(
            Candidate(
                slot=slot,
                id=item.id,
                text=item.text,
                tokens=tokens,
                score=item.score,
                required=required[index],
                order=index,
                original_tokens=tokens,
            )
        )
    return SlotPool(
        slot=slot,
        policy=policy,
        candidates=candidates,
        section_tokens=cost.section(slot),
        items_in=len(items),
        duplicates=[dropped[index] for index in sorted(dropped)],
    )


def build_pools(
    request: AssemblyRequest, policies: Mapping[SlotKind, SlotPolicy], cost: CostModel
) -> dict[SlotKind, SlotPool]:
    """One :class:`SlotPool` per slot, in canonical slot order."""
    pools: dict[SlotKind, SlotPool] = {}

    def single(slot: SlotKind, item_id: str, text: str) -> SlotPool:
        candidates = []
        if text:
            tokens = cost.count(text)
            candidates.append(
                Candidate(slot, item_id, text, tokens, 1.0, required=True, order=0, original_tokens=tokens)
            )
        return SlotPool(slot, policies[slot], candidates, section_tokens=0, items_in=len(candidates))

    pools[SlotKind.SYSTEM] = single(SlotKind.SYSTEM, "system", request.system)
    pools[SlotKind.QUERY] = single(SlotKind.QUERY, "query", request.query)
    for slot, items in (
        (SlotKind.INSTRUCTIONS, request.instructions),
        (SlotKind.MEMORY, request.memory),
        (SlotKind.RETRIEVED, request.retrieved),
    ):
        pools[slot] = _text_pool(slot, items, policies[slot], cost)

    policy = policies[SlotKind.TOOLS]
    definitions = [to_openai_tool(tool) for tool in request.tools]
    relevance = relevance_scores(request.query, [tool_document(definition) for definition in definitions])
    tool_candidates = []
    for index, (tool, definition) in enumerate(zip(request.tools, definitions, strict=True)):
        tokens = cost.tool(definition)
        tool_candidates.append(
            Candidate(
                slot=SlotKind.TOOLS,
                id=tool.name,
                text=tool.description,
                tokens=tokens,
                score=relevance[index],
                required=tool.required or policy.required,
                order=index,
                original_tokens=tokens,
                tool=definition,
            )
        )
    pools[SlotKind.TOOLS] = SlotPool(
        SlotKind.TOOLS, policy, tool_candidates, cost.section(SlotKind.TOOLS), items_in=len(request.tools)
    )

    policy = policies[SlotKind.HISTORY]
    turns = request.history
    history = []
    for index, turn in enumerate(turns):
        tokens = cost.turn(turn.role, turn.content)
        history.append(
            Candidate(
                slot=SlotKind.HISTORY,
                id=turn.id or f"turn-{index + 1}",
                text=turn.content,
                tokens=tokens,
                score=(index + 1) / len(turns),  # recency
                required=turn.pinned or policy.required,
                order=index,
                original_tokens=tokens,
                role=turn.role,
            )
        )
    pools[SlotKind.HISTORY] = SlotPool(SlotKind.HISTORY, policy, history, section_tokens=0, items_in=len(turns))

    policy = policies[SlotKind.FEW_SHOT]
    examples = []
    for index, example in enumerate(request.few_shot):
        tokens = cost.example(example.user, example.assistant)
        examples.append(
            Candidate(
                slot=SlotKind.FEW_SHOT,
                id=example.id,
                text=example.user,
                tokens=tokens,
                score=example.score,
                required=policy.required,
                order=index,
                original_tokens=tokens,
                example=(example.user, example.assistant),
            )
        )
    pools[SlotKind.FEW_SHOT] = SlotPool(
        SlotKind.FEW_SHOT, policy, examples, section_tokens=0, items_in=len(request.few_shot)
    )
    return {slot: pools[slot] for slot in SLOT_ORDER}
