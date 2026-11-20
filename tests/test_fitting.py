from __future__ import annotations

from collections.abc import Callable

from context_assembler import AssemblyRequest, ContextItem, ExtractiveSummariser, Tokenizer, Turn
from context_assembler.candidates import SUMMARY_ID
from context_assembler.fitting import FitContext, fit_slot
from context_assembler.policy import PolicyOverride, resolve_policies
from context_assembler.pools import build_pools
from context_assembler.render import CostModel
from context_assembler.slots import SlotKind

Factory = Callable[..., AssemblyRequest]


def _fit(tokenizer: Tokenizer, request: AssemblyRequest, slot: SlotKind, grant: int):  # type: ignore[no-untyped-def]
    cost = CostModel(tokenizer)
    pools = build_pools(request, resolve_policies(request.policies), cost)
    context = FitContext(cost=cost, summariser=ExtractiveSummariser(), query=request.query)
    return pools[slot], fit_slot(pools[slot], grant, context)


def test_everything_is_kept_when_the_grant_covers_demand(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request()
    for slot in SlotKind:
        pool, fit = _fit(tokenizer, request, slot, 10_000)
        assert len(fit.kept) == len(pool.candidates) and not fit.steps


def test_history_keeps_a_recent_window_and_summarises_the_rest(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request()
    pool, fit = _fit(tokenizer, request, SlotKind.HISTORY, 200)
    assert fit.tokens <= 200
    assert fit.kept[0].id == SUMMARY_ID and fit.kept[0].covers
    window = [c.order for c in fit.kept[1:]]
    assert window == list(range(len(pool.candidates) - len(window), len(pool.candidates)))  # newest, contiguous
    assert set(fit.kept[0].covers) == {c.id for c in pool.candidates[: len(pool.candidates) - len(window)]}
    for grant in (10, 40, 90):
        assert _fit(tokenizer, request, SlotKind.HISTORY, grant)[1].tokens <= grant


def test_drop_oldest_keeps_the_newest_contiguous_turns(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(policies={SlotKind.HISTORY: PolicyOverride(chain=("drop_oldest",))})
    pool, fit = _fit(tokenizer, request, SlotKind.HISTORY, 120)
    orders = [c.order for c in fit.kept]
    assert fit.tokens <= 120 and orders
    assert orders == list(range(len(pool.candidates) - len(orders), len(pool.candidates)))
    assert all(c.state == "full" for c in fit.kept)


def test_pinned_turns_survive_the_smallest_grant(tokenizer: Tokenizer, make_request: Factory) -> None:
    history = [Turn(role="user", content=f"Turn {i} with some filler words in it.", pinned=i == 1) for i in range(12)]
    request = make_request(history=history)
    pool, _ = _fit(tokenizer, request, SlotKind.HISTORY, 10_000)
    _, fit = _fit(tokenizer, request, SlotKind.HISTORY, pool.floor)  # the allocator never grants less
    assert [c.id for c in fit.kept] == ["turn-2"]
    assert fit.tokens == pool.floor


def test_retrieved_keeps_the_strongest_chunks_and_orders_them_at_the_edges(
    tokenizer: Tokenizer, make_request: Factory
) -> None:
    request = make_request()
    pool, fit = _fit(tokenizer, request, SlotKind.RETRIEVED, 260)
    assert fit.tokens <= 260
    scores = [c.score for c in fit.kept]
    assert max(scores) == scores[0]  # best first ...
    if len(scores) > 2:
        assert sorted(scores)[-2] == scores[-1]  # ... second best last


def test_trim_to_sentence_uses_spare_grant(tokenizer: Tokenizer, make_request: Factory) -> None:
    long = " ".join(f"Sentence number {i} explains a refund rule in plain words." for i in range(30))
    request = make_request(retrieved=[ContextItem(id="long", text=long, score=0.9)])
    _, fit = _fit(tokenizer, request, SlotKind.RETRIEVED, 120)
    assert len(fit.kept) == 1 and fit.kept[0].state == "trimmed"
    assert fit.kept[0].text.endswith("…") and fit.tokens <= 120


def test_tools_are_compressed_before_relevant_ones_are_dropped(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request()
    pool, full = _fit(tokenizer, request, SlotKind.TOOLS, 10_000)
    _, fit = _fit(tokenizer, request, SlotKind.TOOLS, full.tokens - 10)
    assert fit.tokens <= full.tokens - 10
    assert any(c.state == "compressed" for c in fit.kept)
    _, tight = _fit(tokenizer, request, SlotKind.TOOLS, 80)
    assert [c.id for c in tight.kept] == ["issue_refund"]  # the tool the query needs survives longest


def test_top_k_only_drops_what_is_needed(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(policies={SlotKind.TOOLS: PolicyOverride(top_k=1)})
    pool, full = _fit(tokenizer, request, SlotKind.TOOLS, 10_000)
    smallest = min(c.tokens for c in pool.candidates)
    _, fit = _fit(tokenizer, request, SlotKind.TOOLS, full.tokens - 1)
    assert len(fit.kept) == len(pool.candidates) - 1  # one tool outside the top 1 is enough
    assert fit.tokens >= full.tokens - smallest - 40


def test_memory_keeps_by_importance(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request()
    _, fit = _fit(tokenizer, request, SlotKind.MEMORY, 60)
    kept = sorted(c.score for c in fit.kept)
    dropped_best = max(c.score for c in _fit(tokenizer, request, SlotKind.MEMORY, 10_000)[0].candidates
                       if c.id not in {k.id for k in fit.kept})  # fmt: skip
    assert kept and dropped_best <= kept[0]
    assert fit.kept == sorted(fit.kept, key=lambda c: -c.score)  # rendered most important first
