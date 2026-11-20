from __future__ import annotations

import pytest

from context_assembler import Budget, Tokenizer, count_prompt
from context_assembler.bench.baselines import (
    EngineStrategy,
    FixedProportional,
    HeadTruncation,
    KeepRecent,
    all_strategies,
)
from context_assembler.bench.metrics import fact_kept, percentile, query_kept, wasted_tokens
from context_assembler.bench.synthetic import make_scenario


def test_scenarios_are_reproducible_and_seed_dependent() -> None:
    first, again, other = make_scenario(3), make_scenario(3), make_scenario(4)
    assert first.request == again.request and first.facts == again.facts
    assert first.request != other.request
    assert {fact.location for fact in first.facts} == {
        "history_early", "history_recent", "memory", "retrieved_top", "retrieved_mid", "tool"
    }  # fmt: skip


def test_every_fact_is_present_at_full_fidelity(tokenizer: Tokenizer) -> None:
    scenario = make_scenario(5)
    result = HeadTruncation(tokenizer).assemble(scenario.request.with_budget(Budget.prompt_only(100_000)))
    assert result.lossless
    assert all(fact_kept(fact, result.messages, result.tools) for fact in scenario.facts)


def test_head_truncation_loses_the_query_and_ignores_chat_overhead(tokenizer: Tokenizer) -> None:
    scenario = make_scenario(6)
    request = scenario.request.with_budget(Budget.prompt_only(1000))
    result = HeadTruncation(tokenizer).assemble(request)
    assert not query_kept(request.query, result.messages)
    assert count_prompt(tokenizer, result.messages, result.tools) > 1000


def test_keep_recent_overflows_when_fixed_parts_are_too_big(tokenizer: Tokenizer) -> None:
    scenario = make_scenario(7)
    request = scenario.request.with_budget(Budget.prompt_only(1500))
    result = KeepRecent(tokenizer).assemble(request)
    assert count_prompt(tokenizer, result.messages, result.tools) > 1500
    assert query_kept(request.query, result.messages)


def test_fixed_split_leaves_budget_unused(tokenizer: Tokenizer) -> None:
    scenario = make_scenario(8)
    request = scenario.request.with_budget(Budget.prompt_only(3000))
    result = FixedProportional(tokenizer).assemble(request)
    verified = count_prompt(tokenizer, result.messages, result.tools)
    assert not result.lossless and wasted_tokens(3000, verified, result.lossless) > 100


@pytest.mark.parametrize("budget", [512, 2048, 6144])
def test_engine_never_violates_and_keeps_the_query(tokenizer: Tokenizer, budget: int) -> None:
    scenario = make_scenario(9)
    request = scenario.request.with_budget(Budget.prompt_only(budget))
    result = EngineStrategy(tokenizer).assemble(request)
    assert count_prompt(tokenizer, result.messages, result.tools, use_cache=False) <= budget
    assert query_kept(request.query, result.messages)


def test_all_strategies_have_distinct_names(tokenizer: Tokenizer) -> None:
    assert [s.name for s in all_strategies(tokenizer)] == ["head", "recent", "proportional", "engine"]


def test_metric_helpers() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    with pytest.raises(ValueError):
        percentile([], 50)
    assert wasted_tokens(100, 80, lossless=True) == 0
    assert wasted_tokens(100, 120, lossless=False) == 0
    assert wasted_tokens(100, 80, lossless=False) == 20
