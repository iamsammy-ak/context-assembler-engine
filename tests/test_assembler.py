from __future__ import annotations

from collections.abc import Callable

import pytest

from context_assembler import (
    ApproxTokenizer,
    AssemblyRequest,
    BudgetExceededError,
    ContextAssembler,
    ContextItem,
    PolicyOverride,
    SlotKind,
    Tokenizer,
    Turn,
    count_prompt,
)
from context_assembler.render import QUESTION_TITLE, RETRIEVED_TITLE

Factory = Callable[..., AssemblyRequest]


def _verify(tokenizer: Tokenizer, assembler: ContextAssembler, request: AssemblyRequest) -> int:
    result = assembler.assemble(request)
    recount = count_prompt(tokenizer, result.messages, result.tools, use_cache=False)
    assert recount == result.manifest.verified_tokens <= request.budget.prompt_budget
    return recount


def test_generous_budget_keeps_everything(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(budget=20_000)
    result = ContextAssembler(tokenizer).assemble(request)
    actions = {d.action for d in result.manifest.decisions}
    assert actions <= {"kept", "deduplicated"}
    assert len(result.tools) == len(request.tools)
    assert result.messages[0]["role"] == "system" and result.messages[-1]["content"].endswith(request.query)
    assert result.manifest.estimated_tokens >= result.manifest.verified_tokens - 5


@pytest.mark.parametrize("budget", [150, 300, 600, 1200, 2400])
def test_tight_budgets_are_never_exceeded(tokenizer: Tokenizer, make_request: Factory, budget: int) -> None:
    _verify(tokenizer, ContextAssembler(tokenizer), make_request(budget=budget))


def test_required_items_are_always_present(tokenizer: Tokenizer, make_request: Factory) -> None:
    history = [Turn(role="user", content=f"Turn {i}.", pinned=i == 0) for i in range(10)]
    request = make_request(budget=180, history=history)
    result = ContextAssembler(tokenizer).assemble(request)
    system = result.messages[0]["content"]
    assert request.system in system and "Cite sources by id" in system
    assert any(m["content"] == "Turn 0." for m in result.messages)
    assert result.messages[-1]["content"].endswith(request.query)


def test_budget_exceeded_only_when_required_items_cannot_fit(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(budget=60, system="Very long required system prompt. " * 20)
    with pytest.raises(BudgetExceededError) as info:
        ContextAssembler(tokenizer).assemble(request)
    assert info.value.required_tokens > 60 and info.value.budget == 60
    assert "system" in info.value.breakdown


def test_required_only_result_when_floors_exceed_the_estimate(tokenizer: Tokenizer, make_request: Factory) -> None:
    probe = ContextAssembler(tokenizer).assemble(make_request(budget=10_000))
    required = probe.manifest.fixed_overhead_tokens + sum(s.floor_tokens for s in probe.manifest.slots)
    result = ContextAssembler(tokenizer).assemble(make_request(budget=required))
    assert result.manifest.within_budget
    assert result.manifest.verified_tokens <= required


def test_assembly_is_deterministic(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(budget=700)
    first = ContextAssembler(tokenizer).assemble(request)
    second = ContextAssembler(tokenizer).assemble(request)
    assert first.messages == second.messages and first.tools == second.tools
    assert first.manifest.decisions == second.manifest.decisions


def test_manifest_accounting_is_consistent(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(budget=900)
    manifest = ContextAssembler(tokenizer).assemble(request).manifest
    used = sum(s.used_tokens for s in manifest.slots)
    assert manifest.estimated_tokens == manifest.fixed_overhead_tokens + used
    assert manifest.headroom == manifest.prompt_budget - manifest.verified_tokens
    for report in manifest.slots:
        assert report.floor_tokens <= report.granted_tokens <= report.requested_tokens
        assert sum(report.actions.values()) == len(manifest.decisions_for(report.slot))
        assert report.items_in == len(manifest.decisions_for(report.slot))


def test_near_duplicates_are_reported(tokenizer: Tokenizer, make_request: Factory) -> None:
    text = "Refunds for duplicate charges go to the original payment method within ten business days."
    retrieved = [ContextItem(id="a", text=text, score=0.9), ContextItem(id="b", text=text, score=0.5)]
    manifest = ContextAssembler(tokenizer).assemble(make_request(retrieved=retrieved)).manifest
    decision = next(d for d in manifest.decisions if d.item_id == "b")
    assert decision.action == "deduplicated" and "near-duplicate of a" in decision.reason


def test_retrieved_block_puts_the_strongest_chunks_at_the_edges(tokenizer: Tokenizer, make_request: Factory) -> None:
    retrieved = [ContextItem(id=f"r{i}", text=f"Distinct topic number {i} about item {i * 7}.", score=s)
                 for i, s in enumerate([0.2, 0.9, 0.5, 0.8, 0.4])]  # fmt: skip
    result = ContextAssembler(tokenizer).assemble(make_request(budget=20_000, retrieved=retrieved))
    block = result.messages[-1]["content"].split(f"{QUESTION_TITLE}")[0].split(f"{RETRIEVED_TITLE}\n")[1]
    ids = [line.split("]")[0].lstrip("[") for line in block.strip().split("\n\n")]
    assert ids[0] == "r1" and ids[-1] == "r3"


def test_policy_overrides_change_the_outcome(tokenizer: Tokenizer, make_request: Factory) -> None:
    request = make_request(budget=500, policies={SlotKind.FEW_SHOT: PolicyOverride(tier=0, min_share=0.1)})
    manifest = ContextAssembler(tokenizer).assemble(request).manifest
    assert manifest.slot(SlotKind.FEW_SHOT).items_kept == 1
    with pytest.raises(ValueError, match="cannot be overridden"):
        ContextAssembler(tokenizer).assemble(make_request(policies={SlotKind.QUERY: PolicyOverride(tier=3)}))
    with pytest.raises(ValueError, match="cannot be used on the memory slot"):
        ContextAssembler(tokenizer).assemble(
            make_request(policies={SlotKind.MEMORY: PolicyOverride(chain=("drop_oldest",))})
        )


class SeamTokenizer(ApproxTokenizer):
    """Charges extra tokens for the joins that only appear in fully rendered messages, so per-item
    estimates are too low and the verification step has to repair the overflow."""

    name = "seam"

    def _count(self, text: str) -> int:
        return super()._count(text) + 3 * text.count("\n- ") + 5 * text.count("\n\n[")


def test_overflow_repair_restores_the_guarantee(make_request: Factory) -> None:
    tokenizer = SeamTokenizer()
    for budget in (250, 400, 700):
        request = make_request(budget=budget)
        result = ContextAssembler(tokenizer).assemble(request)
        assert result.manifest.verified_tokens <= budget
        assert result.manifest.overflow_repairs > 0
        assert any("overflow repair" in d.reason for d in result.manifest.decisions)


def test_request_kwargs_for_an_openai_client(tokenizer: Tokenizer, make_request: Factory) -> None:
    from context_assembler import Budget

    request = make_request().with_budget(Budget(context_window=3000, output_reserve=500))
    kwargs = ContextAssembler(tokenizer).assemble(request).request_kwargs()
    assert kwargs["max_tokens"] == 500 and kwargs["tools"] and kwargs["messages"]


def test_request_validation(make_request: Factory) -> None:
    with pytest.raises(ValueError, match="duplicate ids"):
        make_request(memory=[ContextItem(id="x", text="a"), ContextItem(id="x", text="b")])
    with pytest.raises(ValueError):
        make_request(query="   ")
    request = make_request(history=[Turn(role="user", content="hi"), Turn(role="assistant", content="hello")])
    assert [t.id for t in request.history] == ["turn-1", "turn-2"]
