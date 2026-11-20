"""Property-based tests for the hard guarantee: for any input, the verified prompt fits the budget, or
BudgetExceededError is raised and the required items alone really do not fit."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from context_assembler import (
    ApproxTokenizer,
    AssemblyRequest,
    Budget,
    BudgetExceededError,
    ContextAssembler,
    ContextItem,
    FewShotExample,
    Tokenizer,
    ToolSpec,
    Turn,
    count_prompt,
)
from context_assembler.fitting import required_only_fit
from context_assembler.policy import resolve_policies
from context_assembler.pools import build_pools
from context_assembler.render import CostModel, render_prompt

from .conftest import _tiktoken

WORDS = ["refund", "invoice", "charge", "the", "a", "VAT", "INV-2304", "duplicate", "card", "Visa", "4417",
         "policy", "days", "€249", "über", "naïve", "日本", "🎉", "\n", ".", ",", "?", "{", "}",
         "<|endoftext|>"]  # fmt: skip
text = st.lists(st.sampled_from(WORDS), max_size=40).map(" ".join)
short_text = st.lists(st.sampled_from(WORDS), min_size=1, max_size=8).map(" ".join).filter(lambda s: s.strip())
score = st.floats(min_value=-1, max_value=2, allow_nan=False, allow_infinity=False)


@st.composite
def requests(draw: st.DrawFn) -> AssemblyRequest:
    def items(prefix: str, max_size: int, required_rate: float = 0.1) -> list[ContextItem]:
        count = draw(st.integers(0, max_size))
        return [
            ContextItem(
                id=f"{prefix}{i}",
                text=draw(text),
                score=draw(score),
                required=draw(st.floats(0, 1)) < required_rate,
            )
            for i in range(count)
        ]

    tools = [
        ToolSpec(
            name=f"tool_{i}",
            description=draw(text),
            parameters={"type": "object", "properties": {"x": {"type": "string", "description": draw(text)}}},
            required=draw(st.floats(0, 1)) < 0.1,
        )
        for i in range(draw(st.integers(0, 4)))
    ]
    history = [
        Turn(role=draw(st.sampled_from(["user", "assistant"])), content=draw(text), pinned=draw(st.floats(0, 1)) < 0.05)
        for _ in range(draw(st.integers(0, 14)))
    ]
    few_shot = [FewShotExample(id=f"e{i}", user=draw(text), assistant=draw(text), score=draw(score))
                for i in range(draw(st.integers(0, 3)))]  # fmt: skip
    return AssemblyRequest(
        budget=Budget.prompt_only(draw(st.integers(20, 2500))),
        system=draw(text),
        instructions=items("i", 3),
        tools=tools,
        memory=items("m", 6),
        retrieved=items("r", 8),
        history=history,
        few_shot=few_shot,
        query=draw(short_text),
    )


def _check(tokenizer: Tokenizer, request: AssemblyRequest) -> None:
    budget = request.budget.prompt_budget
    try:
        result = ContextAssembler(tokenizer).assemble(request)
    except BudgetExceededError as error:
        cost = CostModel(tokenizer)
        pools = build_pools(request, resolve_policies(), cost)
        messages, tools = render_prompt({slot: required_only_fit(pool).kept for slot, pool in pools.items()})
        assert count_prompt(tokenizer, messages, tools, use_cache=False) > budget
        assert error.required_tokens > budget
        return
    recount = count_prompt(tokenizer, result.messages, result.tools, use_cache=False)
    assert recount == result.manifest.verified_tokens <= budget
    assert result.messages[-1]["content"].endswith(request.query)
    rendered = "\n".join(m["content"] for m in result.messages)
    for item in [*request.instructions, *request.memory, *request.retrieved]:
        if item.required:
            assert item.text in rendered
    for turn in request.history:
        if turn.pinned:
            assert any(m["content"] == turn.content for m in result.messages)


SETTINGS = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(requests())
def test_budget_is_never_exceeded_with_the_real_tokenizer(request: AssemblyRequest) -> None:
    _check(_tiktoken() or ApproxTokenizer(), request)


@SETTINGS
@given(requests())
def test_budget_is_never_exceeded_with_the_approximate_tokenizer(request: AssemblyRequest) -> None:
    _check(ApproxTokenizer(), request)


class NonAdditiveTokenizer(ApproxTokenizer):
    """Adversarial: joined text costs more than its parts, so estimates are systematically low."""

    def _count(self, text: str) -> int:
        return super()._count(text) + 2 * text.count("\n")


@SETTINGS
@given(requests())
def test_budget_holds_even_when_estimates_are_wrong(request: AssemblyRequest) -> None:
    _check(NonAdditiveTokenizer(), request)


@settings(max_examples=40, deadline=None)
@given(requests())
def test_assembly_is_deterministic_for_any_input(request: AssemblyRequest) -> None:
    tokenizer = ApproxTokenizer()
    try:
        first = ContextAssembler(tokenizer).assemble(request)
    except BudgetExceededError:
        return
    second = ContextAssembler(tokenizer).assemble(request)
    assert first.messages == second.messages and first.tools == second.tools
