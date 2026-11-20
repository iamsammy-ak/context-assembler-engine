from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from context_assembler.allocator import InsufficientBudgetError, SlotDemand, allocate
from context_assembler.schema import ToolSpec
from context_assembler.slots import SLOT_ORDER, SlotKind
from context_assembler.tokenizer import compact_json
from context_assembler.tools import compress_tool, short_description, strip_annotations, to_openai_tool

SCHEMA = {
    "type": "object",
    "title": "Args",
    "properties": {
        "description": {"type": "string", "description": "A property literally named description."},
        "amount": {"type": "number", "minimum": 0, "examples": [12.5], "default": 1},
        "items": {"type": "array", "items": {"type": "string", "description": "An item."}},
        "mode": {"anyOf": [{"type": "string", "enum": ["a", "b"], "title": "Mode"}, {"type": "null"}]},
    },
    "required": ["description", "amount"],
}


def test_strip_annotations_keeps_structure_and_property_names() -> None:
    stripped = strip_annotations(SCHEMA)
    assert "title" not in stripped
    assert set(stripped["properties"]) == {"description", "amount", "items", "mode"}
    assert stripped["properties"]["description"] == {"type": "string"}
    assert stripped["properties"]["amount"] == {"type": "number", "minimum": 0}
    assert stripped["properties"]["items"]["items"] == {"type": "string"}
    assert stripped["properties"]["mode"]["anyOf"][0] == {"type": "string", "enum": ["a", "b"]}
    assert stripped["required"] == ["description", "amount"]


def test_compress_tool_is_smaller_and_keeps_the_name() -> None:
    definition = to_openai_tool(
        ToolSpec(name="issue_refund", description="Refund an invoice. Only after confirmation. " * 3, parameters=SCHEMA)
    )
    compact = compress_tool(definition)
    assert compact["function"]["name"] == "issue_refund"
    assert compact["function"]["description"] == "Refund an invoice."
    assert len(compact_json(compact)) < len(compact_json(definition))


def test_short_description_cuts_at_a_word_boundary() -> None:
    text = "word " * 60
    cut = short_description(text, limit=30)
    assert len(cut) <= 31 and cut.endswith("…")


def _demand(slot: SlotKind, tier: int, floor: int, demand: int, weight: float = 1.0, **shares: float) -> SlotDemand:
    return SlotDemand(slot=slot, tier=tier, weight=weight, floor=floor, demand=demand, **shares)


def test_floors_are_granted_and_insufficient_budget_raises() -> None:
    demands = [_demand(SlotKind.SYSTEM, 0, 50, 50), _demand(SlotKind.RETRIEVED, 2, 0, 500)]
    grants = allocate(demands, 300)
    assert grants[SlotKind.SYSTEM] == 50 and grants[SlotKind.RETRIEVED] == 250
    with pytest.raises(InsufficientBudgetError):
        allocate(demands, 40)


def test_water_filling_splits_by_weight_and_caps_at_demand() -> None:
    demands = [
        _demand(SlotKind.RETRIEVED, 2, 0, 10_000, weight=3),
        _demand(SlotKind.HISTORY, 2, 0, 10_000, weight=1),
        _demand(SlotKind.MEMORY, 2, 0, 50, weight=1),
    ]
    grants = allocate(demands, 1050)
    assert grants[SlotKind.MEMORY] == 50  # capped; its unused share flows to the others
    assert grants[SlotKind.RETRIEVED] == 750 and grants[SlotKind.HISTORY] == 250


def test_higher_tiers_are_filled_first() -> None:
    demands = [_demand(SlotKind.FEW_SHOT, 3, 0, 400), _demand(SlotKind.RETRIEVED, 2, 0, 400)]
    grants = allocate(demands, 500)
    assert grants[SlotKind.RETRIEVED] == 400 and grants[SlotKind.FEW_SHOT] == 100


def test_min_share_reserves_room_for_lower_tiers() -> None:
    demands = [_demand(SlotKind.RETRIEVED, 2, 0, 5000), _demand(SlotKind.FEW_SHOT, 3, 0, 5000, min_share=0.2)]
    assert allocate(demands, 1000)[SlotKind.FEW_SHOT] == 200


def test_max_share_caps_while_others_want_more_and_spills_otherwise() -> None:
    demands = [
        _demand(SlotKind.TOOLS, 2, 0, 5000, max_share=0.25),
        _demand(SlotKind.FEW_SHOT, 3, 0, 5000),
    ]
    assert allocate(demands, 1000)[SlotKind.TOOLS] == 250
    alone = [_demand(SlotKind.TOOLS, 2, 0, 5000, max_share=0.25), _demand(SlotKind.FEW_SHOT, 3, 0, 100)]
    assert allocate(alone, 1000)[SlotKind.TOOLS] == 900
    assert allocate(alone, 1000, strict_max_share=True)[SlotKind.TOOLS] == 250


def test_slot_demand_validates_inputs() -> None:
    with pytest.raises(ValueError):
        _demand(SlotKind.TOOLS, 2, 10, 5)
    with pytest.raises(ValueError):
        _demand(SlotKind.TOOLS, 2, 0, 5, weight=0)


@given(
    st.lists(
        st.tuples(
            st.integers(0, 3),
            st.integers(0, 200),
            st.integers(0, 3000),
            st.floats(0.1, 5),
            st.floats(0, 0.5),
            st.floats(0.5, 1),
        ),
        min_size=len(SLOT_ORDER),
        max_size=len(SLOT_ORDER),
    ),
    st.integers(0, 6000),
    st.booleans(),
)
def test_allocation_invariants(
    rows: list[tuple[int, int, int, float, float, float]], budget: int, strict: bool
) -> None:
    demands = [
        _demand(slot, tier, floor, floor + extra, weight, min_share=lo, max_share=hi)
        for slot, (tier, floor, extra, weight, lo, hi) in zip(SLOT_ORDER, rows, strict=True)
    ]
    if sum(d.floor for d in demands) > budget:
        with pytest.raises(InsufficientBudgetError):
            allocate(demands, budget)
        return
    grants = allocate(demands, budget, strict_max_share=strict)
    assert sum(grants.values()) <= budget
    for d in demands:
        assert d.floor <= grants[d.slot] <= d.demand
    if not strict:  # tokens are only left over when every slot is fully satisfied
        assert sum(grants.values()) == min(budget, sum(d.demand for d in demands))
