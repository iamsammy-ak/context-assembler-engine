from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from context_assembler.dedupe import containment, find_near_duplicates, jaccard, shingles
from context_assembler.ordering import edges_first, order_items
from context_assembler.relevance import relevance_scores
from context_assembler.text import longest_fitting_prefix, s_stem, split_identifier, split_sentences, terms

BASE = "Refunds for duplicate charges always go back to the original payment method within ten business days."


def test_s_stemmer_folds_plurals() -> None:
    assert [s_stem(w) for w in ("policies", "invoices", "refunds", "status", "class")] == [
        "policy", "invoice", "refund", "status", "class"
    ]  # fmt: skip
    assert "the" not in terms("The refunds and the invoices")


def test_split_identifier_and_sentences() -> None:
    assert split_identifier("issueRefund_now-v2") == "issue Refund now v2"
    assert split_sentences("First one. Second one!\nThird? ") == ["First one.", "Second one!", "Third?"]


def test_longest_fitting_prefix_adds_suffix_only_when_cut() -> None:
    sentences = ["Aa aa.", "Bb bb.", "Cc cc."]
    assert longest_fitting_prefix(sentences, lambda t: len(t) <= 100) == "Aa aa. Bb bb. Cc cc."
    assert longest_fitting_prefix(sentences, lambda t: len(t) <= 16) == "Aa aa. Bb bb. …"
    assert longest_fitting_prefix(sentences, lambda t: False) is None


def test_jaccard_and_containment() -> None:
    a, b = shingles(BASE), shingles(BASE + " Ask support for a trace.")
    assert jaccard(a, a) == 1.0
    assert 0.5 < jaccard(a, b) < 1.0
    assert containment(a, b) == 1.0
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(a, frozenset()) == 0.0
    assert shingles("two words") == frozenset({("two", "words")})
    assert ("go", "back", "to") in shingles(BASE)


def test_near_duplicates_keep_the_best_scored_copy() -> None:
    texts = [BASE, BASE.replace("ten", "10"), "Single sign-on is available on the Enterprise plan only."]
    duplicates = find_near_duplicates(texts, [0.5, 0.9, 0.8], threshold=0.6)
    assert [(d.dropped, d.kept) for d in duplicates] == [(0, 1)]


def test_truncated_copy_is_caught_by_containment_not_jaccard() -> None:
    long = BASE + " Ask support for a trace if it has not arrived. Quote the refund reference."
    assert jaccard(shingles(long), shingles(BASE)) < 0.8
    assert find_near_duplicates([long, BASE], [0.9, 0.8], containment_threshold=None) == []
    duplicates = find_near_duplicates([long, BASE], [0.9, 0.8])
    assert duplicates[0].measure == "containment" and duplicates[0].dropped == 1


def test_protected_items_are_never_dropped() -> None:
    duplicates = find_near_duplicates([BASE, BASE], [0.9, 0.1], protected=[False, True])
    assert [(d.dropped, d.kept) for d in duplicates] == [(0, 1)]


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), max_size=30))
def test_edges_first_is_a_permutation_with_best_items_at_the_ends(scores: list[float]) -> None:
    items = list(enumerate(scores))
    ordered = edges_first(items, lambda item: item[1])
    assert sorted(ordered) == sorted(items)
    if len(items) >= 2:
        ranked = sorted(items, key=lambda item: (-item[1], item[0]))
        assert ordered[0] == ranked[0] and ordered[-1] == ranked[1]


def test_order_items_modes() -> None:
    items = [("a", 0.1), ("b", 0.9), ("c", 0.5)]
    position = {"a": 0, "b": 1, "c": 2}
    args = {"score": lambda i: i[1], "position": lambda i: position[i[0]]}
    assert [i[0] for i in order_items(items, "score", **args)] == ["b", "c", "a"]
    assert [i[0] for i in order_items(items, "original", **args)] == ["a", "b", "c"]
    assert [i[0] for i in order_items(items, "edges", **args)] == ["b", "a", "c"]


def test_bm25_ranks_the_matching_tool_first() -> None:
    documents = ["issue refund to the original payment method", "get the weather forecast", "send an email"]
    scores = relevance_scores("please refund my duplicate payment", documents)
    assert scores[0] == 1.0 and scores[1] == 0.0
    assert relevance_scores("zzz", documents) == [0.0, 0.0, 0.0]
    assert relevance_scores("anything", []) == []
