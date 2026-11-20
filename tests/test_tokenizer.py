from __future__ import annotations

import warnings

import pytest

from context_assembler.tokenizer import (
    ApproxTokenizer,
    ChatFormat,
    TiktokenTokenizer,
    Tokenizer,
    compact_json,
    count_prompt,
    get_tokenizer,
)

ENGLISH = (
    "Refunds for duplicate charges go back to the original payment method. Banks usually take five to ten "
    "business days to show them on a statement, and some business cards take longer at month-end."
)


def test_tiktoken_count_matches_encoding(real_tokenizer: TiktokenTokenizer) -> None:
    assert real_tokenizer.count(ENGLISH) == len(real_tokenizer.encode(ENGLISH))
    assert real_tokenizer.count("") == 0


def test_special_token_text_is_counted_as_plain_text(real_tokenizer: TiktokenTokenizer) -> None:
    assert real_tokenizer.count("<|endoftext|>") > 1


def test_count_cache_records_hits_and_misses(real_tokenizer: TiktokenTokenizer) -> None:
    real_tokenizer.count("cache me")
    real_tokenizer.count("cache me")
    info = real_tokenizer.cache_info
    assert (info.hits, info.misses, info.size) == (1, 1, 1)
    real_tokenizer.count("cache me", use_cache=False)
    assert real_tokenizer.cache_info.hits == 1


def test_cache_is_bounded() -> None:
    tokenizer = ApproxTokenizer(cache_size=3)
    for text in ("a", "b", "c", "d"):
        tokenizer.count(text)
    assert tokenizer.cache_info.size == 3


def test_negative_cache_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        ApproxTokenizer(cache_size=-1)


def test_encodings_differ(real_tokenizer: TiktokenTokenizer) -> None:
    try:
        legacy = TiktokenTokenizer("cl100k_base")
    except Exception:
        pytest.skip("cl100k_base is not available")
    text = "Überweisungen, ドキュメント und Rückerstattungen " * 5
    assert legacy.count(text) != real_tokenizer.count(text)
    assert legacy.name == "cl100k_base"


@pytest.mark.parametrize("limit", [0, 1, 5, 17, 40])
def test_truncate_returns_a_prefix_within_the_limit(tokenizer: Tokenizer, limit: int) -> None:
    cut = tokenizer.truncate(ENGLISH, limit)
    assert ENGLISH.startswith(cut)
    assert tokenizer.count(cut) <= limit


def test_truncate_handles_multibyte_characters(real_tokenizer: TiktokenTokenizer) -> None:
    text = "日本語のテキストと絵文字 🎉🎉🎉 が混ざった文章です。" * 3
    for limit in range(1, 12):
        cut = real_tokenizer.truncate(text, limit)
        assert "�" not in cut
        assert real_tokenizer.count(cut) <= limit


def test_approx_tokenizer_is_close_to_and_mostly_above_tiktoken(real_tokenizer: TiktokenTokenizer) -> None:
    approx = ApproxTokenizer()
    samples = [ENGLISH, compact_json({"type": "object", "properties": {"invoice_id": {"type": "string"}}}),
               "Order ORD-48213 shipped on 2026-11-04 to Rotterdam.\n\nTracking 1Z999AA10123456784."]  # fmt: skip
    for text in samples:
        ratio = approx.count(text) / real_tokenizer.count(text)
        assert 0.9 <= ratio <= 1.8, (text, ratio)


def test_get_tokenizer_selects_approx_by_name() -> None:
    assert isinstance(get_tokenizer("approx"), ApproxTokenizer)


def test_unknown_encoding_raises() -> None:
    with pytest.raises(ValueError, match="unknown tiktoken encoding"):
        get_tokenizer("no_such_encoding")


def test_falls_back_to_approx_when_encoding_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    def offline(name: str) -> None:
        raise ConnectionError("no network")

    monkeypatch.setattr(tiktoken, "get_encoding", offline)
    with pytest.warns(RuntimeWarning, match="falling back"):
        tokenizer = get_tokenizer("o200k_base")
    assert isinstance(tokenizer, ApproxTokenizer)
    with warnings.catch_warnings(), pytest.raises(ConnectionError):
        warnings.simplefilter("ignore")
        get_tokenizer("o200k_base", fallback=False)


def test_count_prompt_adds_per_message_and_priming_overhead(tokenizer: Tokenizer) -> None:
    fmt = ChatFormat()
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello there."}]
    raw = sum(tokenizer.count(m["role"]) + tokenizer.count(m["content"]) for m in messages)
    assert count_prompt(tokenizer, messages, chat_format=fmt) == raw + 2 * fmt.per_message + fmt.reply_priming


def test_count_prompt_counts_names_and_tools(tokenizer: Tokenizer) -> None:
    fmt = ChatFormat()
    base = [{"role": "user", "content": "hi"}]
    named = [{"role": "user", "content": "hi", "name": "priya"}]
    assert count_prompt(tokenizer, named) - count_prompt(tokenizer, base) == tokenizer.count("priya") + fmt.per_name
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    with_tool = count_prompt(tokenizer, base, [tool])
    assert with_tool - count_prompt(tokenizer, base) == fmt.tools_overhead + fmt.per_tool + tokenizer.count(
        compact_json(tool)
    )
