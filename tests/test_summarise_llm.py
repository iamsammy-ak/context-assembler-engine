from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from context_assembler import ExtractiveSummariser, LLMSummariser, Tokenizer, Turn, select_summariser
from context_assembler.llm import LLMError, LLMSettings, OpenRouterLLM, extract_json, has_api_key, load_settings
from context_assembler.summarise import SummaryReply, cap_text

TURNS = [
    Turn(role="user", content="Hi there. My account number is ACC-48213 and I was charged twice."),
    Turn(role="assistant", content="Thanks. Which invoice looks wrong?"),
    Turn(role="user", content="Invoice INV-2304 duplicates INV-2291. The weather is nice today."),
    Turn(role="assistant", content="I can see both invoices. A duplicate charge is refunded in full."),
]


def test_extractive_summary_respects_the_cap_and_keeps_key_facts(tokenizer: Tokenizer) -> None:
    summariser = ExtractiveSummariser()
    text = summariser.summarise(TURNS, 45, tokenizer, focus="refund the duplicate on my account")
    assert 0 < tokenizer.count(text) <= 45
    assert "ACC-48213" in text and "INV-2304" in text
    assert "Which invoice looks wrong" not in text  # assistant questions carry no facts
    assert summariser.summarise(TURNS, 30, tokenizer, focus="refund") == summariser.summarise(
        TURNS, 30, tokenizer, focus="refund"
    )


def test_extractive_summary_edge_cases(tokenizer: Tokenizer) -> None:
    summariser = ExtractiveSummariser()
    assert summariser.summarise([], 50, tokenizer) == ""
    assert summariser.summarise(TURNS, 0, tokenizer) == ""
    tiny = summariser.summarise(TURNS, 3, tokenizer)
    assert tokenizer.count(tiny) <= 3


def test_cap_text_prefers_sentence_boundaries(tokenizer: Tokenizer) -> None:
    text = "One short sentence. Another short sentence. A third one here."
    capped = cap_text(text, tokenizer.count("One short sentence. Another short sentence.") + 2, tokenizer)
    assert capped.startswith("One short sentence.") and tokenizer.count(capped) <= tokenizer.count(text)


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, messages: Any, *, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def complete_json(self, messages: Any, schema: Any, *, temperature: float = 0.1, max_tokens: int = 1024) -> Any:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return SummaryReply(summary=self.reply)


def test_llm_summary_is_capped_with_the_tokenizer(tokenizer: Tokenizer) -> None:
    llm = FakeLLM("The customer was double charged. " * 40)
    text = LLMSummariser(llm).summarise(TURNS, 25, tokenizer)
    assert llm.calls == 1 and tokenizer.count(text) <= 25


def test_llm_summariser_falls_back_on_errors(tokenizer: Tokenizer) -> None:
    summariser = LLMSummariser(FakeLLM(LLMError("all models busy")))
    text = summariser.summarise(TURNS, 30, tokenizer, focus="account")
    assert summariser.fallbacks_used == 1 and "ACC-48213" in text


def test_offline_by_default_without_a_key() -> None:
    assert not has_api_key()
    assert isinstance(select_summariser(), ExtractiveSummariser)


def test_settings_read_models_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODELS", "a/one:free, b/two:free")
    settings = load_settings()
    assert settings.api_key == "test-key" and settings.models == ("a/one:free", "b/two:free")
    assert isinstance(select_summariser(offline=True), ExtractiveSummariser)


def test_extract_json_tolerates_fences_and_reasoning() -> None:
    assert extract_json('<think>hmm</think>```json\n{"summary": "x"}\n```') == {"summary": "x"}
    assert extract_json('Sure! {"summary": "y"} Hope that helps.') == {"summary": "y"}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("[1, 2]")


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class FakeClient:
    """Mimics ``client.chat.completions.create``: the first model times out, the second answers."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.models: list[str] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> _Response:
        self.models.append(kwargs["model"])
        if kwargs["model"] == "busy/model":
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return _Response(self.replies.pop(0))


def test_openrouter_client_falls_back_caches_and_repairs(tmp_path: Path) -> None:
    settings = LLMSettings(api_key="test", models=("busy/model", "good/model"))
    client = FakeClient(['{"summary": 5}', '{"summary": "fixed"}'])
    llm = OpenRouterLLM(settings, cache_dir=tmp_path, client=client)
    messages = [{"role": "system", "content": "Summarise."}, {"role": "user", "content": "text"}]
    reply = llm.complete_json(messages, SummaryReply)  # first reply fails validation, repair succeeds
    assert reply.summary == "fixed"
    assert client.models == ["busy/model", "good/model", "busy/model", "good/model"]
    assert len(list(tmp_path.glob("*.json"))) == 1  # the invalid first reply was evicted from the cache


def test_openrouter_client_caches_plain_replies(tmp_path: Path) -> None:
    settings = LLMSettings(api_key="test", models=("good/model",))
    client = FakeClient(["first answer"])
    llm = OpenRouterLLM(settings, cache_dir=tmp_path, client=client)
    messages = [{"role": "user", "content": "hello"}]
    assert llm.complete(messages) == "first answer"
    assert llm.complete(messages) == "first answer"  # served from the disk cache
    assert client.models == ["good/model"]


def test_openrouter_client_requires_a_key() -> None:
    with pytest.raises(LLMError):
        OpenRouterLLM(LLMSettings(api_key=None, models=("m",)))
