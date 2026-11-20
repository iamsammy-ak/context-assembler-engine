from __future__ import annotations

import functools
import os
import tempfile
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import settings

from context_assembler import (
    AssemblyRequest,
    Budget,
    ContextItem,
    FewShotExample,
    TiktokenTokenizer,
    Tokenizer,
    ToolSpec,
    Turn,
)
from context_assembler.tokenizer import ApproxTokenizer

# Keep Hypothesis' example database and caches out of the repository.
os.environ.setdefault("HYPOTHESIS_STORAGE_DIRECTORY", os.path.join(tempfile.gettempdir(), "ctxasm-hypothesis"))
# No per-example deadline: shared CI runners and busy laptops make wall-clock limits flaky.
settings.register_profile("repo", deadline=None)
settings.load_profile("repo")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call a model, even if a key is set in the environment or in a .env file."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODELS", raising=False)
    monkeypatch.delenv("CTXASM_MODELS_FILE", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)


@functools.lru_cache(maxsize=1)
def _tiktoken() -> Tokenizer | None:
    try:
        return TiktokenTokenizer("o200k_base")
    except Exception:  # encoding file not cached and no network
        return None


@pytest.fixture
def real_tokenizer() -> Tokenizer:
    """tiktoken o200k_base; skips the test if the encoding file cannot be loaded."""
    tokenizer = _tiktoken()
    if tokenizer is None:
        pytest.skip("tiktoken encoding o200k_base is not available")
    tokenizer.clear_cache()
    return tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """The real tokenizer when available, else the approximate one (the guarantees hold for both)."""
    return _tiktoken() or ApproxTokenizer()


def _tool(name: str, description: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "Invoice number.", "examples": ["INV-1"]},
                "note": {"type": "string", "description": "Free-text note for the billing team."},
            },
            "required": ["invoice_id"],
        },
    )


@pytest.fixture
def make_request() -> Callable[..., AssemblyRequest]:
    """Factory for a request with every slot populated; keyword arguments override fields."""

    def factory(budget: int = 2000, **overrides: Any) -> AssemblyRequest:
        fields: dict[str, Any] = {
            "budget": Budget.prompt_only(budget),
            "system": "You are a careful billing assistant for a software company.",
            "instructions": [
                ContextItem(id="cite", text="Cite sources by id in square brackets.", required=True),
                ContextItem(id="short", text="Keep answers under 120 words.", score=0.5),
            ],
            "tools": [
                _tool("issue_refund", "Refund a paid invoice to the original payment method."),
                _tool("get_weather", "Get the weather forecast for a city."),
                _tool("create_credit_note", "Issue a credit note against an invoice instead of a refund."),
                _tool("send_email", "Send an email to the customer."),
            ],
            "memory": [
                ContextItem(
                    id=f"m{i}", text=f"Customer fact {i}: prefers topic {i} and contact by email.", score=i / 10
                )
                for i in range(8)
            ],
            "retrieved": [
                ContextItem(
                    id=f"doc-{i}",
                    text=f"Article {i}. Refunds for duplicate charges go to the original payment method. "
                    f"Processing takes {i + 3} business days. Contact support with reference R-{100 + i}.",
                    score=round(1 - i / 12, 3),
                )
                for i in range(10)
            ],
            "history": [
                Turn(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Turn {i}: we discussed invoice INV-{2000 + i} and the duplicate charge. It was odd.",
                )
                for i in range(16)
            ],
            "few_shot": [
                FewShotExample(id="ex1", user="Can I get a refund?", assistant="Yes, duplicates are refunded [kb-1].")
            ],
            "query": "Please refund the duplicate charge on my latest invoice.",
        }
        fields.update(overrides)
        return AssemblyRequest(**fields)

    return factory
