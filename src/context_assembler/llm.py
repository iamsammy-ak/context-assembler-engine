"""Optional OpenRouter chat client, used only by :class:`~context_assembler.summarise.LLMSummariser`.

The engine itself never needs a model: budgeting, selection and rendering are deterministic. When
``OPENROUTER_API_KEY`` is set, old conversation turns can be summarised by a model instead of the
offline extractive summariser. Free OpenRouter models are rate-limited, so the client tries each
configured model once (skipping busy ones), caches every successful reply on disk keyed by a hash
of the request, and validates structured replies with one repair round-trip.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "google/gemma-4-31b-it:free",
)
T = TypeVar("T", bound=BaseModel)
Messages = Sequence[dict[str, str]]

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class LLMError(RuntimeError):
    """No configured model produced a usable answer."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    models: tuple[str, ...]


def load_settings(*, use_dotenv: bool = True) -> LLMSettings:
    """Read ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODELS`` from the environment (and ``.env``)."""
    if use_dotenv:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip() or None
    models = tuple(m.strip() for m in os.environ.get("OPENROUTER_MODELS", "").split(",") if m.strip())
    return LLMSettings(api_key=key, models=models or DEFAULT_MODELS)


def has_api_key() -> bool:
    """True when an OpenRouter key is configured."""
    return load_settings().api_key is not None


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in a model reply (tolerates code fences, prose and reasoning tags)."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the model reply") from None
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("the model reply is JSON but not an object")
    return value


class ChatLLM(Protocol):
    """What the summariser needs from a model provider."""

    name: str

    def complete(self, messages: Messages, *, temperature: float = 0.2, max_tokens: int = 1024) -> str: ...

    def complete_json(
        self, messages: Messages, schema: type[T], *, temperature: float = 0.1, max_tokens: int = 1024
    ) -> T: ...


def _default_cache_dir() -> Path:
    return Path(os.environ.get("CTXASM_CACHE_DIR", ".cache")) / "llm"


class OpenRouterLLM:
    """OpenRouter's OpenAI-compatible API with model fallback, an on-disk cache and JSON repair."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        self.settings = settings or load_settings()
        if not self.settings.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set.")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.settings.api_key,
                default_headers={"X-Title": "Context Assembler Engine"},
                timeout=timeout,
                max_retries=0,
            )
        self._client = client
        self.cache_dir = (cache_dir or _default_cache_dir()) if use_cache else None
        self.name = f"openrouter:{self.settings.models[0]}"
        self.last_model: str | None = None
        self.last_errors: list[str] = []

    def _cache_file(self, messages: Messages, json_mode: bool, temperature: float, max_tokens: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = {"models": list(self.settings.models), "messages": list(messages), "json": json_mode,
               "t": temperature, "max": max_tokens}  # fmt: skip
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:40]
        return self.cache_dir / f"{digest}.json"

    def complete(
        self, messages: Messages, *, temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False
    ) -> str:
        """Reply text from the first model that answers; cached on disk by request hash."""
        path = self._cache_file(messages, json_mode, temperature, max_tokens)
        if path is not None and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            self.last_model = cached.get("model")
            return str(cached["content"])
        content, model = self._call_with_fallback(messages, json_mode, temperature, max_tokens)
        self.last_model = model
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "content": content}), encoding="utf-8")
        return content

    def forget(
        self, messages: Messages, *, temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False
    ) -> None:
        """Remove a cached reply (used when a cached reply turns out to be unusable)."""
        path = self._cache_file(messages, json_mode, temperature, max_tokens)
        if path is not None and path.exists():
            path.unlink()

    def _call_with_fallback(
        self, messages: Messages, json_mode: bool, temperature: float, max_tokens: int
    ) -> tuple[str, str]:
        """Try each model once; busy or failing models are skipped, and only busy ones get a second pass."""
        from openai import APIConnectionError, APIStatusError, RateLimitError

        self.last_errors = []
        retryable: list[str] = []
        for round_ in range(2):
            candidates = list(self.settings.models) if round_ == 0 else retryable
            if round_ == 1:
                if not retryable:
                    break
                time.sleep(5)
            retryable = []
            for model in candidates:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": list(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = self._client.chat.completions.create(**kwargs)
                except RateLimitError as exc:
                    if "per-day" in str(exc) or "per day" in str(exc):
                        raise LLMError("OpenRouter's daily free-model limit is used up.") from exc
                    self.last_errors.append(f"{model}: busy (HTTP 429)")
                    retryable.append(model)
                    continue
                except APIConnectionError as exc:  # includes timeouts
                    self.last_errors.append(f"{model}: {type(exc).__name__}")
                    retryable.append(model)
                    continue
                except APIStatusError as exc:
                    self.last_errors.append(f"{model}: HTTP {exc.status_code}")
                    if exc.status_code == 401:
                        raise LLMError("OpenRouter rejected the API key (HTTP 401).") from exc
                    if exc.status_code >= 500:
                        retryable.append(model)
                    continue
                choices = getattr(response, "choices", None) or []
                content = ((choices[0].message.content if choices else None) or "").strip()
                content = _THINK.sub("", content).strip()
                if content:
                    return content, model
                self.last_errors.append(f"{model}: empty reply")
        raise LLMError("No model returned an answer. " + "; ".join(self.last_errors[-6:]))

    def complete_json(
        self, messages: Messages, schema: type[T], *, temperature: float = 0.1, max_tokens: int = 1024
    ) -> T:
        """Ask for JSON matching ``schema``; one repair round-trip if the first reply is invalid."""
        hint = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        request = [dict(m) for m in messages]
        request[0] = {
            "role": request[0]["role"],
            "content": f"{request[0]['content']}\n\nReply with one JSON object only, valid against this "
            f"JSON Schema:\n{hint}",
        }
        raw = self.complete(request, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as first_error:
            self.forget(request, temperature=temperature, max_tokens=max_tokens, json_mode=True)
            repair = request + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That reply could not be used ({str(first_error)[:300]}). "
                 "Return the corrected JSON object only."},
            ]  # fmt: skip
            raw = self.complete(repair, temperature=0.0, max_tokens=max_tokens, json_mode=True)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                self.forget(repair, temperature=0.0, max_tokens=max_tokens, json_mode=True)
                raise LLMError(f"The model's reply did not match the expected format: {exc}") from exc
