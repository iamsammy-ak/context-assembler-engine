"""Token counting: a cached tokenizer interface, tiktoken encodings, an approximate fallback and
chat-format accounting.

Counting raw text is not enough for a chat model. Every message is wrapped in role markers, the
reply is primed with an assistant header and tool definitions are serialised into the prompt.
:class:`ChatFormat` holds those per-message and per-tool overheads, and :func:`count_prompt` applies
them to a rendered ``messages`` list exactly the way the assembler's estimates do.
"""

from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_ENCODING = "o200k_base"
SUPPORTED_ENCODINGS = ("o200k_base", "cl100k_base")
APPROX = "approx"


@dataclass(frozen=True)
class ChatFormat:
    """Token overhead of the chat format.

    ``per_message``, ``per_name`` and ``reply_priming`` follow the counting recipe OpenAI publishes
    for its chat models (3 tokens per message, 1 per ``name`` field, 3 to prime the reply). Providers
    render tool definitions into an internal format that is not public, so ``tools_overhead`` and
    ``per_tool`` are conservative approximations added on top of the tool JSON. Adjust them, or set a
    ``safety_margin`` on the budget, for providers whose overhead differs.
    """

    per_message: int = 3
    per_name: int = 1
    reply_priming: int = 3
    tools_overhead: int = 12
    per_tool: int = 8


@dataclass(frozen=True)
class CacheInfo:
    """Token-count cache statistics."""

    hits: int
    misses: int
    size: int


def compact_json(value: Any) -> str:
    """The compact JSON form used both to send and to count tool definitions."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class Tokenizer(ABC):
    """Counts tokens with an LRU cache keyed by the exact text.

    The assembler counts the same strings many times (every candidate, every degradation attempt), so
    caching matters. ``count(text, use_cache=False)`` bypasses the cache; the final budget check uses it.
    """

    name: str = "tokenizer"

    def __init__(self, cache_size: int = 65_536) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_size = cache_size
        self._hits = 0
        self._misses = 0

    def count(self, text: str, *, use_cache: bool = True) -> int:
        """Number of tokens in ``text``."""
        if not text:
            return 0
        if not use_cache or self._cache_size == 0:
            return self._count(text)
        cached = self._cache.get(text)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(text)
            return cached
        self._misses += 1
        value = self._count(text)
        self._cache[text] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value

    def clear_cache(self) -> None:
        """Forget cached counts and reset the statistics."""
        self._cache.clear()
        self._hits = self._misses = 0

    @property
    def cache_info(self) -> CacheInfo:
        return CacheInfo(hits=self._hits, misses=self._misses, size=len(self._cache))

    @abstractmethod
    def _count(self, text: str) -> int:
        """Uncached count."""

    @abstractmethod
    def truncate(self, text: str, max_tokens: int) -> str:
        """The longest prefix of ``text`` whose count is at most ``max_tokens``."""


class TiktokenTokenizer(Tokenizer):
    """A tiktoken byte-pair encoding (``o200k_base`` by default, ``cl100k_base`` also supported).

    Special-token strings such as ``<|endoftext|>`` inside user content are counted as plain text.
    tiktoken downloads the encoding file on first use and caches it (``TIKTOKEN_CACHE_DIR``).
    """

    def __init__(self, encoding: str = DEFAULT_ENCODING, cache_size: int = 65_536) -> None:
        import tiktoken

        if encoding not in tiktoken.list_encoding_names():
            raise ValueError(f"unknown tiktoken encoding {encoding!r}; try one of {SUPPORTED_ENCODINGS}")
        super().__init__(cache_size)
        self._encoding = tiktoken.get_encoding(encoding)
        self.name = encoding

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def _count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        for length in range(max_tokens, 0, -1):
            # A cut can split a multi-byte character; drop the replacement char and re-check the count.
            prefix = self._encoding.decode(tokens[:length]).rstrip("�")
            if self.count(prefix) <= max_tokens:
                return prefix
        return ""


# Words, digit runs, whitespace runs, runs of ASCII punctuation, then any other single character.
_APPROX_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|\s+|[!-/:-@\[-`{-~]+|.", re.S)


def _approx_piece_cost(piece: str) -> int:
    first = piece[0]
    if first.isspace():
        # A single space is absorbed by the next word; longer runs and line breaks cost a token.
        return 0 if piece == " " else 1
    if first.isascii() and first.isalpha():
        return 1 + (len(piece) - 1) // 6
    if first.isascii() and (first.isdigit() or not first.isalnum()):
        return (len(piece) + 2) // 3  # digits and punctuation merge in groups of up to three
    return 1


class ApproxTokenizer(Tokenizer):
    """Dependency-free estimate for when tiktoken's encoding files are unavailable.

    Rules: an ASCII word costs one token per started 6 letters, runs of digits or of ASCII punctuation
    cost one token per started 3 characters, every non-ASCII character costs one, and whitespace is
    free except for line breaks and runs of spaces. On English prose and JSON it tends to over-count
    slightly compared with ``o200k_base``, which is the safe direction for a budget. The engine's hard
    guarantee holds with respect to whichever tokenizer it is given, so prefer a real encoding whenever
    one is available.
    """

    name = APPROX

    def _count(self, text: str) -> int:
        return sum(_approx_piece_cost(match.group()) for match in _APPROX_PIECE.finditer(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        used = 0
        for match in _APPROX_PIECE.finditer(text):
            cost = _approx_piece_cost(match.group())
            if used + cost > max_tokens:
                return text[: match.start()]
            used += cost
        return text


def get_tokenizer(encoding: str = DEFAULT_ENCODING, *, fallback: bool = True, cache_size: int = 65_536) -> Tokenizer:
    """A tokenizer for ``encoding`` (``"approx"`` selects the approximate one).

    If the tiktoken encoding cannot be loaded (for example offline without a cached file) and
    ``fallback`` is true, a :class:`RuntimeWarning` is emitted and the approximate tokenizer is returned.
    Unknown encoding names always raise :class:`ValueError`.
    """
    if encoding == APPROX:
        return ApproxTokenizer(cache_size)
    try:
        return TiktokenTokenizer(encoding, cache_size)
    except ValueError:
        raise
    except Exception as exc:  # network or cache failure while fetching the encoding file
        if not fallback:
            raise
        warnings.warn(
            f"tiktoken encoding {encoding!r} is unavailable ({type(exc).__name__}); "
            "falling back to the approximate tokenizer",
            RuntimeWarning,
            stacklevel=2,
        )
        return ApproxTokenizer(cache_size)


def count_prompt(
    tokenizer: Tokenizer,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] = (),
    chat_format: ChatFormat | None = None,
    *,
    use_cache: bool = True,
) -> int:
    """Tokens a rendered prompt occupies, including chat-format and tool-definition overhead."""
    fmt = chat_format or ChatFormat()
    total = fmt.reply_priming
    for message in messages:
        total += fmt.per_message
        for key, value in message.items():
            text = value if isinstance(value, str) else compact_json(value)
            total += tokenizer.count(text, use_cache=use_cache)
            if key == "name":
                total += fmt.per_name
    if tools:
        total += fmt.tools_overhead
        for tool in tools:
            total += fmt.per_tool + tokenizer.count(compact_json(tool), use_cache=use_cache)
    return total
