"""Rendering kept items into OpenAI-style messages, and the matching per-item cost model.

The prompt layout is fixed so that every piece has a predictable cost:

1. one ``system`` message: the system prompt, then ``## Instructions``, ``## Memory`` and
   ``## Conversation summary`` sections (each only if non-empty);
2. few-shot examples as user/assistant message pairs;
3. the kept conversation turns, oldest first;
4. a final ``user`` message: ``## Retrieved context`` with ``[id] text`` chunks, then
   ``## Question`` and the query (just the query when nothing was retrieved);
5. tool definitions in the separate ``tools`` list.

:class:`CostModel` prices each piece with the same strings the renderer emits. Byte-pair encodings are
not perfectly additive (two strings can merge into fewer, or occasionally more, tokens when joined),
so the assembler always re-counts the final rendered prompt and repairs any overflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .candidates import Candidate
from .slots import SlotKind
from .tokenizer import ChatFormat, Tokenizer, compact_json

INSTRUCTIONS_TITLE = "## Instructions"
MEMORY_TITLE = "## Memory"
SUMMARY_TITLE = "## Conversation summary"
RETRIEVED_TITLE = "## Retrieved context"
QUESTION_TITLE = "## Question"
SECTION_SEPARATOR = "\n\n"
_TITLES = {SlotKind.INSTRUCTIONS: INSTRUCTIONS_TITLE, SlotKind.MEMORY: MEMORY_TITLE}


def bullet_line(text: str) -> str:
    return f"- {text}"


def chunk_line(item_id: str, text: str) -> str:
    return f"[{item_id}] {text}"


class CostModel:
    """Token cost of each renderable piece, including chat-format overhead."""

    def __init__(self, tokenizer: Tokenizer, chat_format: ChatFormat | None = None) -> None:
        self.tokenizer = tokenizer
        self.format = chat_format or ChatFormat()

    def count(self, text: str) -> int:
        return self.tokenizer.count(text)

    def item(self, slot: SlotKind, item_id: str, text: str) -> int:
        """A bullet (instructions, memory) or a cited chunk (retrieved), priced together with the line
        break that follows it, because BPE merges trailing punctuation with it (``".\\n"`` is one token)."""
        if slot is SlotKind.RETRIEVED:
            return self.count(chunk_line(item_id, text) + SECTION_SEPARATOR)
        return self.count(bullet_line(text) + "\n")

    def turn(self, role: str, content: str) -> int:
        return self.format.per_message + self.count(role) + self.count(content)

    def example(self, user: str, assistant: str) -> int:
        return self.turn("user", user) + self.turn("assistant", assistant)

    def tool(self, definition: Mapping[str, Any]) -> int:
        return self.format.per_tool + self.count(compact_json(definition))

    def summary(self, text: str) -> int:
        """A conversation summary, including its section heading."""
        return self.count(f"{SECTION_SEPARATOR}{SUMMARY_TITLE}\n") + self.count(text)

    def section(self, slot: SlotKind) -> int:
        """Overhead a slot adds once when it has at least one kept item."""
        if slot in _TITLES:
            return self.count(f"{SECTION_SEPARATOR}{_TITLES[slot]}\n")
        if slot is SlotKind.RETRIEVED:  # the separator before the question is priced with the last chunk
            return self.count(f"{RETRIEVED_TITLE}\n") + self.count(f"{QUESTION_TITLE}\n")
        if slot is SlotKind.TOOLS:
            return self.format.tools_overhead
        return 0

    def fixed(self) -> int:
        """Reply priming plus the wrappers of the system message and the final user message."""
        fmt = self.format
        return fmt.reply_priming + 2 * fmt.per_message + self.count("system") + self.count("user")


def render_prompt(kept: Mapping[SlotKind, Sequence[Candidate]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render already-selected, already-ordered candidates into ``(messages, tools)``."""

    def items(slot: SlotKind) -> Sequence[Candidate]:
        return kept.get(slot, ())

    sections: list[str] = []
    system_text = "".join(candidate.text for candidate in items(SlotKind.SYSTEM))
    if system_text:
        sections.append(system_text)
    for slot in (SlotKind.INSTRUCTIONS, SlotKind.MEMORY):
        lines = [bullet_line(candidate.text) for candidate in items(slot)]
        if lines:
            sections.append(_TITLES[slot] + "\n" + "\n".join(lines))
    summaries = [candidate.text for candidate in items(SlotKind.HISTORY) if candidate.state == "summary"]
    if summaries:
        sections.append(SUMMARY_TITLE + "\n" + "\n".join(summaries))

    messages: list[dict[str, Any]] = []
    if sections:
        messages.append({"role": "system", "content": SECTION_SEPARATOR.join(sections)})
    for candidate in items(SlotKind.FEW_SHOT):
        if candidate.example is not None:
            user, assistant = candidate.example
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
    for candidate in items(SlotKind.HISTORY):
        if candidate.state != "summary":
            messages.append({"role": candidate.role or "user", "content": candidate.text})

    query = "".join(candidate.text for candidate in items(SlotKind.QUERY))
    chunks = [chunk_line(candidate.id, candidate.text) for candidate in items(SlotKind.RETRIEVED)]
    if chunks:
        content = (
            f"{RETRIEVED_TITLE}\n" + SECTION_SEPARATOR.join(chunks) + f"{SECTION_SEPARATOR}{QUESTION_TITLE}\n{query}"
        )
    else:
        content = query
    messages.append({"role": "user", "content": content})

    tools = [candidate.tool for candidate in items(SlotKind.TOOLS) if candidate.tool is not None]
    return messages, tools
