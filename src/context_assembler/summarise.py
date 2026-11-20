"""Summarisers that compress old conversation turns into a token-capped summary.

* :class:`ExtractiveSummariser` is the deterministic **offline stand-in** for a model: it keeps the
  most informative sentences verbatim. It is what the tests, the demo and the benchmark use.
* :class:`LLMSummariser` asks a chat model for an abstractive summary and falls back to the
  extractive one on any failure. Its output is still hard-capped with the tokenizer.

Both honour ``max_tokens`` exactly as measured by the tokenizer they are given.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from .llm import ChatLLM, LLMError, OpenRouterLLM, has_api_key
from .schema import Turn
from .text import longest_fitting_prefix, split_sentences, terms
from .tokenizer import Tokenizer

# Tokens that look like identifiers: codes with digits (ACC-48213, INV-2304, ws_48213) or 3+ digit numbers.
_IDENTIFIER = re.compile(r"\b[A-Za-z]+[-_]?\d[\w-]*|\b\d{3,}\b")


class Summariser(Protocol):
    """Compresses turns into at most ``max_tokens`` tokens (as counted by ``tokenizer``)."""

    name: str

    def summarise(self, turns: Sequence[Turn], max_tokens: int, tokenizer: Tokenizer, focus: str = "") -> str: ...


def cap_text(text: str, max_tokens: int, tokenizer: Tokenizer) -> str:
    """``text`` cut to ``max_tokens``: at a sentence boundary when possible, else mid-sentence."""
    text = text.strip()
    if tokenizer.count(text) <= max_tokens:
        return text
    prefix = longest_fitting_prefix(split_sentences(text), lambda t: tokenizer.count(t) <= max_tokens)
    return prefix if prefix is not None else tokenizer.truncate(text, max_tokens)


class ExtractiveSummariser:
    """Offline stand-in for an LLM summariser: picks informative sentences verbatim (SumBasic-style).

    Selection is greedy. A sentence's salience is the mean probability of its content terms across
    the turns being summarised (frequent topics matter), rescaled so the best sentence scores 1. It
    gets bonuses for overlap with the focus text (normally the current query), for identifiers such
    as invoice numbers and codes, and for user turns (users state facts); questions asked by the
    assistant are penalised (they carry no facts). After each pick the probabilities of the picked
    terms are squared, as in SumBasic, and sentences similar to a picked one (term-set Jaccard) are
    penalised, as in maximal marginal relevance, so later picks favour new information.
    Kept sentences are emitted in their original order as ``role: sentence`` lines.
    """

    name = "extractive (offline)"

    def __init__(
        self,
        *,
        focus_bonus: float = 1.0,
        identifier_bonus: float = 1.0,
        user_bonus: float = 0.25,
        question_penalty: float = 0.75,
        redundancy_penalty: float = 1.5,
    ) -> None:
        self.focus_bonus = focus_bonus
        self.identifier_bonus = identifier_bonus
        self.user_bonus = user_bonus
        self.question_penalty = question_penalty
        self.redundancy_penalty = redundancy_penalty

    def summarise(self, turns: Sequence[Turn], max_tokens: int, tokenizer: Tokenizer, focus: str = "") -> str:
        if max_tokens <= 0 or not turns:
            return ""
        sentences = [(turn.role, sentence) for turn in turns for sentence in split_sentences(turn.content)]
        bags = [set(terms(sentence)) for _, sentence in sentences]
        frequency = Counter(term for bag in bags for term in bag)
        total = sum(frequency.values()) or 1
        probability = {term: count / total for term, count in frequency.items()}
        focus_terms = set(terms(focus))
        bonus = []
        for (role, sentence), bag in zip(sentences, bags, strict=True):
            value = self.focus_bonus * min(1.0, len(bag & focus_terms) / 2)
            value += self.identifier_bonus * bool(_IDENTIFIER.search(sentence))
            value += self.user_bonus * (role == "user")
            value -= self.question_penalty * (role == "assistant" and sentence.endswith("?"))
            bonus.append(value)

        remaining = [i for i, bag in enumerate(bags) if bag]
        lines = {i: f"{sentences[i][0]}: {sentences[i][1]}" for i in remaining}
        costs = {i: tokenizer.count(lines[i]) + 1 for i in remaining}
        overlap = dict.fromkeys(remaining, 0.0)  # highest Jaccard similarity to any chosen sentence
        chosen: list[int] = []
        used = 0
        while True:
            remaining = [i for i in remaining if used + costs[i] <= max_tokens]
            if not remaining:
                break
            salience = {i: sum(probability[t] for t in bags[i]) / len(bags[i]) for i in remaining}
            top = max(salience.values()) or 1.0
            best = max(
                remaining, key=lambda i: (salience[i] / top + bonus[i] - self.redundancy_penalty * overlap[i], -i)
            )
            remaining.remove(best)
            chosen.append(best)
            used += costs[best]
            for term in bags[best]:
                probability[term] **= 2
            for i in remaining:
                shared = len(bags[i] & bags[best])
                overlap[i] = max(overlap[i], shared / (len(bags[i]) + len(bags[best]) - shared))
        text = "\n".join(lines[i] for i in sorted(chosen))
        while chosen and tokenizer.count(text) > max_tokens:  # joins are not perfectly additive
            chosen.pop()  # the last pick is the least valuable
            text = "\n".join(lines[i] for i in sorted(chosen))
        if not chosen and lines:
            first_best = max(lines, key=lambda i: (bonus[i], -i))
            text = tokenizer.truncate(lines[first_best], max_tokens)
        return text


class SummaryReply(BaseModel):
    """Structured reply expected from the model."""

    summary: str


_SUMMARY_PROMPT = (
    "You compress earlier turns of a conversation so an assistant can keep working with less context. "
    "Keep facts, numbers, identifiers, decisions and open questions; drop greetings and filler. "
    "Write plain sentences, at most {words} words."
)


class LLMSummariser:
    """Abstractive summary from a chat model, hard-capped with the tokenizer.

    Any model or format failure falls back to ``fallback`` (the extractive summariser by default), so
    assembly never fails because a free model is busy.
    """

    def __init__(self, llm: ChatLLM, *, fallback: Summariser | None = None) -> None:
        self.llm = llm
        self.fallback: Summariser = fallback or ExtractiveSummariser()
        self.name = f"llm ({llm.name})"
        self.fallbacks_used = 0

    def summarise(self, turns: Sequence[Turn], max_tokens: int, tokenizer: Tokenizer, focus: str = "") -> str:
        if max_tokens <= 0 or not turns:
            return ""
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
        messages = [
            {"role": "system", "content": _SUMMARY_PROMPT.format(words=max(10, int(max_tokens * 0.7)))},
            {"role": "user", "content": f"Current question: {focus}\n\nTurns to summarise:\n{transcript}"},
        ]
        try:
            reply = self.llm.complete_json(messages, SummaryReply, max_tokens=max(256, 2 * max_tokens))
        except (LLMError, ValueError):
            self.fallbacks_used += 1
            return self.fallback.summarise(turns, max_tokens, tokenizer, focus)
        return cap_text(reply.summary, max_tokens, tokenizer)


def select_summariser(*, offline: bool = False) -> Summariser:
    """The LLM summariser when a key is configured and ``offline`` is false, else the extractive one."""
    if not offline and has_api_key():
        return LLMSummariser(OpenRouterLLM())
    return ExtractiveSummariser()
