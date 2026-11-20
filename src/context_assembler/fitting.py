"""Fit one slot's candidates into its token grant by walking the slot's degradation chain.

Every step receives the current state and the grant. *Reducing* steps act only while the slot is over
its grant; ``trim_to_sentence`` is a *filling* step that spends spare grant on a shortened version of
the best item that was dropped. After the chain, a fallback drops the least valuable optional items
until the slot fits, so a fit never exceeds its grant as long as the grant covers the required items
(the allocator guarantees that).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .candidates import SUMMARY_ID, Candidate
from .ordering import order_items
from .pools import SlotPool
from .render import CostModel
from .schema import Turn
from .slots import SlotKind
from .summarise import Summariser
from .text import longest_fitting_prefix, split_sentences
from .tools import compress_tool

MIN_SUMMARY_TOKENS = 24

SCORE_LABEL = {
    SlotKind.INSTRUCTIONS: "priority",
    SlotKind.MEMORY: "importance",
    SlotKind.RETRIEVED: "relevance",
    SlotKind.TOOLS: "relevance",
    SlotKind.FEW_SHOT: "score",
}


@dataclass
class FitContext:
    """What steps need besides the pool: costs, a summariser and the query (for focus).

    Summaries are memoised per (summarised turns, token cap) because the reclaim pass refits slots.
    """

    cost: CostModel
    summariser: Summariser
    query: str
    summaries: dict[tuple[tuple[str, ...], int], str] = field(default_factory=dict)

    def summarise(self, turns: list[Candidate], max_tokens: int) -> str:
        key = (tuple(c.id for c in turns), max_tokens)
        if key not in self.summaries:
            self.summaries[key] = self.summariser.summarise(
                [Turn(role=c.role or "user", content=c.text) for c in turns],  # type: ignore[arg-type]
                max_tokens,
                self.cost.tokenizer,
                focus=self.query,
            )
        return self.summaries[key]


@dataclass
class SlotFit:
    """The kept candidates of one slot plus why the others were not kept."""

    slot: SlotKind
    kept: list[Candidate]
    section_tokens: int
    reasons: dict[str, str] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        if not self.kept:
            return 0
        return self.section_tokens + sum(candidate.tokens for candidate in self.kept)


def drop_candidate(fit: SlotFit, candidate: Candidate, reason: str) -> None:
    """Remove ``candidate`` from the fit and record why (also for the turns a summary covered)."""
    fit.kept = [kept for kept in fit.kept if kept.id != candidate.id]
    fit.reasons[candidate.id] = reason
    for covered in candidate.covers:
        fit.reasons[covered] = f"was in the conversation summary, which was removed: {reason}"


def victim_key(slot: SlotKind) -> Callable[[Candidate], tuple[float, int]]:
    """Sort key putting the first candidate to give up at the front."""
    if slot is SlotKind.HISTORY:
        return lambda c: (0 if c.state == "summary" else 1, c.order)  # summary of the oldest turns, then oldest
    return lambda c: (c.score, -c.order)


def _optional(fit: SlotFit) -> list[Candidate]:
    return [candidate for candidate in fit.kept if not candidate.required]


def _greedy_fill(fit: SlotFit, grant: int, ordered: list[Candidate], why: Callable[[Candidate], str]) -> None:
    """Keep the required items, then add ``ordered`` items while they fit; record a reason for the rest."""
    required = [candidate for candidate in fit.kept if candidate.required]
    capacity = grant - fit.section_tokens - sum(candidate.tokens for candidate in required)
    chosen: list[Candidate] = []
    used = 0
    for candidate in ordered:
        if used + candidate.tokens <= capacity:
            chosen.append(candidate)
            used += candidate.tokens
        else:
            fit.reasons[candidate.id] = why(candidate)
    fit.kept = required + chosen


# --------------------------------------------------------------------------- steps for item slots
def keep_by_score(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Keep the highest-scored items that fit, skipping any that are too large for what is left."""
    if fit.tokens <= grant:
        return
    before = len(fit.kept)
    label = SCORE_LABEL.get(pool.slot, "score")
    ordered = sorted(_optional(fit), key=lambda c: (-c.score, c.order))
    _greedy_fill(fit, grant, ordered, lambda c: f"{label} {c.score:.2f} ranked below the kept items for the "
                 f"{grant}-token grant")  # fmt: skip
    fit.steps.append(f"keep_by_score: kept {len(fit.kept)} of {before} by {label}")


def keep_by_density(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Greedy 0/1 knapsack by value density (value per token), with the best-single-item safeguard.

    ``value = max(score, 0) ** value_exponent``. An exponent above 1 favours one strong chunk over
    several weak ones of the same total length. Comparing the greedy pick with the most valuable
    single item that fits bounds the result at half the optimum (the classic modified greedy).
    """
    if fit.tokens <= grant:
        return
    exponent = pool.policy.value_exponent

    def value(candidate: Candidate) -> float:
        return float(max(candidate.score, 0.0) ** exponent)

    before = len(fit.kept)
    optional = _optional(fit)
    ordered = sorted(optional, key=lambda c: (-value(c) / c.tokens, -c.score, c.order))
    _greedy_fill(fit, grant, ordered, lambda c: f"value density {1000 * value(c) / c.tokens:.2f} per 1k tokens "
                 f"was below the cut for the {grant}-token grant")  # fmt: skip
    chosen = _optional(fit)
    capacity = grant - fit.section_tokens - sum(c.tokens for c in fit.kept if c.required)
    singles = [c for c in optional if c.tokens <= capacity]
    if singles:
        best = max(singles, key=lambda c: (value(c), -c.order))
        if value(best) > sum(value(c) for c in chosen):
            for candidate in chosen:
                fit.reasons[candidate.id] = f"replaced by the single higher-value item {best.id}"
            fit.reasons.pop(best.id, None)
            fit.kept = [c for c in fit.kept if c.required] + [best]
    fit.steps.append(f"keep_by_density: kept {len(fit.kept)} of {before} by {SCORE_LABEL.get(pool.slot)}/token")


def trim_to_sentence(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Spend spare grant on the best dropped items, cut at a sentence boundary."""
    kept_ids = {candidate.id for candidate in fit.kept}
    dropped = sorted(
        (c for c in pool.candidates if c.id not in kept_ids and not c.required), key=lambda c: (-c.score, c.order)
    )
    trimmed = 0
    for candidate in dropped:
        slack = grant - (fit.tokens if fit.kept else fit.section_tokens)
        if slack < pool.policy.min_trim_tokens:
            break

        def fits(text: str, candidate: Candidate = candidate, slack: int = slack) -> bool:
            return ctx.cost.item(pool.slot, candidate.id, text) <= slack

        text = longest_fitting_prefix(split_sentences(candidate.text), fits)
        if text is None:
            continue
        tokens = ctx.cost.item(pool.slot, candidate.id, text)
        if tokens < pool.policy.min_trim_tokens:
            continue
        fit.kept.append(candidate.replaced(text=text, tokens=tokens, state="trimmed"))
        fit.reasons.pop(candidate.id, None)
        trimmed += 1
    if trimmed:
        fit.steps.append(f"trim_to_sentence: added {trimmed} item(s) cut at a sentence boundary")


# ------------------------------------------------------------------------------- steps for tools
def top_k_relevant(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Drop tools ranked below the ``top_k`` most relevant to the query, least relevant first, until the
    slot fits. Tools inside the top k (and required tools) are left for the later steps."""
    if fit.tokens <= grant:
        return
    k = pool.policy.top_k
    ranked = sorted(_optional(fit), key=lambda c: (-c.score, c.order))
    dropped = 0
    for candidate in reversed(ranked[k:]):
        if fit.tokens <= grant:
            break
        drop_candidate(fit, candidate, f"outside the top {k} tools by relevance to the query ({candidate.score:.2f})")
        dropped += 1
    if dropped:
        fit.steps.append(f"top_k_relevant: dropped {dropped} of {len(ranked)} optional tools ranked below the top {k}")


def compress_schemas(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Strip descriptions and examples from tool schemas, least relevant first, until the slot fits."""
    if fit.tokens <= grant:
        return
    compressed = saved = 0
    for candidate in sorted(_optional(fit), key=lambda c: (c.score, -c.order)):
        if fit.tokens <= grant:
            break
        if candidate.state != "full" or candidate.tool is None:
            continue
        compact = compress_tool(candidate.tool)
        tokens = ctx.cost.tool(compact)
        if tokens >= candidate.tokens:
            continue
        index = fit.kept.index(candidate)
        fit.kept[index] = candidate.replaced(tool=compact, tokens=tokens, state="compressed")
        compressed += 1
        saved += candidate.tokens - tokens
    if compressed:
        fit.steps.append(f"compress_schemas: compressed {compressed} tool schema(s), saving {saved} tokens")


# ----------------------------------------------------------------------------- steps for history
def summarise_oldest(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Keep a recent window of turns verbatim and fold the older ones into a capped summary."""
    if fit.tokens <= grant:
        return
    turns = [c for c in fit.kept if c.state != "summary"]
    pinned = [c for c in turns if c.required]
    pinned_tokens = sum(c.tokens for c in pinned)
    heading = ctx.cost.summary("")
    summary_cap = max(MIN_SUMMARY_TOKENS, int(grant * pool.policy.summary_share))
    window_budget = grant - pinned_tokens - heading - summary_cap
    if window_budget < 0:
        return
    window: list[Candidate] = []
    used = 0
    for candidate in reversed(turns):
        if candidate.required:
            continue
        if used + candidate.tokens > window_budget:
            break
        window.append(candidate)
        used += candidate.tokens
    window_ids = {c.id for c in window}
    older = [c for c in turns if not c.required and c.id not in window_ids]
    older_tokens = sum(c.tokens for c in older)
    room = grant - pinned_tokens - used - heading
    max_tokens = min(room, older_tokens - 1)  # a summary must be smaller than what it replaces
    if not older or max_tokens < MIN_SUMMARY_TOKENS:
        return
    text = ctx.summarise(older, max_tokens)
    if not text.strip():
        return
    summary = Candidate(
        slot=SlotKind.HISTORY,
        id=SUMMARY_ID,
        text=text,
        tokens=ctx.cost.summary(text),
        score=0.0,
        required=False,
        order=-1,
        original_tokens=older_tokens,
        state="summary",
        covers=tuple(c.id for c in older),
    )
    if summary.tokens + pinned_tokens + used > grant:
        return
    for candidate in older:
        fit.reasons[candidate.id] = "older than the recent window; folded into the conversation summary"
    fit.kept = [summary] + sorted(pinned + window, key=lambda c: c.order)
    fit.steps.append(
        f"summarise_oldest: {len(older)} older turns ({older_tokens} tokens) -> summary ({summary.tokens} tokens)"
    )


def drop_oldest(fit: SlotFit, pool: SlotPool, grant: int, ctx: FitContext) -> None:
    """Drop the summary, then the oldest unpinned turns, until the slot fits."""
    if fit.tokens <= grant:
        return
    removed = 0
    for candidate in sorted(_optional(fit), key=victim_key(SlotKind.HISTORY)):
        if fit.tokens <= grant:
            break
        drop_candidate(fit, candidate, f"oldest remaining history; dropped to fit the {grant}-token grant")
        removed += 1
    fit.steps.append(f"drop_oldest: dropped {removed} oldest item(s)")


Step = Callable[[SlotFit, SlotPool, int, FitContext], None]

STEPS: dict[str, Step] = {
    "keep_by_score": keep_by_score,
    "keep_by_density": keep_by_density,
    "trim_to_sentence": trim_to_sentence,
    "top_k_relevant": top_k_relevant,
    "compress_schemas": compress_schemas,
    "summarise_oldest": summarise_oldest,
    "drop_oldest": drop_oldest,
}


def _enforce_fit(fit: SlotFit, grant: int) -> None:
    if fit.tokens <= grant:
        return
    removed = 0
    for candidate in sorted(_optional(fit), key=victim_key(fit.slot)):
        if fit.tokens <= grant:
            break
        drop_candidate(fit, candidate, f"dropped to meet the {grant}-token grant (fallback after the chain)")
        removed += 1
    if removed:
        fit.steps.append(f"fallback: dropped {removed} item(s)")


def order_kept(fit: SlotFit, pool: SlotPool) -> None:
    """Put the kept items in render order according to the slot's ordering policy."""
    if fit.slot is SlotKind.HISTORY:
        fit.kept.sort(key=lambda c: c.order)  # the summary (order -1) first, then chronological
        return
    fit.kept = order_items(fit.kept, pool.policy.ordering, score=lambda c: c.score, position=lambda c: c.order)


def fit_slot(pool: SlotPool, grant: int, ctx: FitContext) -> SlotFit:
    """Fit ``pool`` into ``grant`` tokens: run the chain, enforce the grant, order for rendering."""
    fit = SlotFit(slot=pool.slot, kept=list(pool.candidates), section_tokens=pool.section_tokens)
    for name in pool.policy.chain:
        STEPS[name](fit, pool, grant, ctx)
    _enforce_fit(fit, grant)
    order_kept(fit, pool)
    return fit


def required_only_fit(pool: SlotPool) -> SlotFit:
    """A fit holding just the required candidates (used when the budget only covers those)."""
    fit = SlotFit(slot=pool.slot, kept=[c for c in pool.candidates if c.required], section_tokens=pool.section_tokens)
    for candidate in pool.candidates:
        if not candidate.required:
            fit.reasons[candidate.id] = "the budget only covers required items"
    order_kept(fit, pool)
    return fit
