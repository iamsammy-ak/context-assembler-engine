"""The assembler: allocate, fit, reclaim, render, verify, repair.

The guarantee: the prompt returned by :meth:`ContextAssembler.assemble` never exceeds
``budget.prompt_budget`` tokens, as measured by re-counting the final rendered messages and tool
definitions with the tokenizer (cache bypassed) and the chat-format overheads. If the per-item
estimates were too optimistic, a deterministic repair loop removes the least valuable optional item
of the lowest-priority slot and re-counts until the prompt fits. :class:`BudgetExceededError` is raised
only when the required items alone, rendered and counted, do not fit.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from .allocator import SlotDemand, allocate
from .candidates import Candidate
from .fitting import FitContext, SlotFit, drop_candidate, fit_slot, required_only_fit, victim_key
from .manifest import Action, ItemDecision, Manifest, SlotReport
from .policy import SlotPolicy, resolve_policies
from .pools import SlotPool, build_pools
from .render import CostModel, render_prompt
from .schema import AssemblyRequest
from .slots import SLOT_ORDER, SlotKind, slot_index
from .summarise import ExtractiveSummariser, Summariser
from .tokenizer import ChatFormat, Tokenizer, count_prompt, get_tokenizer

_KEPT_ACTION: dict[str, Action] = {"full": "kept", "compressed": "compressed", "trimmed": "trimmed"}
_KEPT_REASON = {
    "full": "",
    "compressed": "schema descriptions, titles and examples stripped",
    "trimmed": "cut at a sentence boundary to use spare budget",
}


class BudgetExceededError(ValueError):
    """The required items alone do not fit the prompt budget."""

    def __init__(self, required_tokens: int, budget: int, breakdown: Mapping[str, int]) -> None:
        self.required_tokens = required_tokens
        self.budget = budget
        self.breakdown = dict(breakdown)
        parts = ", ".join(f"{name} {tokens}" for name, tokens in self.breakdown.items() if tokens)
        super().__init__(
            f"required content needs {required_tokens} tokens but the prompt budget is {budget} ({parts}). "
            "Raise the budget or output reserve split, shorten the system prompt or query, or unpin items."
        )


class AssembledContext(BaseModel):
    """The assembled prompt (OpenAI-style ``messages`` and ``tools``) and its manifest."""

    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    manifest: Manifest

    def request_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for an OpenAI-compatible ``chat.completions.create`` call."""
        kwargs: dict[str, Any] = {"messages": self.messages}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.manifest.output_reserve:
            kwargs["max_tokens"] = self.manifest.output_reserve
        return kwargs


class ContextAssembler:
    """Assembles prompts under a hard token budget.

    Args:
        tokenizer: token counter; defaults to tiktoken ``o200k_base`` (approximate fallback if unavailable).
        chat_format: per-message and per-tool overheads used for estimates and verification.
        summariser: compresses old history turns; defaults to the offline extractive summariser.
        strict_max_share: never relax ``max_share`` caps, even if tokens would otherwise go unused.
        reclaim: hand tokens a slot could not use (item granularity) to slots that still want more.
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        chat_format: ChatFormat | None = None,
        summariser: Summariser | None = None,
        strict_max_share: bool = False,
        reclaim: bool = True,
    ) -> None:
        self.tokenizer = tokenizer or get_tokenizer()
        self.chat_format = chat_format or ChatFormat()
        self.summariser: Summariser = summariser or ExtractiveSummariser()
        self.strict_max_share = strict_max_share
        self.reclaim = reclaim

    def assemble(self, request: AssemblyRequest) -> AssembledContext:
        """Build the prompt for ``request``; see the module docstring for the guarantee."""
        started = time.perf_counter()
        budget = request.budget.prompt_budget
        policies = resolve_policies(request.policies)
        cost = CostModel(self.tokenizer, self.chat_format)
        pools = build_pools(request, policies, cost)
        fixed = cost.fixed()
        allocatable = budget - fixed
        floors = {slot: pools[slot].floor for slot in SLOT_ORDER}

        required_only = allocatable < sum(floors.values())
        if required_only:
            grants = dict(floors)
            fits = {slot: required_only_fit(pools[slot]) for slot in SLOT_ORDER}
        else:
            demands = [
                SlotDemand(
                    slot=slot,
                    tier=policies[slot].tier,
                    weight=policies[slot].weight,
                    floor=floors[slot],
                    demand=pools[slot].demand,
                    min_share=policies[slot].min_share,
                    max_share=policies[slot].max_share,
                )
                for slot in SLOT_ORDER
            ]
            grants = allocate(demands, allocatable, strict_max_share=self.strict_max_share)
            context = FitContext(cost=cost, summariser=self.summariser, query=request.query)
            fits = {slot: fit_slot(pools[slot], grants[slot], context) for slot in SLOT_ORDER}
            if self.reclaim:
                self._reclaim(pools, policies, fits, allocatable, context)

        messages, tools, verified, repairs = self._render_verified(pools, fits, budget, fixed)
        estimated = fixed + sum(fit.tokens for fit in fits.values())
        decisions = {slot: self._decisions(pools[slot], fits[slot]) for slot in SLOT_ORDER}
        manifest = Manifest(
            tokenizer=self.tokenizer.name,
            summariser=self.summariser.name,
            context_window=request.budget.context_window,
            output_reserve=request.budget.output_reserve,
            safety_margin=request.budget.safety_margin,
            prompt_budget=budget,
            fixed_overhead_tokens=fixed,
            estimated_tokens=estimated,
            verified_tokens=verified,
            headroom=budget - verified,
            within_budget=verified <= budget,
            overflow_repairs=repairs,
            required_only=required_only,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            slots=[self._report(pools[s], policies[s], fits[s], grants[s], decisions[s]) for s in SLOT_ORDER],
            decisions=[d for slot in SLOT_ORDER for d in decisions[slot]],
        )
        return AssembledContext(messages=messages, tools=tools, manifest=manifest)

    # ------------------------------------------------------------------------------------ phases
    def _reclaim(
        self,
        pools: dict[SlotKind, SlotPool],
        policies: dict[SlotKind, SlotPolicy],
        fits: dict[SlotKind, SlotFit],
        allocatable: int,
        context: FitContext,
    ) -> None:
        """Offer unused tokens to slots that still have content left out, highest priority first."""
        order = sorted(SLOT_ORDER, key=lambda s: (policies[s].tier, slot_index(s)))
        for _ in range(2):
            changed = False
            for slot in order:
                slack = allocatable - sum(fit.tokens for fit in fits.values())
                if slack <= 0:
                    return
                pool, current = pools[slot], fits[slot]
                if not pool.candidates or current.tokens >= pool.demand:
                    continue
                grant = current.tokens + slack
                if self.strict_max_share:
                    grant = min(grant, max(current.tokens, int(policies[slot].max_share * allocatable)))
                refit = fit_slot(pool, grant, context)
                if refit.tokens > current.tokens:
                    refit.steps.append(f"reclaim: refitted with {grant} tokens (spare budget from other slots)")
                    fits[slot] = refit
                    changed = True
            if not changed:
                return

    def _render_verified(
        self, pools: dict[SlotKind, SlotPool], fits: dict[SlotKind, SlotFit], budget: int, fixed: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
        """Render, count exactly, and remove items deterministically while the prompt is over budget."""
        repairs = 0
        while True:
            messages, tools = render_prompt({slot: fit.kept for slot, fit in fits.items()})
            verified = count_prompt(self.tokenizer, messages, tools, self.chat_format, use_cache=False)
            if verified <= budget:
                return messages, tools, verified, repairs
            victim = self._overflow_victim(pools, fits)
            if victim is None:
                breakdown = {"format overhead": fixed} | {s.value: pools[s].floor for s in SLOT_ORDER}
                raise BudgetExceededError(verified, budget, breakdown)
            slot, candidate = victim
            drop_candidate(
                fits[slot], candidate, f"overflow repair: the rendered prompt counted {verified - budget} token(s) over"
            )
            fits[slot].steps.append(f"overflow_repair: removed {candidate.id}")
            repairs += 1

    @staticmethod
    def _overflow_victim(
        pools: dict[SlotKind, SlotPool], fits: dict[SlotKind, SlotFit]
    ) -> tuple[SlotKind, Candidate] | None:
        """Least valuable optional item of the lowest-priority slot that still has one."""
        for slot in sorted(SLOT_ORDER, key=lambda s: (-pools[s].policy.tier, -slot_index(s))):
            optional = [c for c in fits[slot].kept if not c.required]
            if optional:
                return slot, min(optional, key=victim_key(slot))
        return None

    # ---------------------------------------------------------------------------------- manifest
    @staticmethod
    def _decisions(pool: SlotPool, fit: SlotFit) -> list[ItemDecision]:
        decisions = list(pool.duplicates)
        kept = {candidate.id: candidate for candidate in fit.kept}
        covered = {cid for candidate in fit.kept for cid in candidate.covers}
        for candidate in pool.candidates:
            final = kept.get(candidate.id)
            if final is not None and final.state in _KEPT_ACTION:
                decisions.append(
                    ItemDecision(
                        slot=pool.slot,
                        item_id=candidate.id,
                        action=_KEPT_ACTION[final.state],
                        tokens_before=candidate.tokens,
                        tokens_after=final.tokens,
                        reason=_KEPT_REASON[final.state],
                    )
                )
            else:
                action: Action = "summarised" if candidate.id in covered else "dropped"
                decisions.append(
                    ItemDecision(
                        slot=pool.slot,
                        item_id=candidate.id,
                        action=action,
                        tokens_before=candidate.tokens,
                        tokens_after=0,
                        reason=fit.reasons.get(candidate.id, "not selected"),
                    )
                )
        return decisions

    @staticmethod
    def _report(
        pool: SlotPool, policy: SlotPolicy, fit: SlotFit, grant: int, decisions: list[ItemDecision]
    ) -> SlotReport:
        actions = Counter(decision.action for decision in decisions)
        return SlotReport(
            slot=pool.slot,
            tier=policy.tier,
            required=policy.required,
            items_in=pool.items_in,
            items_kept=actions["kept"] + actions["compressed"] + actions["trimmed"],
            requested_tokens=pool.demand,
            floor_tokens=pool.floor,
            granted_tokens=grant,
            used_tokens=fit.tokens,
            steps=list(fit.steps),
            actions={action: count for action, count in actions.items() if count},
        )
