"""Rich views of assemblies: the multi-budget demo, stacked composition bars and the explain report."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from .assembler import AssembledContext
from .manifest import Manifest
from .slots import SlotKind

SLOT_COLOURS: dict[SlotKind, str] = {
    SlotKind.SYSTEM: "#94A3B8",
    SlotKind.INSTRUCTIONS: "#A78BFA",
    SlotKind.TOOLS: "#F59E0B",
    SlotKind.MEMORY: "#10B981",
    SlotKind.RETRIEVED: "#3B82F6",
    SlotKind.HISTORY: "#EC4899",
    SlotKind.FEW_SHOT: "#14B8A6",
    SlotKind.QUERY: "#EF4444",
}
ACTION_STYLES = {
    "kept": "green",
    "compressed": "cyan",
    "trimmed": "yellow",
    "summarised": "magenta",
    "dropped": "red",
    "deduplicated": "bright_black",
}
DEGRADABLE = (SlotKind.TOOLS, SlotKind.MEMORY, SlotKind.RETRIEVED, SlotKind.HISTORY, SlotKind.FEW_SHOT)
CORE = (SlotKind.SYSTEM, SlotKind.INSTRUCTIONS, SlotKind.QUERY)


def short_tokens(n: int) -> str:
    """1024 -> '1k', 6144 -> '6k', 1500 -> '1,500'."""
    return f"{n // 1024}k" if n % 1024 == 0 and n >= 1024 else f"{n:,}"


def core_tokens(manifest: Manifest) -> int:
    """System prompt, instructions, query and the fixed chat-format overhead."""
    return manifest.fixed_overhead_tokens + sum(manifest.slot(slot).used_tokens for slot in CORE)


def action_summary(manifest: Manifest) -> str:
    """One line describing what changed, e.g. ``history: 23 summarised · tools: 16 dropped``."""
    parts = []
    for report in manifest.slots:
        actions = {k: v for k, v in report.actions.items() if k != "kept" and v}
        if actions:
            detail = ", ".join(f"{count} {action}" for action, count in actions.items())
            parts.append(f"{report.slot.label}: {detail}")
    return " · ".join(parts) if parts else "nothing degraded; everything fits"


def budget_table(results: Sequence[AssembledContext]) -> Table:
    """One row per assembly: tokens used by each degradable slot, the core, and the verified total."""
    table = Table(title="Tokens used per slot  (▼ = degraded to fit)", title_justify="left", header_style="bold")
    table.add_column("Window", justify="right")
    table.add_column("Budget", justify="right")
    for slot in DEGRADABLE:
        table.add_column(slot.label.capitalize(), justify="right", style=SLOT_COLOURS[slot])
    table.add_column("Core*", justify="right", style=SLOT_COLOURS[SlotKind.SYSTEM])
    table.add_column("Verified", justify="right", style="bold")
    table.add_column("Fit", justify="right")
    for result in results:
        manifest = result.manifest
        cells = []
        for slot in DEGRADABLE:
            report = manifest.slot(slot)
            cells.append(f"{report.used_tokens:,}{' ▼' if report.degraded else '  '}")
        share = manifest.verified_tokens / manifest.prompt_budget
        fit = Text(f"✓ {share:.0%}", style="green") if manifest.within_budget else Text("✗ over", style="red")
        table.add_row(
            short_tokens(manifest.context_window),
            f"{manifest.prompt_budget:,}",
            *cells,
            f"{core_tokens(manifest):,}",
            f"{manifest.verified_tokens:,}",
            fit,
        )
    return table


def composition_bars(results: Sequence[AssembledContext], width: int = 64) -> Group:
    """A stacked bar per assembly: each slot's share of that assembly's prompt budget."""
    rows: list[RenderableType] = [Text("Prompt composition  (each bar = that window's prompt budget)", style="bold")]
    for result in results:
        manifest = result.manifest
        budget = manifest.prompt_budget
        segments = [(SLOT_COLOURS[SlotKind.SYSTEM], core_tokens(manifest))]
        segments += [(SLOT_COLOURS[slot], manifest.slot(slot).used_tokens) for slot in DEGRADABLE]
        line = Text(f"{short_tokens(manifest.context_window):>4} ")
        drawn = cumulative = 0
        for colour, tokens in segments:
            cumulative += tokens
            end = round(cumulative / budget * width)
            if end > drawn:
                line.append("█" * (end - drawn), style=colour)
                drawn = end
        line.append("·" * max(0, width - drawn), style="bright_black")
        line.append(f" {manifest.verified_tokens:,}/{budget:,}", style="bright_black")
        rows.append(line)
    legend = Text("     ")
    for label, slot in [("core*", SlotKind.SYSTEM)] + [(slot.label, slot) for slot in DEGRADABLE]:
        legend.append("■ ", style=SLOT_COLOURS[slot])
        legend.append(f"{label}  ")
    legend.append("· headroom", style="bright_black")
    rows.append(legend)
    return Group(*rows)


def _kept_cell(manifest: Manifest, slot: SlotKind) -> str:
    report = manifest.slot(slot)
    actions = report.actions
    offered = report.items_in - actions.get("deduplicated", 0)
    cell = f"{report.items_kept}/{offered}"
    if actions.get("summarised"):
        cell += f" + summary of {actions['summarised']}"
    for action in ("compressed", "trimmed"):
        if actions.get(action):
            cell += f" · {actions[action]} {action}"
    return cell


def decisions_table(results: Sequence[AssembledContext]) -> Table:
    """Items kept out of those offered (after near-duplicate removal), per slot and window."""
    table = Table(title="Items kept / offered", title_justify="left", header_style="bold")
    table.add_column("Window", justify="right")
    for slot in DEGRADABLE:
        table.add_column(slot.label.capitalize(), style=SLOT_COLOURS[slot])
    for result in results:
        manifest = result.manifest
        table.add_row(short_tokens(manifest.context_window), *(_kept_cell(manifest, slot) for slot in DEGRADABLE))
    return table


def demo_view(title: str, results: Sequence[AssembledContext]) -> Group:
    """The full ``ctxasm demo`` output."""
    largest = max(results, key=lambda r: r.manifest.prompt_budget).manifest
    duplicates = sum(r.actions.get("deduplicated", 0) for r in largest.slots)
    source_tokens = largest.fixed_overhead_tokens + sum(r.requested_tokens for r in largest.slots)
    header = Text()
    header.append("ctxasm demo", style="bold")
    header.append(f" · {title}\n", style="bold")
    header.append(
        f"Sources need {source_tokens:,} tokens in full ({duplicates} near-duplicates removed) · "
        f"output reserve {largest.output_reserve:,}\ntokenizer {largest.tokenizer} · summariser {largest.summariser}",
        style="bright_black",
    )
    footnote = Text(
        "* Core = system prompt, instructions, query and chat-format overhead: required, never degraded.\n"
        "  Verified = exact re-count of the final rendered messages and tool definitions.",
        style="bright_black",
    )
    return Group(header, Text(""), budget_table(results), Text(""), composition_bars(results), Text(""),
                 decisions_table(results), footnote)  # fmt: skip


def explain_view(result: AssembledContext, *, show_all: bool = False, title: str = "") -> Group:
    """The ``ctxasm explain`` output: budget arithmetic, per-slot accounting and per-item decisions."""
    m = result.manifest
    header = Text()
    header.append(f"ctxasm explain{' · ' + title if title else ''}\n", style="bold")
    header.append(
        f"Budget: window {m.context_window:,} − output reserve {m.output_reserve:,} − safety margin "
        f"{m.safety_margin:,} = {m.prompt_budget:,} prompt tokens\n"
    )
    status = Text("within budget", style="green") if m.within_budget else Text("OVER BUDGET", style="red")
    header.append(f"Verified {m.verified_tokens:,} tokens ({m.verified_tokens / m.prompt_budget:.1%}) · ")
    header.append_text(status)
    header.append(
        f" · estimate {m.estimated_tokens:,} · headroom {m.headroom:,} · overflow repairs {m.overflow_repairs}\n"
        f"Tokenizer {m.tokenizer} · summariser {m.summariser} · "
        f"chat-format overhead {m.fixed_overhead_tokens} tokens · {m.latency_ms:.1f} ms",
        style="bright_black",
    )
    if m.required_only:
        header.append("\nOnly the required items fitted this budget.", style="yellow")

    slots = Table(title="Slots", title_justify="left", header_style="bold")
    slots.add_column("Slot")
    for column in ("Tier", "Kept/in", "Requested", "Floor", "Granted", "Used"):
        slots.add_column(column, justify="right")
    slots.add_column("Degradation steps")
    for r in m.slots:
        slots.add_row(
            Text(r.slot.label, style=SLOT_COLOURS[r.slot]),
            str(r.tier),
            f"{r.items_kept}/{r.items_in}",
            f"{r.requested_tokens:,}",
            f"{r.floor_tokens:,}",
            f"{r.granted_tokens:,}",
            f"{r.used_tokens:,}",
            "\n".join(r.steps) or "-",
        )

    items = Table(title="Items" + ("" if show_all else " not kept verbatim"), title_justify="left", header_style="bold")
    items.add_column("Slot")
    items.add_column("Item")
    items.add_column("Action")
    items.add_column("Tokens", justify="right")
    items.add_column("Reason")
    shown = 0
    for d in m.decisions:
        if d.action == "kept" and not show_all:
            continue
        shown += 1
        tokens = f"{d.tokens_before} → {d.tokens_after}" if d.tokens_after != d.tokens_before else str(d.tokens_before)
        items.add_row(
            Text(d.slot.label, style=SLOT_COLOURS[d.slot]),
            d.item_id,
            Text(d.action, style=ACTION_STYLES[d.action]),
            tokens,
            d.reason,
        )
    parts: list[RenderableType] = [header, Text(""), slots, Text("")]
    parts.append(items if shown else Text("Every item was kept verbatim.", style="green"))
    return Group(*parts)


def assembly_summary(result: AssembledContext) -> Text:
    """A short summary printed (to stderr) by ``ctxasm assemble``."""
    m = result.manifest
    text = Text()
    text.append(f"{m.verified_tokens:,}/{m.prompt_budget:,} prompt tokens", style="green" if m.within_budget else "red")
    text.append(f" · {len(result.messages)} messages · {len(result.tools)} tools · {m.latency_ms:.1f} ms\n")
    text.append(action_summary(m), style="bright_black")
    return text
