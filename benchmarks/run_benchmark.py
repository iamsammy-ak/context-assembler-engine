"""Benchmark: required-fact retention, budget violations, wasted tokens and latency over a budget sweep.

Deterministic and offline: seeded synthetic scenarios (see ``context_assembler/bench/synthetic.py``), the
real tiktoken tokenizer, no model calls. Compares naive head truncation, keep-most-recent truncation,
a fixed proportional split and the engine (strategies documented in ``bench/baselines.py``).

Reproduce from the repository root:

    PYTHONPATH=src python benchmarks/run_benchmark.py

Writes ``benchmarks/results/benchmark.json``, ``benchmarks/results/benchmark.md`` and
``assets/benchmark.png``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import tiktoken

from context_assembler import Budget, __version__, count_prompt, get_tokenizer
from context_assembler.bench.baselines import HeadTruncation, all_strategies
from context_assembler.bench.metrics import fact_kept, percentile, query_kept, wasted_tokens
from context_assembler.bench.synthetic import make_scenario

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "head": "Head truncation",
    "recent": "Keep most recent",
    "proportional": "Fixed proportional split",
    "engine": "Context Assembler Engine",
}
COLOURS = {"engine": "#2a78d6", "head": "#eb6834", "recent": "#1baf7a", "proportional": "#eda100"}
MARKERS = {"engine": "o", "head": "s", "recent": "^", "proportional": "D"}
LOCATIONS = ("history_early", "history_recent", "memory", "retrieved_top", "retrieved_mid", "tool")
COMMAND = "PYTHONPATH=src python benchmarks/run_benchmark.py"


def run(seeds: int, budgets: list[int], encoding: str) -> dict[str, Any]:
    tokenizer = get_tokenizer(encoding, fallback=False)
    tokenizer.count("warm up the encoder before timing anything")
    strategies = all_strategies(tokenizer)
    records: list[dict[str, Any]] = []
    source_tokens: list[int] = []
    for seed in range(seeds):
        scenario = make_scenario(seed)
        full = HeadTruncation(tokenizer).assemble(scenario.request.with_budget(Budget.prompt_only(10**6)))
        source_tokens.append(count_prompt(tokenizer, full.messages, full.tools))
        for budget in budgets:
            request = scenario.request.with_budget(Budget.prompt_only(budget))
            for strategy in strategies:
                tokenizer.clear_cache()  # every strategy starts with a cold token-count cache
                started = time.perf_counter()
                result = strategy.assemble(request)
                latency_ms = (time.perf_counter() - started) * 1000
                verified = count_prompt(tokenizer, result.messages, result.tools, use_cache=False)
                kept = {fact.location: fact_kept(fact, result.messages, result.tools) for fact in scenario.facts}
                records.append(
                    {
                        "strategy": strategy.name,
                        "seed": seed,
                        "budget": budget,
                        "verified_tokens": verified,
                        "violation": verified > budget,
                        "facts_kept": kept,
                        "query_kept": query_kept(request.query, result.messages),
                        "wasted_tokens": wasted_tokens(budget, verified, result.lossless),
                        "latency_ms": latency_ms,
                        "repairs": result.repairs,
                        "estimated_tokens": result.estimated_tokens,
                    }
                )
    return {"records": records, "source_tokens": source_tokens, "tokenizer": tokenizer.name}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    facts = [sum(r["facts_kept"].values()) / len(LOCATIONS) for r in rows]
    latencies = [r["latency_ms"] for r in rows]
    violations = [r for r in rows if r["violation"]]
    return {
        "assemblies": len(rows),
        "fact_retention_pct": round(100 * statistics.mean(facts), 1),
        "query_kept_pct": round(100 * statistics.mean(r["query_kept"] for r in rows), 1),
        "violations": len(violations),
        "violation_rate_pct": round(100 * len(violations) / len(rows), 1),
        "mean_overshoot_tokens": round(statistics.mean(r["verified_tokens"] - r["budget"] for r in violations), 1)
        if violations
        else 0.0,
        "mean_wasted_tokens": round(statistics.mean(r["wasted_tokens"] for r in rows), 1),
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
    }


def aggregate(raw: dict[str, Any], seeds: int, budgets: list[int], encoding: str) -> dict[str, Any]:
    records = raw["records"]
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_budget: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_strategy[record["strategy"]].append(record)
        by_budget[(record["strategy"], record["budget"])].append(record)
    names = list(LABELS)
    engine = by_strategy["engine"]
    errors = [r["estimated_tokens"] - r["verified_tokens"] for r in engine if r["estimated_tokens"] is not None]
    source = raw["source_tokens"]
    return {
        "command": COMMAND,
        "config": {"seeds": seeds, "budgets": budgets, "encoding": encoding, "facts_per_scenario": len(LOCATIONS)},
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "context_assembler": __version__,
            "tiktoken": tiktoken.__version__,
            "tokenizer": raw["tokenizer"],
        },
        "scenario_source_tokens": {"mean": round(statistics.mean(source)), "min": min(source), "max": max(source)},
        "overall": {name: summarise(by_strategy[name]) for name in names},
        "per_budget": {
            name: {str(budget): summarise(by_budget[(name, budget)]) for budget in budgets} for name in names
        },
        "retention_by_location_pct": {
            name: {
                location: round(100 * statistics.mean(r["facts_kept"][location] for r in by_strategy[name]), 1)
                for location in LOCATIONS
            }
            for name in names
        },
        "engine_estimates": {
            "overflow_repairs": sum(r["repairs"] for r in engine),
            "estimate_minus_verified_mean": round(statistics.mean(errors), 2),
            "estimate_minus_verified_min": min(errors),
            "estimate_minus_verified_max": max(errors),
        },
    }


def markdown(result: dict[str, Any]) -> str:
    names = list(LABELS)
    budgets = result["config"]["budgets"]
    env, source = result["environment"], result["scenario_source_tokens"]
    lines = [
        "# Benchmark results",
        "",
        f"Reproduce: `{result['command']}`",
        "",
        f"Data: {result['config']['seeds']} seeded **synthetic** support conversations (scenario sizes "
        f"{source['min']:,}-{source['max']:,} tokens, mean {source['mean']:,}), each with "
        f"{result['config']['facts_per_scenario']} labelled required facts, assembled at prompt budgets of "
        f"{', '.join(f'{b:,}' for b in budgets)} tokens. Tokenizer `{env['tokenizer']}` (tiktoken "
        f"{env['tiktoken']}), Python {env['python']} on {env['os']}. No model calls.",
        "",
        "## Overall (all budgets)",
        "",
        "| Strategy | Fact retention | Query kept | Budget violations | Mean overshoot | Mean wasted tokens "
        "| Latency p50 | Latency p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        o = result["overall"][name]
        lines.append(
            f"| {LABELS[name]} | {o['fact_retention_pct']:.1f}% | {o['query_kept_pct']:.1f}% | "
            f"{o['violations']} / {o['assemblies']} ({o['violation_rate_pct']:.1f}%) | "
            f"{o['mean_overshoot_tokens']:.0f} | {o['mean_wasted_tokens']:.0f} | {o['latency_p50_ms']:.1f} ms | "
            f"{o['latency_p95_ms']:.1f} ms |"
        )
    for title, key, fmt in [
        ("Required-fact retention by budget", "fact_retention_pct", "{:.1f}%"),
        ("Budget violations by budget (of each cell's assemblies)", "violations", "{}"),
        ("Mean wasted tokens by budget", "mean_wasted_tokens", "{:.0f}"),
    ]:
        lines += ["", f"## {title}", "", "| Budget | " + " | ".join(LABELS[n] for n in names) + " |",
                  "|---:|" + "---:|" * len(names)]  # fmt: skip
        for budget in budgets:
            cells = [fmt.format(result["per_budget"][name][str(budget)][key]) for name in names]
            lines.append(f"| {budget:,} | " + " | ".join(cells) + " |")
    header = "| Location | " + " | ".join(LABELS[n] for n in names) + " |"
    lines += ["", "## Retention by where the fact lives", "", header, "|---|" + "---:|" * len(names)]
    for location in LOCATIONS:
        cells = [f"{result['retention_by_location_pct'][name][location]:.1f}%" for name in names]
        lines.append(f"| `{location}` | " + " | ".join(cells) + " |")
    e = result["engine_estimates"]
    lines += [
        "",
        "## Engine estimate accuracy",
        "",
        f"Estimated minus verified tokens per assembly: mean {e['estimate_minus_verified_mean']}, "
        f"min {e['estimate_minus_verified_min']}, max {e['estimate_minus_verified_max']}. Overflow repairs "
        f"triggered: {e['overflow_repairs']}.",
        "",
        "## Definitions and assumptions",
        "",
        "- **Fact retention**: share of the labelled facts whose marker survives in the rendered prompt (a "
        "tool fact counts if that tool's schema is sent).",
        "- **Budget violation**: the rendered messages and tools, re-counted with the tokenizer and the chat-format "
        "overheads, exceed the prompt budget.",
        "- **Wasted tokens**: budget left unused while some content was dropped or cut; 0 when everything fitted "
        "or the budget was violated.",
        "- **Latency**: wall-clock time of one assembly with a cold token-count cache, measured on a shared "
        "laptop with other work running; compare strategies with each other rather than reading the absolute "
        "values as a hardware benchmark.",
        "- The baselines count tokens the way their simple counterparts usually do (see "
        "`src/context_assembler/bench/baselines.py`): head truncation ignores per-message chat overhead, "
        "keep-most-recent only budgets the history, the fixed split never re-verifies. Their violations "
        "come from those choices.",
        "- The engine is not told which items hold facts. The facts are planted in places a real assistant "
        "would need them, and the scenario generator's scores are noisy on purpose.",
    ]
    return "\n".join(lines) + "\n"


def chart(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(LABELS)
    budgets = result["config"]["budgets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="white")
    panels = [
        (axes[0], "fact_retention_pct", "Required-fact retention (%)", "Facts kept in the prompt"),
        (axes[1], "violation_rate_pct", "Assemblies over budget (%)", "Budget violations"),
    ]
    for ax, key, ylabel, title in panels:
        for name in names:
            values = [result["per_budget"][name][str(b)][key] for b in budgets]
            width = 2.6 if name == "engine" else 1.8
            ax.plot(budgets, values, color=COLOURS[name], marker=MARKERS[name], markersize=6, linewidth=width,
                    label=LABELS[name], zorder=3 if name == "engine" else 2)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b // 1024}k" if b % 1024 == 0 else f"{b / 1024:.1f}k" for b in budgets])
        ax.set_xlabel("Prompt budget (tokens)", color="#52514e")
        ax.set_ylabel(ylabel, color="#52514e")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#0b0b0b")
        ax.set_ylim(-4, 104)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
    for name in ("engine", "proportional"):
        if result["overall"][name]["violations"] == 0:
            axes[1].annotate(
                f"{LABELS[name]}: 0 at every budget",
                xy=(budgets[0], 0),
                xytext=(budgets[0], 9 if name == "engine" else 17),
                ha="left",
                fontsize=9,
                color="#52514e",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    seeds = result["config"]["seeds"]
    fig.suptitle(f"{seeds} synthetic scenarios × {len(budgets)} budgets · tokenizer "
                 f"{result['environment']['tokenizer']}", x=0.01, ha="left", fontsize=10, color="#52514e")  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--budgets", default="512,1024,1536,2048,3072,4096,6144,8192")
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--chart", type=Path, default=ROOT / "assets" / "benchmark.png")
    parser.add_argument("--from-json", type=Path, help="Redraw the chart from saved results without re-running.")
    args = parser.parse_args(argv)
    if args.from_json:
        chart(json.loads(args.from_json.read_text(encoding="utf-8")), args.chart)
        return 0
    budgets = [int(b) for b in args.budgets.split(",")]
    raw = run(args.seeds, budgets, args.encoding)
    result = aggregate(raw, args.seeds, budgets, args.encoding)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "benchmark.md").write_text(markdown(result), encoding="utf-8")
    chart(result, args.chart)
    for name in LABELS:
        o = result["overall"][name]
        print(
            f"{LABELS[name]:<28} retention {o['fact_retention_pct']:5.1f}%  violations {o['violations']:4d}  "
            f"wasted {o['mean_wasted_tokens']:6.0f}  p50 {o['latency_p50_ms']:6.1f} ms  "
            f"p95 {o['latency_p95_ms']:6.1f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
