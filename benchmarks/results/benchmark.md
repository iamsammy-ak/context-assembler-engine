# Benchmark results

Reproduce: `PYTHONPATH=src python benchmarks/run_benchmark.py`

Data: 40 seeded **synthetic** support conversations (scenario sizes 4,746-5,492 tokens, mean 5,105), each with 6 labelled required facts, assembled at prompt budgets of 512, 1,024, 1,536, 2,048, 3,072, 4,096, 6,144, 8,192 tokens. Tokenizer `o200k_base` (tiktoken 0.14.0), Python 3.13.2 on Windows. No model calls.

## Overall (all budgets)

| Strategy | Fact retention | Query kept | Budget violations | Mean overshoot | Mean wasted tokens | Latency p50 | Latency p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Head truncation | 61.1% | 25.0% | 209 / 320 (65.3%) | 174 | 3 | 10.6 ms | 41.7 ms |
| Keep most recent | 75.6% | 100.0% | 227 / 320 (70.9%) | 2224 | 1 | 11.0 ms | 41.5 ms |
| Fixed proportional split | 66.2% | 100.0% | 0 / 320 (0.0%) | 0 | 902 | 8.2 ms | 39.0 ms |
| Context Assembler Engine | 88.6% | 100.0% | 0 / 320 (0.0%) | 0 | 8 | 37.6 ms | 108.5 ms |

## Required-fact retention by budget

| Budget | Head truncation | Keep most recent | Fixed proportional split | Context Assembler Engine |
|---:|---:|---:|---:|---:|
| 512 | 16.7% | 66.7% | 31.2% | 45.0% |
| 1,024 | 23.8% | 66.7% | 42.1% | 81.2% |
| 1,536 | 29.2% | 66.7% | 49.2% | 87.1% |
| 2,048 | 37.1% | 66.7% | 56.7% | 95.4% |
| 3,072 | 82.9% | 66.7% | 71.2% | 100.0% |
| 4,096 | 99.6% | 71.2% | 87.5% | 100.0% |
| 6,144 | 100.0% | 100.0% | 95.0% | 100.0% |
| 8,192 | 100.0% | 100.0% | 97.1% | 100.0% |

## Budget violations by budget (of each cell's assemblies)

| Budget | Head truncation | Keep most recent | Fixed proportional split | Context Assembler Engine |
|---:|---:|---:|---:|---:|
| 512 | 15 | 40 | 0 | 0 |
| 1,024 | 34 | 40 | 0 | 0 |
| 1,536 | 40 | 40 | 0 | 0 |
| 2,048 | 40 | 40 | 0 | 0 |
| 3,072 | 40 | 40 | 0 | 0 |
| 4,096 | 40 | 27 | 0 | 0 |
| 6,144 | 0 | 0 | 0 | 0 |
| 8,192 | 0 | 0 | 0 | 0 |

## Mean wasted tokens by budget

| Budget | Head truncation | Keep most recent | Fixed proportional split | Context Assembler Engine |
|---:|---:|---:|---:|---:|
| 512 | 18 | 0 | 131 | 6 |
| 1,024 | 3 | 0 | 176 | 6 |
| 1,536 | 0 | 0 | 252 | 8 |
| 2,048 | 0 | 0 | 335 | 10 |
| 3,072 | 0 | 0 | 497 | 10 |
| 4,096 | 0 | 5 | 711 | 24 |
| 6,144 | 0 | 0 | 1741 | 0 |
| 8,192 | 0 | 0 | 3369 | 0 |

## Retention by where the fact lives

| Location | Head truncation | Keep most recent | Fixed proportional split | Context Assembler Engine |
|---|---:|---:|---:|---:|
| `history_early` | 52.8% | 25.0% | 37.5% | 100.0% |
| `history_recent` | 50.0% | 28.4% | 96.6% | 97.2% |
| `memory` | 100.0% | 100.0% | 76.2% | 88.4% |
| `retrieved_top` | 49.4% | 100.0% | 100.0% | 92.8% |
| `retrieved_mid` | 37.5% | 100.0% | 52.5% | 63.4% |
| `tool` | 77.2% | 100.0% | 34.7% | 89.7% |

## Engine estimate accuracy

Estimated minus verified tokens per assembly: mean 2.58, min 2, max 3. Overflow repairs triggered: 0.

## Definitions and assumptions

- **Fact retention**: share of the labelled facts whose marker survives in the rendered prompt (a tool fact counts if that tool's schema is sent).
- **Budget violation**: the rendered messages and tools, re-counted with the tokenizer and the chat-format overheads, exceed the prompt budget.
- **Wasted tokens**: budget left unused while some content was dropped or cut; 0 when everything fitted or the budget was violated.
- **Latency**: wall-clock time of one assembly with a cold token-count cache, measured on a shared laptop with other work running; compare strategies with each other rather than reading the absolute values as a hardware benchmark.
- The baselines count tokens the way their simple counterparts usually do (see `src/context_assembler/bench/baselines.py`): head truncation ignores per-message chat overhead, keep-most-recent only budgets the history, the fixed split never re-verifies. Their violations come from those choices.
- The engine is not told which items hold facts. The facts are planted in places a real assistant would need them, and the scenario generator's scores are noisy on purpose.
