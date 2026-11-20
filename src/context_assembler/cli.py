"""``ctxasm`` command line: ``demo``, ``assemble``, ``explain`` and ``models``."""

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .assembler import AssembledContext, BudgetExceededError, ContextAssembler
from .models import ModelRegistry, default_registry
from .report import assembly_summary, demo_view, explain_view
from .schema import Budget
from .spec import LoadedSpec, SpecError, demo_spec, load_spec
from .summarise import ExtractiveSummariser, select_summariser
from .tokenizer import DEFAULT_ENCODING, get_tokenizer

app = typer.Typer(
    name="ctxasm",
    help="Assemble LLM prompts from many sources under a hard, verified token budget.",
    no_args_is_help=True,
    add_completion=False,
)

ModelsFile = Annotated[
    Path | None,
    typer.Option("--models-file", help="YAML/JSON file of extra model profiles (or set CTXASM_MODELS_FILE)."),
]
BudgetOpt = Annotated[int | None, typer.Option("--budget", "-b", help="Prompt budget in tokens (no output reserve).")]
ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Model profile from the registry.")]
WindowOpt = Annotated[int | None, typer.Option("--window", help="Context window in tokens.")]
ReserveOpt = Annotated[int | None, typer.Option("--reserve", help="Output tokens to reserve.")]
EncodingOpt = Annotated[str | None, typer.Option("--encoding", "-e", help="o200k_base, cl100k_base or approx.")]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="Never call a model (extractive summariser).")]
SvgOpt = Annotated[Path | None, typer.Option("--save-svg", help="Also save the output as an SVG image.")]


def _fail(message: str, code: int = 1) -> typer.Exit:
    Console(stderr=True).print(f"[red]error:[/red] {message}", highlight=False)
    return typer.Exit(code)


def _registry(models_file: Path | None) -> ModelRegistry:
    registry = default_registry()
    path = models_file or (Path(os.environ["CTXASM_MODELS_FILE"]) if os.environ.get("CTXASM_MODELS_FILE") else None)
    if path is not None:
        try:
            registry.load_file(path)
        except (OSError, ValueError) as exc:
            raise _fail(f"could not load model profiles from {path.name}: {exc}") from None
    return registry


def _budget_override(
    registry: ModelRegistry, budget: int | None, model: str | None, window: int | None, reserve: int | None
) -> Budget | None:
    try:
        if budget is not None:
            return Budget.prompt_only(budget)
        if model is not None:
            return registry.get(model).budget(output_reserve=reserve)
        if window is not None:
            return Budget(context_window=window, output_reserve=reserve or 0)
    except (KeyError, ValueError) as exc:
        raise _fail(str(exc.args[0]) if exc.args else str(exc)) from None
    return None


def _load(
    spec: Path, models_file: Path | None, budget: int | None, model: str | None, window: int | None, reserve: int | None
) -> LoadedSpec:
    registry = _registry(models_file)
    override = _budget_override(registry, budget, model, window, reserve)
    try:
        loaded = load_spec(spec, registry=registry, budget=override)
    except SpecError as exc:
        raise _fail(str(exc)) from None
    if override is None and reserve is not None:
        current = loaded.request.budget
        try:
            new_budget = Budget(context_window=current.context_window, output_reserve=reserve,
                                safety_margin=current.safety_margin)  # fmt: skip
        except ValueError as exc:
            raise _fail(str(exc)) from None
        loaded = LoadedSpec(loaded.request.with_budget(new_budget), loaded.encoding, loaded.model, loaded.title)
    return loaded


def _run(loaded: LoadedSpec, encoding: str | None, offline: bool) -> AssembledContext:
    try:
        tokenizer = get_tokenizer(encoding or loaded.encoding)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    assembler = ContextAssembler(tokenizer, summariser=select_summariser(offline=offline))
    try:
        return assembler.assemble(loaded.request)
    except BudgetExceededError as exc:
        raise _fail(str(exc), code=2) from None


def _console(width: int, record: bool) -> Console:
    return Console(width=width, record=record, highlight=False)


def _save_svg(console: Console, path: Path, title: str) -> None:
    """Write the recorded output as an SVG, without the exporter's template comment."""
    svg = re.sub(r"\s*<!--.*?-->", "", console.export_svg(title=title), count=1, flags=re.S)
    path.write_text(svg, encoding="utf-8")


@app.command()
def demo(
    windows: Annotated[
        str, typer.Option(help="Comma-separated context windows to assemble for.")
    ] = "2048,4096,6144,8192",
    reserve: Annotated[int, typer.Option(help="Output tokens reserved in every window.")] = 1024,
    encoding: Annotated[
        str, typer.Option("--encoding", "-e", help="o200k_base, cl100k_base or approx.")
    ] = DEFAULT_ENCODING,
    width: Annotated[int, typer.Option(help="Output width in characters.")] = 100,
    save_svg: SvgOpt = None,
) -> None:
    """Assemble the bundled billing-support scenario at several context windows (offline)."""
    try:
        sizes = [int(part) for part in windows.split(",") if part.strip()]
        budgets = [Budget(context_window=size, output_reserve=reserve) for size in sizes]
        tokenizer = get_tokenizer(encoding)
    except ValueError as exc:
        raise _fail(str(exc)) from None
    if not budgets:
        raise _fail("give at least one window, e.g. --windows 2048,4096")
    loaded = demo_spec()
    assembler = ContextAssembler(tokenizer, summariser=ExtractiveSummariser())
    try:
        results = [assembler.assemble(loaded.request.with_budget(budget)) for budget in budgets]
    except BudgetExceededError as exc:
        raise _fail(str(exc), code=2) from None
    console = _console(width, record=save_svg is not None)
    console.print(demo_view(loaded.title, results))
    if save_svg is not None:
        _save_svg(console, save_svg, "ctxasm demo")


@app.command()
def assemble(
    spec: Annotated[Path, typer.Argument(help="Spec file (.yaml, .yml or .json).")],
    budget: BudgetOpt = None,
    model: ModelOpt = None,
    window: WindowOpt = None,
    reserve: ReserveOpt = None,
    encoding: EncodingOpt = None,
    offline: OfflineOpt = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON here instead of stdout.")
    ] = None,
    models_file: ModelsFile = None,
) -> None:
    """Assemble a spec and print the messages, tools and manifest as JSON."""
    loaded = _load(spec, models_file, budget, model, window, reserve)
    result = _run(loaded, encoding, offline)
    payload = result.model_dump(mode="json")
    if output is not None:
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        typer.echo(json.dumps(payload, indent=2))
    Console(stderr=True, highlight=False).print(assembly_summary(result))


@app.command()
def explain(
    spec: Annotated[Path, typer.Argument(help="Spec file (.yaml, .yml or .json).")],
    budget: BudgetOpt = None,
    model: ModelOpt = None,
    window: WindowOpt = None,
    reserve: ReserveOpt = None,
    encoding: EncodingOpt = None,
    offline: OfflineOpt = False,
    all_items: Annotated[bool, typer.Option("--all", help="List kept items too, not only degraded ones.")] = False,
    width: Annotated[int, typer.Option(help="Output width in characters.")] = 120,
    models_file: ModelsFile = None,
    save_svg: SvgOpt = None,
) -> None:
    """Show how a spec was budgeted: per-slot grants and the fate of every item."""
    loaded = _load(spec, models_file, budget, model, window, reserve)
    result = _run(loaded, encoding, offline)
    console = _console(width, record=save_svg is not None)
    console.print(explain_view(result, show_all=all_items, title=loaded.title))
    if save_svg is not None:
        _save_svg(console, save_svg, "ctxasm explain")


@app.command("models")
def list_models(models_file: ModelsFile = None) -> None:
    """List the model profiles in the registry (built-in profiles are illustrative)."""
    table = Table(header_style="bold")
    for column, justify in [("Model", "left"), ("Context window", "right"), ("Output reserve", "right"),
                            ("Encoding", "left"), ("Notes", "left")]:  # fmt: skip
        table.add_column(column, justify=justify)  # type: ignore[arg-type]
    for spec in _registry(models_file):
        table.add_row(spec.name, f"{spec.context_window:,}", f"{spec.output_reserve:,}", spec.encoding, spec.notes)
    Console(highlight=False).print(table)


def main() -> None:
    """Console-script entry point."""
    for stream in (sys.stdout, sys.stderr):  # piped output on Windows defaults to a legacy code page
        if isinstance(stream, io.TextIOWrapper) and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8")
    app()
