from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_assembler.cli import app
from context_assembler.spec import SpecError, demo_spec, load_spec, parse_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
runner = CliRunner()


def test_example_spec_loads() -> None:
    loaded = load_spec(EXAMPLES / "incident_assistant.yaml")
    assert loaded.request.budget.prompt_budget == 1024 - 256
    assert loaded.encoding == "o200k_base" and loaded.title.startswith("On-call")


def test_json_spec_and_model_lookup(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"model": "generic-4k", "query": "hello", "system": "be nice"}))
    loaded = load_spec(path)
    assert loaded.model == "generic-4k" and loaded.encoding == "cl100k_base"
    assert loaded.request.budget.prompt_budget == 4096 - 512


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"query": "q"}, "needs a 'model' or a 'budget'"),
        ({"model": "nope", "query": "q"}, "unknown model"),
        ({"budget": {"context_window": 100}, "query": "q", "bogus": 1}, "invalid spec"),
        (["not", "a", "mapping"], "must be a mapping"),
    ],
)
def test_invalid_specs_raise_clear_errors(data: object, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_spec(data)


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("query: [unclosed\n")
    with pytest.raises(SpecError, match="could not parse"):
        load_spec(bad)


def test_demo_spec_is_bundled() -> None:
    loaded = demo_spec()
    assert len(loaded.request.tools) >= 10 and len(loaded.request.history) >= 20


def test_cli_demo_runs_offline() -> None:
    result = runner.invoke(app, ["demo", "--windows", "2048,8192"])
    assert result.exit_code == 0, result.output
    assert "Tokens used per slot" in result.output and "Items kept / offered" in result.output


def test_cli_demo_can_save_svg(tmp_path: Path) -> None:
    target = tmp_path / "demo.svg"
    result = runner.invoke(app, ["demo", "--windows", "4096", "--save-svg", str(target)])
    assert result.exit_code == 0 and target.read_text(encoding="utf-8").startswith("<svg")


def test_cli_assemble_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "prompt.json"
    result = runner.invoke(app, ["assemble", str(EXAMPLES / "incident_assistant.yaml"), "--budget", "500",
                                 "--offline", "-o", str(out)])  # fmt: skip
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["manifest"]["verified_tokens"] <= 500
    assert payload["messages"][-1]["role"] == "user"


def test_cli_explain_and_models() -> None:
    result = runner.invoke(app, ["explain", str(EXAMPLES / "incident_assistant.yaml"), "--all"])
    assert result.exit_code == 0, result.output
    assert "Slots" in result.output and "within budget" in result.output
    models = runner.invoke(app, ["models", "--models-file", str(EXAMPLES / "custom_models.yaml")])
    assert models.exit_code == 0 and "team-small" in models.output


def test_cli_reports_budget_errors_with_exit_code_2() -> None:
    result = runner.invoke(app, ["assemble", str(EXAMPLES / "incident_assistant.yaml"), "--budget", "40"])
    assert result.exit_code == 2


def test_cli_model_and_reserve_overrides() -> None:
    result = runner.invoke(app, ["explain", str(EXAMPLES / "incident_assistant.yaml"), "--model", "team-small",
                                 "--models-file", str(EXAMPLES / "custom_models.yaml")])  # fmt: skip
    assert result.exit_code == 0 and "1,152 prompt tokens" in result.output
    reserve = runner.invoke(app, ["explain", str(EXAMPLES / "incident_assistant.yaml"), "--reserve", "500"])
    assert reserve.exit_code == 0 and "524 prompt tokens" in reserve.output
    unknown = runner.invoke(app, ["explain", str(EXAMPLES / "incident_assistant.yaml"), "--model", "nope"])
    assert unknown.exit_code == 1
