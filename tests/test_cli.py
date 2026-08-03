"""Tests for the dist-perturb CLI."""

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from dist_pert.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI colour/style escape sequences from CLI output."""
    return _ANSI_RE.sub("", text)


def test_run_numeric_perturbation(tmp_path: Path) -> None:
    """The CLI perturbs a numeric field and writes one output record per input."""
    input_path = tmp_path / "input.jsonl"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "output.jsonl"

    records = [{"id": 1, "value": 1.0}, {"id": 2, "value": 2.0}]
    input_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    config_path.write_text("type: numeric.additive.AdditiveGaussianPerturber\nsigma: 0.0\n")

    result = runner.invoke(
        app,
        [
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--field",
            "value",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    out_records = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    assert len(out_records) == len(records)
    assert {r["id"] for r in out_records} == {1, 2}


def test_help() -> None:
    """The CLI exposes a help message describing its options.

    The output is rendered by rich, which injects ANSI styling and wraps text to
    the terminal width. We force a wide terminal and strip ANSI codes so the
    option names appear as stable, unwrapped substrings across environments.
    """
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200", "TERM": "dumb"})
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--input" in output
    assert "--config" in output
    assert "--output" in output
