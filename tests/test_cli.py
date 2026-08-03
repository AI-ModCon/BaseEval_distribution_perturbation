"""Tests for the dist-perturb CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from dist_pert.cli import app

runner = CliRunner()


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
    """The CLI exposes a help message describing its options."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--config" in result.output
    assert "--output" in result.output
