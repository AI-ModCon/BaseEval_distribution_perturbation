"""Command-line interface for dist_pert: dist-perturb."""

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from dist_pert.base import BasePerturber

app = typer.Typer(
    name="dist-perturb",
    help="Apply distribution perturbation to a JSONL file.",
    add_completion=False,
)

_PERTURBER_REGISTRY: dict[str, type] = {}


def _build_perturber(config: dict) -> BasePerturber:
    """Instantiate a perturber from a config dict.

    The config must contain a 'type' key of the form 'module.ClassName'
    (e.g. 'text.contextual_word') that maps to a class in dist_pert, plus
    any keyword arguments accepted by that class's constructor.

    Args:
        config: Parsed YAML config dict with at least a 'type' key.

    Returns:
        An instantiated BasePerturber subclass.

    Raises:
        ValueError: If 'type' is missing or the class cannot be resolved.
    """
    perturber_type = config.pop("type", None)
    if perturber_type is None:
        raise ValueError("Config must contain a 'type' field.")

    ############### resolve class ###############
    import importlib

    cls: type[BasePerturber] | None
    parts = perturber_type.rsplit(".", 1)
    if len(parts) == 2:
        module_path, class_name = parts
        module = importlib.import_module(f"dist_pert.{module_path}")
        cls = getattr(module, class_name, None)
    else:
        # Bare name: search all submodules
        cls = None
        for pkg in ("text", "image", "numeric"):
            try:
                module = importlib.import_module(f"dist_pert.{pkg}")
                cls = getattr(module, parts[0], None)
                if cls is not None:
                    break
            except ImportError:
                continue

    if cls is None:
        raise ValueError(f"Could not resolve perturber type '{perturber_type}'.")

    ############### instantiate ###############
    perturber: BasePerturber = cls(**config)
    return perturber


@app.command()
def run(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input JSONL file.")],
    config: Annotated[Path, typer.Option("--config", "-c", help="Perturbation config YAML.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output JSONL file.")],
    field: Annotated[
        str, typer.Option("--field", "-f", help="JSONL field to perturb.")
    ] = "question",
) -> None:
    """Perturb a single field in every record of a JSONL file.

    Reads INPUT line-by-line, applies the perturbation specified in CONFIG to
    the value of FIELD in each record, and writes the results to OUTPUT.

    Config YAML format::

        type: text.contextual_word.ContextualWordPerturber
        aug_p: 0.1
        model_path: google-bert/bert-base-cased
        model_type: bert

    Args:
        input: Path to the input JSONL file.
        config: Path to the perturbation YAML config file.
        output: Path to write the perturbed JSONL file.
        field: Key in each JSONL record whose value will be perturbed.
    """

    ############### load config ###############
    with config.open() as f:
        cfg = yaml.safe_load(f)

    ############### build perturber ###############
    perturber = _build_perturber(cfg)

    ############### load data ###############
    records: list[dict] = []
    with input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        typer.echo("Input file is empty.", err=True)
        raise typer.Exit(code=1)

    ############### perturb ###############
    values = [r[field] for r in records]
    perturbed_values = perturber(values)

    ############### write output ###############
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record, perturbed in zip(records, perturbed_values, strict=False):
            out_record = {**record, field: perturbed}
            f.write(json.dumps(out_record) + "\n")

    typer.echo(f"Wrote {len(records)} records to {output}.")
