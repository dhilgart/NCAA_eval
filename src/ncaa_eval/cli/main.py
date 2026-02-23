"""Typer CLI application for NCAA_eval training."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from ncaa_eval.model import get_model, list_models
from ncaa_eval.model.base import Model
from ncaa_eval.model.registry import ModelNotFoundError

app = typer.Typer(help="NCAA_eval model training CLI")
console = Console()


@app.callback()
def _callback() -> None:
    """NCAA_eval CLI — model training and evaluation."""


def _instantiate_model(model_cls: type[Model], config_path: Path | None) -> Model:
    """Instantiate a model, optionally overriding its config from JSON.

    When *config_path* is given, creates a default instance to discover
    the config class, then validates the JSON overrides through that
    Pydantic model and reinstantiates.
    """
    if config_path is not None:
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(code=1)
        override = json.loads(config_path.read_text())
        default = model_cls()
        config_cls = type(default.get_config())
        config_obj = config_cls(**override)
        return model_cls(config_obj)  # type: ignore[call-arg]
    return model_cls()


@app.command()
def train(  # noqa: PLR0913
    model: str = typer.Option(..., "--model", help="Registered model plugin name"),
    start_year: int = typer.Option(2015, "--start-year", help="First season year (inclusive)"),
    end_year: int = typer.Option(2025, "--end-year", help="Last season year (inclusive)"),
    data_dir: Path = typer.Option(Path("data/"), "--data-dir", help="Local Parquet data directory"),
    output_dir: Path = typer.Option(Path("data/"), "--output-dir", help="Output directory for run artifacts"),
    config: Path | None = typer.Option(None, "--config", help="Path to JSON config override"),
) -> None:
    """Train a model on NCAA basketball data and persist run artifacts."""
    if start_year > end_year:
        console.print(f"[red]Error: --start-year ({start_year}) must be ≤ --end-year ({end_year})[/red]")
        raise typer.Exit(code=1)

    try:
        model_cls = get_model(model)
    except ModelNotFoundError:
        available = list_models()
        console.print(f"[red]Error: Unknown model {model!r}[/red]")
        console.print(f"Available models: {', '.join(available)}")
        raise typer.Exit(code=1)

    model_instance = _instantiate_model(model_cls, config)

    from ncaa_eval.cli.train import run_training

    run_training(
        model_instance,
        start_year=start_year,
        end_year=end_year,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model,
    )
