"""Command-line interface for dormancy-detect."""

from __future__ import annotations

import os
import sys

import click

from .scanner import DormancyScanner


@click.group()
@click.version_option(package_name="dormancy_detect")
def main() -> None:
    """Detect temporal attack patterns in multi-session AI conversations."""


@main.command()
@click.option(
    "--sessions",
    required=True,
    type=click.Path(exists=True),
    help="Path to sessions directory or single JSON file.",
)
@click.option(
    "--memory",
    type=click.Path(exists=True),
    default=None,
    help="Optional path to memory state directory.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output file path (.json or .html).",
)
@click.option(
    "--judge-model",
    default=None,
    help="LLM judge model (e.g. anthropic/claude-sonnet-4.5, ollama/llama3.1:8b).",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for the judge model. Defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY env var.",
)
@click.option("--penalty", default=3.0, type=float, help="PELT penalty parameter.")
@click.option("--decay-rate", default=0.1, type=float, help="Suspicion decay rate.")
@click.option(
    "--format",
    "fmt",
    default="auto",
    type=click.Choice(["auto", "json", "chatgpt", "claude", "plain"], case_sensitive=False),
    help="Input format. Default: auto-detect.",
)
def analyse(
    sessions: str,
    memory: str | None,
    output: str,
    judge_model: str | None,
    api_key: str | None,
    penalty: float,
    decay_rate: float,
    fmt: str,
) -> None:
    """Analyse conversation sessions for dormancy attack patterns."""
    # Resolve API key from env if not provided
    if api_key is None and judge_model is not None:
        if "anthropic" in judge_model:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    from pathlib import Path

    sessions_path = Path(sessions)
    scanner = DormancyScanner(
        sessions_dir=sessions_path if sessions_path.is_dir() else None,
        sessions_file=sessions_path if sessions_path.is_file() else None,
        memory_dir=memory,
        judge_model=judge_model,
        api_key=api_key,
        penalty=penalty,
        decay_rate=decay_rate,
        fmt=fmt,
    )

    click.echo("Analysing sessions...")
    try:
        timeline = scanner.analyse()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    output_path = Path(output)
    if output_path.suffix == ".html":
        timeline.to_html(output_path)
    else:
        timeline.to_json(output_path)

    # Summary
    red = sum(1 for s in timeline.output.sessions if s.risk_level.value == "red")
    amber = sum(1 for s in timeline.output.sessions if s.risk_level.value == "amber")
    green = sum(1 for s in timeline.output.sessions if s.risk_level.value == "green")
    click.echo(
        f"Done. {len(timeline.output.sessions)} sessions analysed: "
        f"{red} red, {amber} amber, {green} green. "
        f"{len(timeline.output.patterns)} dormancy pattern(s) detected."
    )
    click.echo(f"Output written to {output_path}")
