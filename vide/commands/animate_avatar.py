"""Animate an avatar with a reference image and an audio track via local Cog."""

import subprocess
from pathlib import Path

import click

DEFAULT_MODEL = "longcat-avatar"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "longcat-video-avatar-1.5"
)


@click.command("animate-avatar")
@click.argument(
    "reference_image", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output video path. Defaults to ./<audio name>_avatar.mp4.",
)
@click.option(
    "--model",
    type=click.Choice([DEFAULT_MODEL]),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Avatar animation model to use.",
)
def cli(reference_image: Path, audio: Path, output: Path | None, model: str):
    """Animate REFERENCE_IMAGE with AUDIO via a local Cog package."""
    _ = model
    model_dir = DEFAULT_MODEL_DIR

    if output is None:
        output = Path.cwd() / f"{audio.stem}_avatar.mp4"

    if not model_dir.exists():
        raise click.ClickException(
            f"Model directory not found: {model_dir}. "
            "Expected a local Cog package at models/longcat-video-avatar-1.5."
        )

    command = [
        "cog",
        "predict",
        "-i",
        f"reference_image=@{reference_image}",
        "-i",
        f"audio=@{audio}",
        "-o",
        f"output_video={output}",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=model_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise click.ClickException("`cog` was not found on PATH.") from exc

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise click.ClickException(f"cog predict failed: {error}")

    click.echo(f"Animated avatar written to {output}")
