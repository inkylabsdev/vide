"""Concatenate clips into one video without re-encoding (ffmpeg concat demuxer)."""

import tempfile
from pathlib import Path

import click
import ffmpeg


@click.command("merge-scenes")
@click.argument(
    "clips",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("merged.mp4"),
    show_default=True,
    help="Path of the merged video.",
)
def cli(clips: tuple[Path, ...], output: Path):
    """Merge CLIPS into a single video, in the order given, using stream copy."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for clip in clips:
            # concat demuxer escaping: ' inside a quoted path becomes '\''
            escaped = str(clip.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        inputs_txt = f.name

    try:
        (
            ffmpeg.input(inputs_txt, format="concat", safe=0)
            .output(str(output), c="copy")
            .overwrite_output()
            .run()
        )
    finally:
        Path(inputs_txt).unlink()

    click.echo(f"Merged {len(clips)} clip(s) into {output}")
