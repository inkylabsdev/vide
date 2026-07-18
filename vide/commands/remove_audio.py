"""Write a copy of a video with the audio track stripped (stream copy)."""

from pathlib import Path

import click
import ffmpeg


@click.command("remove-audio")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output video path. Defaults to ./<video name>_noaudio<ext>.",
)
def cli(video: Path, output: Path | None):
    """Write a copy of VIDEO with its audio track removed, without re-encoding."""
    if output is None:
        output = Path.cwd() / f"{video.stem}_noaudio{video.suffix}"

    (
        ffmpeg.input(str(video))
        .output(str(output), c="copy", an=None)
        .overwrite_output()
        .run()
    )
    click.echo(f"Wrote audio-free copy to {output}")
