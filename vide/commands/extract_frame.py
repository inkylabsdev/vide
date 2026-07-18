"""Extract a single frame from a video as an image."""

import re
from pathlib import Path

import click
import ffmpeg

# ffmpeg time syntax without the sign: [[HH:]MM:]SS[.ms]
_TIME = re.compile(r"^(?:\d+:){0,2}\d+(?:\.\d+)?$")


def _seconds(time: str) -> float:
    total = 0.0
    for part in time.split(":"):
        total = total * 60 + float(part)
    return total


def _validate_time(ctx, param, value):
    if value == "-1" or _TIME.match(value):
        return value
    raise click.BadParameter(
        "expected [[HH:]MM:]SS[.ms] (e.g. 00:01.11), or -1 for the last frame"
    )


@click.command("extract-frame")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--time",
    "-t",
    "time",
    default="00:00",
    callback=_validate_time,
    help="Timestamp of the frame ([[HH:]MM:]SS[.ms], e.g. 00:01.11); "
    "-1 picks the last frame. Defaults to 00:00, the first frame.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output image path; the extension picks the format. "
    "Defaults to ./<video name>_<time>.png.",
)
def cli(video: Path, time: str, output: Path | None):
    """Extract the frame of VIDEO at --time as an image."""
    if output is None:
        label = "last" if time == "-1" else time.replace(":", "-")
        output = Path.cwd() / f"{video.stem}_{label}.png"

    if time == "-1":
        # -ss can't seek from the end; -sseof -1 starts one second before
        # it, and image2's -update rewrites the file per decoded frame, so
        # the last frame is what remains.
        stream = ffmpeg.input(str(video), sseof=-1).output(str(output), update=1)
    else:
        duration = float(ffmpeg.probe(str(video))["format"]["duration"])
        if _seconds(time) >= duration:
            raise click.ClickException(
                f"--time {time} is past the end of {video} ({duration:.2f}s)"
            )
        stream = ffmpeg.input(str(video), ss=time).output(str(output), vframes=1)

    stream.overwrite_output().run()
    click.echo(f"Extracted frame at {'end' if time == '-1' else time} to {output}")
