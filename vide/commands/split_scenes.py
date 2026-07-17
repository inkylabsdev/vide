"""Split a video into clips at detected scene cuts (PySceneDetect wrapper)."""

from pathlib import Path

import click
from scenedetect import ContentDetector, detect, split_video_ffmpeg


@click.command("split-scenes")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to write clips to. Defaults to ./<video name>_scenes.",
)
@click.option(
    "--threshold",
    default=27.0,
    show_default=True,
    help="Scene cut sensitivity; lower detects more cuts.",
)
def cli(video: Path, output: Path | None, threshold: float):
    """Detect scenes in VIDEO and split each one into a clip."""
    if output is None:
        output = Path.cwd() / f"{video.stem}_scenes"

    click.echo(f"Detecting scenes in {video} ...")
    scenes = detect(str(video), ContentDetector(threshold=threshold), show_progress=True)
    if not scenes:
        click.echo("No scene cuts detected; nothing to split.")
        return

    output.mkdir(parents=True, exist_ok=True)
    click.echo(f"Detected {len(scenes)} scene(s); splitting with ffmpeg ...")
    split_video_ffmpeg(str(video), scenes, output_dir=output, show_progress=True)
    click.echo(f"Wrote {len(scenes)} clip(s) to {output}")
