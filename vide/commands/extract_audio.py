"""Extract a video's audio track into its own file without re-encoding."""

from pathlib import Path

import click
import ffmpeg

# Default output extension per codec, so the stream copy lands in a
# container that can hold it. Anything unlisted falls back to Matroska
# audio (.mka), which holds everything.
CODEC_EXTENSIONS = {
    "aac": ".m4a",
    "mp3": ".mp3",
    "opus": ".opus",
    "vorbis": ".ogg",
    "flac": ".flac",
}


@click.command("extract-audio")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output audio path. Defaults to ./<video name>.<ext matching the codec>.",
)
def cli(video: Path, output: Path | None):
    """Extract VIDEO's audio track into its own file, without re-encoding."""
    streams = ffmpeg.probe(str(video))["streams"]
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    if audio is None:
        raise click.ClickException(f"{video} has no audio track")

    codec = audio["codec_name"]
    if output is None:
        output = Path.cwd() / f"{video.stem}{CODEC_EXTENSIONS.get(codec, '.mka')}"

    # Stream-copy only when the target container can hold the source codec
    # (e.g. AAC won't fit in .mp3); otherwise let ffmpeg re-encode to the
    # container's default codec.
    copy = output.suffix in (CODEC_EXTENSIONS.get(codec), ".mka")
    (
        ffmpeg.input(str(video))
        .audio.output(str(output), **({"acodec": "copy"} if copy else {}))
        .overwrite_output()
        .run()
    )
    verb = "Extracted" if copy else "Re-encoded"
    click.echo(f"{verb} {codec} audio to {output}")
