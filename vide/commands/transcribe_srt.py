"""Transcribe an audio or video file into an SRT subtitle file.

Speech is transcribed with WhisperX and aligned to word-level timestamps, then
shaped into readable, reading-speed-normalized cues (see vide.subtitles). Heavy
dependencies are imported lazily so that loading the vide CLI stays fast.
"""

from pathlib import Path

import click

from vide import models, subtitles
from vide.models import whisperx as whisperx_model


@click.command("transcribe-srt")
@click.argument(
    "media", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output subtitle path. Defaults to ./<media name>.srt.",
)
@click.option(
    "--model",
    default=whisperx_model.DEFAULT_MODEL,
    show_default=True,
    help="faster-whisper model size or Hugging Face model id.",
)
@click.option(
    "--language",
    default=None,
    help="ISO 639-1 code of the spoken language. Omit to auto-detect.",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Number of audio chunks transcribed in parallel.",
)
@click.option(
    "--max-line-length",
    default=subtitles.MAX_LINE_LENGTH,
    show_default=True,
    help="Maximum characters per subtitle line.",
)
@click.option(
    "--max-lines",
    default=subtitles.MAX_LINES,
    show_default=True,
    help="Maximum lines per subtitle cue.",
)
@click.option(
    "--max-cps",
    default=subtitles.MAX_CPS,
    show_default=True,
    help="Maximum reading speed in characters per second.",
)
@click.option(
    "--min-duration",
    default=subtitles.MIN_DURATION,
    show_default=True,
    help="Minimum seconds a cue stays on screen.",
)
@click.option(
    "--max-duration",
    default=subtitles.MAX_DURATION,
    show_default=True,
    help="Maximum seconds a cue stays on screen.",
)
def cli(
    media: Path,
    output: Path | None,
    model: str,
    language: str | None,
    batch_size: int,
    max_line_length: int,
    max_lines: int,
    max_cps: float,
    min_duration: float,
    max_duration: float,
):
    """Transcribe MEDIA (audio or video) into an SRT subtitle file."""
    if output is None:
        output = Path.cwd() / f"{media.stem}.srt"

    device = models.pick_device()
    click.echo(f"Loading whisper '{model}' on {device} ...")
    try:
        asr_model = whisperx_model.load(model, device)
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "whisperx is not installed. Install it with: pip install whisperx"
        ) from exc

    audio, result = whisperx_model.transcribe(
        asr_model, media, batch_size=batch_size, language=language
    )
    language = result["language"]
    if result["segments"]:
        result = whisperx_model.align(audio, result, device)
    else:
        click.echo("No speech detected; writing empty subtitles.")

    srt = subtitles.generate_srt(
        result["segments"],
        language,
        max_line_length=max_line_length,
        max_lines=max_lines,
        max_cps=max_cps,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    output.write_text(srt, encoding="utf-8")

    cues = srt.count(" --> ")
    click.echo(f"Wrote {cues} cue(s) ({language}) to {output}")
