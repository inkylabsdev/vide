import shutil
import subprocess

import pytest
from click.testing import CliRunner

from vide.cli import cli

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def make_video(path, *, acodec="aac", duration=2):
    """A tiny test video; acodec=None makes it video-only."""
    audio = [] if acodec is None else [
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:a", acodec,
    ]
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=24",
            *audio,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )


def audio_codecs(path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stdout.split()


def test_cli_registers_audio_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "extract-audio" in result.output
    assert "remove-audio" in result.output


@needs_ffmpeg
def test_extract_audio(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)
    output = tmp_path / "sound.m4a"

    result = CliRunner().invoke(cli, ["extract-audio", str(video), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert audio_codecs(output) == ["aac"]


@needs_ffmpeg
def test_extract_audio_default_output_matches_codec(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    make_video(video)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["extract-audio", str(video)])

    assert result.exit_code == 0, result.output
    assert audio_codecs(tmp_path / "clip.m4a") == ["aac"]


@needs_ffmpeg
def test_extract_audio_unknown_codec_falls_back_to_mka(tmp_path, monkeypatch):
    video = tmp_path / "clip.mov"
    make_video(video, acodec="pcm_s16le")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["extract-audio", str(video)])

    assert result.exit_code == 0, result.output
    assert audio_codecs(tmp_path / "clip.mka") == ["pcm_s16le"]


@needs_ffmpeg
def test_extract_audio_reencodes_when_container_cannot_copy(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)  # aac audio; .mp3 can't hold it via stream copy
    output = tmp_path / "sound.mp3"

    result = CliRunner().invoke(cli, ["extract-audio", str(video), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert "Re-encoded" in result.output
    assert audio_codecs(output) == ["mp3"]


@needs_ffmpeg
def test_extract_audio_no_audio_track(tmp_path):
    video = tmp_path / "silent.mp4"
    make_video(video, acodec=None)

    result = CliRunner().invoke(cli, ["extract-audio", str(video)])

    assert result.exit_code != 0
    assert "no audio track" in result.output


@needs_ffmpeg
def test_remove_audio(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)
    output = tmp_path / "quiet.mp4"

    result = CliRunner().invoke(cli, ["remove-audio", str(video), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert audio_codecs(output) == []


@needs_ffmpeg
def test_remove_audio_default_output(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    make_video(video)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["remove-audio", str(video)])

    assert result.exit_code == 0, result.output
    assert audio_codecs(tmp_path / "clip_noaudio.mp4") == []
