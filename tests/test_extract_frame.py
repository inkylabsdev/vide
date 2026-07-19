import shutil
import subprocess

import pytest
from click.testing import CliRunner

from vide.cli import cli

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def make_video(path, *, duration=2):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )


def image_codec(path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip()


def test_cli_registers_extract_frame():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "extract-frame" in result.output


@needs_ffmpeg
def test_extract_frame_default_first_frame(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    make_video(video)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["extract-frame", str(video)])

    assert result.exit_code == 0, result.output
    assert image_codec(tmp_path / "clip_00-00.png") == "png"


@needs_ffmpeg
def test_extract_frame_at_time(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)
    first = tmp_path / "first.png"
    later = tmp_path / "later.png"

    for time, output in [("00:00", first), ("00:01.11", later)]:
        result = CliRunner().invoke(
            cli, ["extract-frame", str(video), "--time", time, "-o", str(output)]
        )
        assert result.exit_code == 0, result.output

    assert image_codec(later) == "png"
    # testsrc animates, so different timestamps yield different frames.
    assert first.read_bytes() != later.read_bytes()


@needs_ffmpeg
def test_extract_frame_last(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    make_video(video)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["extract-frame", str(video), "--time", "-1"])

    assert result.exit_code == 0, result.output
    assert image_codec(tmp_path / "clip_last.png") == "png"


@needs_ffmpeg
def test_extract_frame_past_end(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video, duration=2)

    result = CliRunner().invoke(cli, ["extract-frame", str(video), "--time", "00:10"])

    assert result.exit_code != 0
    assert "past the end" in result.output


def test_extract_frame_invalid_time(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()  # click only checks existence; time is rejected before ffmpeg runs

    result = CliRunner().invoke(cli, ["extract-frame", str(video), "--time", "abc"])

    assert result.exit_code != 0
    assert "-1 for the last frame" in result.output
