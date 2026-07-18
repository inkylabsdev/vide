import shutil
import subprocess

import pytest
from click.testing import CliRunner

from vide.cli import cli

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def make_video(path, *, source="testsrc", duration=3):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"{source}=duration={duration}:size=320x240:rate=24",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )


def make_two_scene_video(path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=24",
            "-f", "lavfi", "-i", "smptebars=duration=3:size=320x240:rate=24",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )


def duration_of(path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def test_cli_registers_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "split-scenes" in result.output
    assert "merge-scenes" in result.output


@needs_ffmpeg
def test_split_scenes(tmp_path):
    video = tmp_path / "twoscene.mp4"
    make_two_scene_video(video)
    output = tmp_path / "clips"

    result = CliRunner().invoke(cli, ["split-scenes", str(video), "--output", str(output)])

    assert result.exit_code == 0, result.output
    clips = sorted(output.glob("*.mp4"))
    assert [c.name for c in clips] == [
        "twoscene-Scene-001.mp4",
        "twoscene-Scene-002.mp4",
    ]


@needs_ffmpeg
def test_split_scenes_default_output_dir(tmp_path, monkeypatch):
    video = tmp_path / "twoscene.mp4"
    make_two_scene_video(video)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["split-scenes", str(video)])

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "twoscene_scenes").glob("*.mp4"))) == 2


@needs_ffmpeg
def test_split_scenes_no_cuts(tmp_path):
    video = tmp_path / "static.mp4"
    make_video(video, source="smptebars", duration=2)
    output = tmp_path / "clips"

    result = CliRunner().invoke(cli, ["split-scenes", str(video), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "No scene cuts detected" in result.output
    assert not output.exists()


@needs_ffmpeg
def test_merge_scenes(tmp_path):
    a = tmp_path / "a'quoted.mp4"
    b = tmp_path / "b.mp4"
    make_video(a, source="testsrc")
    make_video(b, source="smptebars")
    output = tmp_path / "merged.mp4"

    result = CliRunner().invoke(
        cli, ["merge-scenes", str(a), str(b), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert duration_of(output) == pytest.approx(6.0, abs=0.2)
