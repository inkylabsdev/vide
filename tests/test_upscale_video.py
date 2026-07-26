import io
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from vide.cli import cli

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def make_video(path, frames, size=(64, 48)):
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, size
    )
    for i in range(frames):
        out.write(np.full((size[1], size[0], 3), i * 40, dtype=np.uint8))
    out.release()


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def frame_size(path):
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def video_codec(path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip()


@pytest.fixture
def fake_replicate(monkeypatch):
    """Double the resolution of each frame to simulate upscaling."""
    import replicate  # noqa: F401 — ensure module is loaded before patching

    calls = []

    def fake_run(model_ref, *, input, **kwargs):
        calls.append({"model": model_ref, "input": input})
        img_bytes = input["image"].read()
        img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        doubled = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        _, out_bytes = cv2.imencode(".png", doubled)
        return io.BytesIO(bytes(out_bytes))

    monkeypatch.setattr("replicate.run", fake_run)
    return calls


def test_cli_registers_upscale_video():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "upscale-video" in result.output


@needs_ffmpeg
def test_upscale_video_real_esrgan(tmp_path, fake_replicate):
    video = tmp_path / "input.mp4"
    make_video(video, frames=3)
    output = tmp_path / "upscaled.mp4"

    result = CliRunner().invoke(
        cli, ["upscale-video", str(video), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert len(fake_replicate) == 3 + 1  # +1 for the probe frame
    assert fake_replicate[0]["model"] == "nightmareai/real-esrgan"
    assert frame_count(output) == 3
    # Fake upscaler doubles each dimension: 64×48 → 128×96
    assert frame_size(output) == (128, 96)
    assert video_codec(output) == "h264,yuv420p"
    assert "Upscaled 3 frame(s)" in result.output


@needs_ffmpeg
def test_upscale_video_anime4k(tmp_path, fake_replicate):
    video = tmp_path / "input.mp4"
    make_video(video, frames=2)
    output = tmp_path / "upscaled.mp4"

    result = CliRunner().invoke(
        cli,
        ["upscale-video", str(video), "--output", str(output), "--model", "anime4k"],
    )

    assert result.exit_code == 0, result.output
    assert fake_replicate[0]["model"] == "phamquiluan/anime4k"
    assert frame_count(output) == 2
    assert "Upscaled 2 frame(s)" in result.output


@needs_ffmpeg
def test_upscale_video_default_output(tmp_path, fake_replicate, monkeypatch):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=1)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["upscale-video", str(video)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "clip_upscaled.mp4").exists()


@needs_ffmpeg
def test_upscale_video_encoder_failure(tmp_path, fake_replicate):
    video = tmp_path / "input.mp4"
    make_video(video, frames=1)
    output = tmp_path / "no-such-dir" / "upscaled.mp4"

    result = CliRunner().invoke(
        cli, ["upscale-video", str(video), "--output", str(output)]
    )

    assert result.exit_code != 0
    assert "ffmpeg failed to encode" in result.output


def test_upscale_video_no_readable_frames(tmp_path, fake_replicate):
    # A file with invalid video bytes causes cap.read() to return (False, None).
    bad_video = tmp_path / "corrupt.mp4"
    bad_video.write_bytes(b"\x00" * 100)
    output = tmp_path / "out.mp4"

    result = CliRunner().invoke(
        cli, ["upscale-video", str(bad_video), "--output", str(output)]
    )

    assert result.exit_code != 0
    assert "no readable frames" in result.output
