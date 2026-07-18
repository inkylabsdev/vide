import shutil
import subprocess

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

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


class FakeEstimator:
    """Returns a horizontal depth ramp at the input image's size."""

    def __call__(self, image):
        width, height = image.size
        ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        return {"depth": Image.fromarray(ramp)}


@pytest.fixture
def fake_pipeline(monkeypatch):
    # transformers lazily materializes itself on first attribute access,
    # replacing its sys.modules entry; resolve it first so the patch lands
    # on the module the command will actually import from.
    from transformers import pipeline as _  # noqa: F401

    calls = []

    def pipeline(task, model, device):
        calls.append({"task": task, "model": model, "device": device})
        return FakeEstimator()

    monkeypatch.setattr("transformers.pipeline", pipeline)
    return calls


def test_cli_registers_convert_depth_video():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "convert-depth-video" in result.output


@needs_ffmpeg
def test_convert_depth_video(tmp_path, fake_pipeline):
    video = tmp_path / "input.mp4"
    make_video(video, frames=5)
    output = tmp_path / "depth.mp4"

    result = CliRunner().invoke(
        cli, ["convert-depth-video", str(video), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert len(fake_pipeline) == 1
    assert fake_pipeline[0]["task"] == "depth-estimation"
    assert fake_pipeline[0]["model"] == "depth-anything/Depth-Anything-V2-Base-hf"
    assert frame_count(output) == 5
    # H.264 + yuv420p so browsers can play it, not just QuickTime.
    assert video_codec(output) == "h264,yuv420p"
    assert "Wrote 5 frame(s)" in result.output


@needs_ffmpeg
def test_convert_depth_video_default_output(tmp_path, fake_pipeline, monkeypatch):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=2)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["convert-depth-video", str(video)])

    assert result.exit_code == 0, result.output
    assert frame_count(tmp_path / "clip_depth.mp4") == 2


@needs_ffmpeg
def test_convert_depth_video_custom_model_and_colormap(tmp_path, fake_pipeline):
    video = tmp_path / "input.mp4"
    make_video(video, frames=1)
    output = tmp_path / "depth.mp4"

    result = CliRunner().invoke(
        cli,
        [
            "convert-depth-video", str(video),
            "--output", str(output),
            "--model", "depth-anything/Depth-Anything-V2-Small-hf",
            "--colormap", "magma",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_pipeline[0]["model"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert frame_count(output) == 1


@needs_ffmpeg
def test_convert_depth_video_encoder_failure(tmp_path, fake_pipeline):
    video = tmp_path / "input.mp4"
    make_video(video, frames=1)
    output = tmp_path / "no-such-dir" / "depth.mp4"

    result = CliRunner().invoke(
        cli, ["convert-depth-video", str(video), "--output", str(output)]
    )

    assert result.exit_code != 0
    assert "ffmpeg failed to encode" in result.output
