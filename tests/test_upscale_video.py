import shutil
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from click.testing import CliRunner

from vide.cli import cli
from vide.models import flashvsr

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def make_video(path, frames, size=(32, 24)):
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, size
    )
    for i in range(frames):
        out.write(np.full((size[1], size[0], 3), i * 20, dtype=np.uint8))
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


class FakeDenoisingModel:
    def __init__(self):
        self.LQ_proj_in = None


class FakeVAEModel:
    def __init__(self):
        self.encoder = "encoder"
        self.conv1 = "conv1"


class FakeVAE:
    def __init__(self):
        self.model = FakeVAEModel()


class FakePipe:
    def __init__(self, manager, device):
        self.manager = manager
        self.device = device
        self._denoising_model = FakeDenoisingModel()
        self.vae = FakeVAE()
        self.calls = {}

    def denoising_model(self):
        return self._denoising_model

    def to(self, device):
        self.calls["to"] = device

    def enable_vram_management(self, num_persistent_param_in_dit):
        self.calls["vram"] = num_persistent_param_in_dit

    def init_cross_kv(self):
        self.calls["cross_kv"] = True

    def load_models_to_device(self, names):
        self.calls["load_models_to_device"] = list(names)

    def __call__(self, **kwargs):
        self.calls["predict_kwargs"] = kwargs
        return torch.zeros(3, kwargs["num_frames"], kwargs["height"], kwargs["width"])


class FakeModelManager:
    def __init__(self, torch_dtype, device):
        self.torch_dtype = torch_dtype
        self.device = device
        self.loaded = []

    def load_models(self, paths):
        self.loaded.extend(paths)


@pytest.fixture
def fake_diffsynth(monkeypatch):
    pipes = []

    class FakeFlashVSRFullPipeline:
        @classmethod
        def from_model_manager(cls, manager, device):
            pipe = FakePipe(manager, device)
            pipes.append(pipe)
            return pipe

    fake_module = types.ModuleType("diffsynth")
    fake_module.ModelManager = FakeModelManager
    fake_module.FlashVSRFullPipeline = FakeFlashVSRFullPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_module)
    return pipes


@pytest.fixture
def fake_snapshot_download(monkeypatch, tmp_path):
    calls = []

    def _snapshot_download(repo_id, allow_patterns, local_dir=None):
        calls.append(
            {"repo_id": repo_id, "allow_patterns": allow_patterns, "local_dir": local_dir}
        )
        dest = Path(local_dir) if local_dir else tmp_path / "hf-cache"
        dest.mkdir(parents=True, exist_ok=True)
        return str(dest)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    return calls


def test_cli_registers_upscale_video():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "upscale-video" in result.output


def test_upscale_video_too_few_frames(tmp_path):
    video = tmp_path / "short.mp4"
    make_video(video, frames=2)

    result = CliRunner().invoke(cli, ["upscale-video", str(video)])

    assert result.exit_code != 0
    assert "too few frames" in result.output


@needs_ffmpeg
def test_upscale_video_happy_path(
    tmp_path, monkeypatch, fake_diffsynth, fake_snapshot_download
):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=6)
    output = tmp_path / "out.mp4"
    monkeypatch.setattr("vide.models.pick_device", lambda: "cpu")

    result = CliRunner().invoke(
        cli, ["upscale-video", str(video), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert len(fake_diffsynth) == 1
    pipe = fake_diffsynth[0]
    assert pipe.calls["to"] == "cpu"
    kwargs = pipe.calls["predict_kwargs"]
    assert kwargs["height"] == 128
    assert kwargs["width"] == 128
    assert frame_count(output) == kwargs["num_frames"]
    assert video_codec(output) == "h264,yuv420p"
    assert f"Wrote {kwargs['num_frames']} frame(s)" in result.output


@needs_ffmpeg
def test_upscale_video_default_output(
    tmp_path, monkeypatch, fake_diffsynth, fake_snapshot_download
):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=6)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vide.models.pick_device", lambda: "cpu")

    result = CliRunner().invoke(cli, ["upscale-video", str(video)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "clip_upscaled.mp4").exists()


@needs_ffmpeg
def test_upscale_video_custom_options(
    tmp_path, monkeypatch, fake_diffsynth, fake_snapshot_download
):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=6)
    output = tmp_path / "out.mp4"
    weights_dir = tmp_path / "weights"
    monkeypatch.setattr("vide.models.pick_device", lambda: "cpu")

    result = CliRunner().invoke(
        cli,
        [
            "upscale-video", str(video),
            "--output", str(output),
            "--model", "flashvsr",
            "--scale", "2.0",
            "--sparse-ratio", "1.5",
            "--local-range", "9",
            "--kv-ratio", "2.5",
            "--seed", "42",
            "--no-color-fix",
            "--weights-dir", str(weights_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_snapshot_download[0]["local_dir"] == str(weights_dir)
    kwargs = fake_diffsynth[0].calls["predict_kwargs"]
    assert kwargs["seed"] == 42
    assert kwargs["local_range"] == 9
    assert kwargs["kv_ratio"] == 2.5
    assert kwargs["color_fix"] is False
    expected_topk = 1.5 * flashvsr.REFERENCE_PIXELS / (kwargs["height"] * kwargs["width"])
    assert kwargs["topk_ratio"] == pytest.approx(expected_topk)


@needs_ffmpeg
def test_upscale_video_encoder_failure(
    tmp_path, monkeypatch, fake_diffsynth, fake_snapshot_download
):
    video = tmp_path / "clip.mp4"
    make_video(video, frames=6)
    output = tmp_path / "no-such-dir" / "out.mp4"
    monkeypatch.setattr("vide.models.pick_device", lambda: "cpu")

    result = CliRunner().invoke(
        cli, ["upscale-video", str(video), "--output", str(output)]
    )

    assert result.exit_code != 0
    assert "ffmpeg failed to encode" in result.output
