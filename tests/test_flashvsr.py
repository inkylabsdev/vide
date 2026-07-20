import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from vide.models import flashvsr


class FakeDenoisingModel:
    def __init__(self):
        self.LQ_proj_in = None


class FakeTCDecoder:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state = (state_dict, strict)


class FakePipe:
    def __init__(self, manager, device):
        self.manager = manager
        self.device = device
        self._denoising_model = FakeDenoisingModel()
        self.TCDecoder = None
        self.calls = {}

    def denoising_model(self):
        return self._denoising_model

    def to(self, device):
        self.calls["to"] = device

    def enable_vram_management(self, num_persistent_param_in_dit):
        self.calls["vram"] = num_persistent_param_in_dit

    def init_cross_kv(self, context_tensor=None):
        self.calls["cross_kv"] = context_tensor

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
    """Stub out the genuinely GPU/network-bound externals: the diffsynth
    pipeline, the TCDecoder builder, and the prompt-context download. The
    vendored LQ-projection module still runs for real."""
    pipes = []

    class FakeFlashVSRTinyLongPipeline:
        @classmethod
        def from_model_manager(cls, manager, device):
            pipe = FakePipe(manager, device)
            pipes.append(pipe)
            return pipe

    fake_module = types.ModuleType("diffsynth")
    fake_module.ModelManager = FakeModelManager
    fake_module.FlashVSRTinyLongPipeline = FakeFlashVSRTinyLongPipeline
    monkeypatch.setitem(sys.modules, "diffsynth", fake_module)

    monkeypatch.setattr(
        "vide.models._flashvsr_tcdecoder.build_tcdecoder",
        lambda **kwargs: FakeTCDecoder(),
    )
    monkeypatch.setattr(
        flashvsr, "_load_prompt_context",
        lambda weights_dir, device, dtype: torch.zeros(1, 1, 1),
    )
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


# -- pure helpers -----------------------------------------------------------


def test_compute_scaled_dims():
    assert flashvsr.compute_scaled_dims(100, 60, 4.0, 128) == (400, 240, 384, 128)


def test_compute_scaled_dims_floors_to_at_least_one_multiple():
    assert flashvsr.compute_scaled_dims(32, 24, 4.0, 128) == (128, 96, 128, 128)


def test_compute_scaled_dims_rejects_bad_size():
    with pytest.raises(ValueError):
        flashvsr.compute_scaled_dims(0, 10)
    with pytest.raises(ValueError):
        flashvsr.compute_scaled_dims(10, 0)


def test_compute_scaled_dims_rejects_bad_scale():
    with pytest.raises(ValueError):
        flashvsr.compute_scaled_dims(10, 10, scale=0)


def test_largest_valid_frame_count():
    assert flashvsr.largest_valid_frame_count(0) == 0
    assert flashvsr.largest_valid_frame_count(1) == 1
    assert flashvsr.largest_valid_frame_count(8) == 1
    assert flashvsr.largest_valid_frame_count(9) == 9
    assert flashvsr.largest_valid_frame_count(16) == 9
    assert flashvsr.largest_valid_frame_count(17) == 17


def test_prepare_lq_video_shape_and_range():
    frames = [
        np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8) for _ in range(6)
    ]
    sw, sh, tw, th = flashvsr.compute_scaled_dims(100, 60, 4.0, 128)

    video, num_frames = flashvsr.prepare_lq_video(frames, sw, sh, tw, th)

    assert num_frames == flashvsr.largest_valid_frame_count(len(frames) + 4)
    assert video.shape == (1, 3, num_frames, th, tw)
    assert video.min() >= -1.0 - 1e-4
    assert video.max() <= 1.0 + 1e-4


def test_tensor_to_frames():
    tensor = torch.rand(3, 4, 16, 16) * 2 - 1

    frames = list(flashvsr.tensor_to_frames(tensor))

    assert len(frames) == 4
    assert all(frame.shape == (16, 16, 3) for frame in frames)
    assert all(frame.dtype == np.uint8 for frame in frames)


def test_build_lq_proj_output_shapes():
    proj = flashvsr._build_lq_proj(in_dim=3, out_dim=8, layer_num=2)

    outputs = proj(torch.randn(1, 3, 9, 16, 16))

    assert len(outputs) == 2
    for out in outputs:
        assert out.shape == (1, 2, 8)


def test_build_lq_proj_stream_forward():
    proj = flashvsr._build_lq_proj(in_dim=3, out_dim=8, layer_num=2)
    proj.clear_cache()

    # The first streaming window only seeds the causal-conv cache.
    assert proj.stream_forward(torch.randn(1, 3, 4, 16, 16)) is None
    assert proj.clip_idx == 1

    # Later windows emit one conditioning token list per linear layer.
    outputs = proj.stream_forward(torch.randn(1, 3, 4, 16, 16))
    assert len(outputs) == 2
    for out in outputs:
        assert out.shape[0] == 1 and out.shape[2] == 8

    # clear_cache resets the streaming state so the next call seeds again.
    proj.clear_cache()
    assert proj.clip_idx == 0
    assert proj.cache == {"conv1": None, "conv2": None}
    assert proj.stream_forward(torch.randn(1, 3, 4, 16, 16)) is None


def test_load_prompt_context_downloads_then_caches(tmp_path, monkeypatch):
    import io

    buffer = io.BytesIO()
    torch.save(torch.zeros(1, 2, 3), buffer)
    payload = buffer.getvalue()
    downloads = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload

    def fake_urlopen(url):
        downloads.append(url)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    ctx = flashvsr._load_prompt_context(tmp_path, "cpu", torch.float32)
    assert ctx.shape == (1, 2, 3)
    assert (tmp_path / flashvsr.PROMPT_CONTEXT_FILE).exists()
    assert downloads == [flashvsr.PROMPT_CONTEXT_URL]

    # Second call reuses the cached file and does not download again.
    ctx2 = flashvsr._load_prompt_context(tmp_path, "cpu", torch.float32)
    assert ctx2.shape == (1, 2, 3)
    assert downloads == [flashvsr.PROMPT_CONTEXT_URL]


# -- weight loading -----------------------------------------------------------


def test_download_weights_default_cache(fake_snapshot_download, tmp_path):
    result = flashvsr.download_weights("org/repo")

    assert fake_snapshot_download[0]["repo_id"] == "org/repo"
    assert fake_snapshot_download[0]["allow_patterns"] == list(flashvsr.WEIGHT_FILES)
    assert fake_snapshot_download[0]["local_dir"] is None
    assert result.is_dir()


def test_download_weights_explicit_dir(fake_snapshot_download, tmp_path):
    weights_dir = tmp_path / "custom-weights"

    result = flashvsr.download_weights("org/repo", weights_dir)

    assert fake_snapshot_download[0]["local_dir"] == str(weights_dir)
    assert result == weights_dir


def test_load_without_checkpoint(fake_diffsynth, fake_snapshot_download):
    pipe = flashvsr.load("org/repo", None, "cpu")

    assert pipe is fake_diffsynth[0]
    assert pipe.calls["to"] == "cpu"
    assert pipe.calls["vram"] == flashvsr.NUM_PERSISTENT_PARAMS
    # init_cross_kv is primed with the (stubbed) prompt-context tensor.
    assert pipe.calls["cross_kv"] is not None
    assert pipe.calls["load_models_to_device"] == ["dit", "vae"]
    assert pipe.TCDecoder is not None
    assert pipe.denoising_model().LQ_proj_in is not None
    # No checkpoints on disk, so the TCDecoder is left at its built weights.
    assert pipe.TCDecoder.loaded_state is None


def test_load_with_checkpoint(fake_diffsynth, monkeypatch, tmp_path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    reference = flashvsr._build_lq_proj(in_dim=3, out_dim=1536, layer_num=1)
    torch.save(reference.state_dict(), weights_dir / "LQ_proj_in.ckpt")
    torch.save({}, weights_dir / "TCDecoder.ckpt")

    calls = []

    def _snapshot_download(repo_id, allow_patterns, local_dir=None):
        calls.append(local_dir)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)

    pipe = flashvsr.load("org/repo", weights_dir, "cpu")

    assert calls == [str(weights_dir)]
    assert pipe.denoising_model().LQ_proj_in is not None
    # The TCDecoder checkpoint is loaded non-strictly.
    assert pipe.TCDecoder.loaded_state == ({}, False)


def test_predict_forwards_kwargs_and_computes_topk_ratio():
    calls = {}

    class FakePredictPipe:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return torch.zeros(3, kwargs["num_frames"], kwargs["height"], kwargs["width"])

    lq_video = torch.zeros(1, 3, 9, 128, 128)

    result = flashvsr.predict(
        FakePredictPipe(), lq_video, num_frames=9, height=128, width=128, seed=3
    )

    assert calls["seed"] == 3
    assert calls["LQ_video"] is lq_video
    assert calls["num_frames"] == 9
    assert calls["color_fix"] is True
    expected_topk = (
        flashvsr.DEFAULT_SPARSE_RATIO * flashvsr.REFERENCE_PIXELS / (128 * 128)
    )
    assert calls["topk_ratio"] == pytest.approx(expected_topk)
    assert result.shape == (3, 9, 128, 128)
