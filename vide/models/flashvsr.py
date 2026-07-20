"""FlashVSR: real-time diffusion video super-resolution.

FlashVSR (https://github.com/OpenImagingLab/FlashVSR) ships as a fork of
the `diffsynth` package plus two small modules the upstream repo keeps in
its example scripts rather than in the installable package: an
LQ-projection module and the streaming TCDecoder. Both are vendored here
(`_build_lq_proj`, adapted from `examples/WanVSR/utils/utils.py`, and
`vide.models._flashvsr_tcdecoder`, adapted from
`examples/WanVSR/utils/TCDecoder.py`, both Apache-2.0) since they aren't
importable any other way; everything else (the diffusion transformer and
sparse attention) comes from `diffsynth` itself, which must be installed
per the FlashVSR README (clone the repo, `pip install -e .`, plus the
Block-Sparse-Attention CUDA extension) and is only ever imported lazily,
so loading the vide CLI stays fast and this module stays importable
without it.

We use FlashVSR's "tiny-long" streaming pipeline
(`FlashVSRTinyLongPipeline`): it denoises the clip in short windows and
decodes each window incrementally with the lightweight TCDecoder (moving
finished frames to CPU as it goes), so an arbitrarily long video runs in
bounded VRAM. The alternative "full" pipeline decodes the whole clip at
once through the Wan2.1 VAE, which does not fit a long clip on a consumer
GPU. On a 12 GB card the practical output ceiling is roughly 768x1280
(FlashVSR's training resolution); for a 4x upscale that means a source of
about 192x320, i.e. downscale first, then upscale.
"""

from pathlib import Path

DEFAULT_MODEL = "JunhaoZhuang/FlashVSR"
WEIGHT_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)

# The fixed empty-prompt text-encoder context FlashVSR conditions on. It ships
# in the GitHub repo's example assets rather than the Hugging Face weights, so
# it's fetched from there and cached alongside the downloaded weights.
PROMPT_CONTEXT_FILE = "posi_prompt.pth"
PROMPT_CONTEXT_URL = (
    "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/"
    "examples/WanVSR/prompt_tensor/posi_prompt.pth"
)

# FlashVSR is trained for 4x super-resolution and aligns output dimensions
# to a multiple of 128 (the DiT's patch/window size).
SCALE = 4.0
DIM_MULTIPLE = 128
# Below this the internal 8n+1 frame padding collapses to a single chunk and
# the LQ-projection module has nothing to emit (see `prepare_lq_video`).
MIN_FRAMES = 5
# `topk_ratio` is defined relative to FlashVSR's 768x1280 training resolution.
REFERENCE_PIXELS = 768 * 1280
DEFAULT_SPARSE_RATIO = 2.0
DEFAULT_KV_RATIO = 3.0
DEFAULT_LOCAL_RANGE = 11


def compute_scaled_dims(
    width: int, height: int, scale: float = SCALE, multiple: int = DIM_MULTIPLE
):
    """Scaled and 128-aligned target dimensions for an upscale.

    Returns `(scaled_w, scaled_h, target_w, target_h)`: the frame size after
    a plain `scale`x resize, and the size after center-cropping that down to
    the nearest multiple of `multiple` (at least one `multiple`).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    scaled_w = round(width * scale)
    scaled_h = round(height * scale)
    target_w = max(multiple, (scaled_w // multiple) * multiple)
    target_h = max(multiple, (scaled_h // multiple) * multiple)
    return scaled_w, scaled_h, target_w, target_h


def largest_valid_frame_count(n: int) -> int:
    """Largest `8k+1 <= n`, FlashVSR's required clip length; 0 if `n < 1`."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def prepare_lq_video(frames, scaled_w: int, scaled_h: int, target_w: int, target_h: int):
    """Build the low-quality conditioning tensor FlashVSR expects.

    `frames` are RGB uint8 arrays (HWC). Each is bicubic-upscaled to
    `(scaled_w, scaled_h)` and center-cropped to `(target_w, target_h)`; the
    last frame is repeated 4x and the clip truncated to the largest valid
    `8k+1` length, matching the upstream inference scripts. Returns a
    `(1, C, F, H, W)` float32 tensor on CPU, normalized to [-1, 1], and the
    resulting frame count `F`.
    """
    import cv2
    import torch

    padded = list(frames) + [frames[-1]] * 4
    num_frames = largest_valid_frame_count(len(padded))
    padded = padded[:num_frames]

    left = (scaled_w - target_w) // 2
    top = (scaled_h - target_h) // 2

    tensors = []
    for frame in padded:
        scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        cropped = scaled[top : top + target_h, left : left + target_w]
        tensor = torch.from_numpy(cropped).to(dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        tensors.append(tensor)

    video = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return video, num_frames


def tensor_to_frames(tensor):
    """Convert a FlashVSR output tensor `(C, T, H, W)` in [-1, 1] to BGR
    uint8 frames ready to pipe into ffmpeg."""
    import cv2
    import torch

    frames = tensor.permute(1, 2, 3, 0)
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    for frame in frames:
        yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _build_lq_proj(in_dim: int, out_dim: int, layer_num: int):
    """Build FlashVSR's LQ4x projection module: a small causal 3D-conv
    stack that bridges pixel-space input frames to the DiT's latent
    conditioning space.

    Adapted from `Buffer_LQ4x_Proj` in FlashVSR's
    `examples/WanVSR/utils/utils.py` (Apache-2.0) -- that module isn't part
    of the installable `diffsynth` package, so it's reproduced here rather
    than imported. `einops.rearrange` calls are replaced with equivalent
    `view`/`permute` to avoid a new dependency for one small module.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    cache_span = 2  # trailing frames carried into the next causal-conv chunk

    class _RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.scale = dim**0.5
            self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

        def forward(self, x):
            return F.normalize(x, dim=1) * self.scale * self.gamma

    class _CausalConv3d(nn.Conv3d):
        """3D conv padded only on the leading edge of the time axis, so it
        never looks into the future of the clip."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._causal_padding = (
                self.padding[2], self.padding[2],
                self.padding[1], self.padding[1],
                2 * self.padding[0], 0,
            )
            self.padding = (0, 0, 0)

        def forward(self, x, cache_x=None):
            padding = list(self._causal_padding)
            if cache_x is not None and padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
            x = F.pad(x, padding, mode="replicate")
            return super().forward(x)

    class _PixelShuffle3d(nn.Module):
        """Space-to-depth over the (F, H, W) axes: trades spatial/temporal
        resolution for channels, in fixed (1, 16, 16) blocks."""

        def __init__(self, ff, hh, ww):
            super().__init__()
            self.ff, self.hh, self.ww = ff, hh, ww

        def forward(self, x):
            b, c, f, h, w = x.shape
            ff, hh, ww = self.ff, self.hh, self.ww
            x = x.view(b, c, f // ff, ff, h // hh, hh, w // ww, ww)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
            return x.view(b, c * ff * hh * ww, f // ff, h // hh, w // ww)

    class _LQProj(nn.Module):
        def __init__(self, in_dim, out_dim, layer_num):
            super().__init__()
            self.layer_num = layer_num
            self.pixel_shuffle = _PixelShuffle3d(1, 16, 16)
            self.conv1 = _CausalConv3d(
                in_dim * 16 * 16, 2048, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)
            )
            self.norm1 = _RMSNorm(2048)
            self.act1 = nn.SiLU()
            self.conv2 = _CausalConv3d(
                2048, 3072, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)
            )
            self.norm2 = _RMSNorm(3072)
            self.act2 = nn.SiLU()
            self.linear_layers = nn.ModuleList(
                nn.Linear(3072, out_dim) for _ in range(layer_num)
            )
            self.clear_cache()

        def clear_cache(self):
            """Reset the causal-conv cache and window counter between videos."""
            self.cache = {"conv1": None, "conv2": None}
            self.clip_idx = 0

        @staticmethod
        def _tokens(x):
            # (B, C, F, H, W) -> (B, F*H*W, C)
            return x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1])

        def _project(self, x):
            tokens = self._tokens(x)
            return [layer(tokens) for layer in self.linear_layers]

        def forward(self, video):
            """Project a whole clip at once (non-streaming)."""
            self.clear_cache()
            t = video.shape[2]
            iterations = 1 + (t - 1) // 4
            first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video = torch.cat([first_frame, video], dim=2)

            chunks = []
            for i in range(iterations):
                x = self.pixel_shuffle(video[:, :, i * 4 : (i + 1) * 4, :, :])
                self.cache["conv1"] = x[:, :, -cache_span:, :, :].clone()
                x = self.act1(self.norm1(self.conv1(x, self.cache["conv1"])))
                self.cache["conv2"] = x[:, :, -cache_span:, :, :].clone()
                if i == 0:
                    # The first chunk only seeds the causal-conv cache; it
                    # doesn't produce an output token of its own.
                    continue
                x = self.act2(self.norm2(self.conv2(x, self.cache["conv2"])))
                chunks.append(x)

            return self._project(torch.cat(chunks, dim=2))

        def stream_forward(self, video_clip):
            """Project one streaming window, carrying the causal-conv cache
            across calls. The first window only seeds that cache and returns
            ``None``; later windows return a per-layer list of conditioning
            tokens. This is what FlashVSR's streaming pipelines actually
            call (`forward` is the equivalent whole-clip path)."""
            if self.clip_idx == 0:
                first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
                video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            self.cache["conv1"] = x[:, :, -cache_span:, :, :].clone()
            x = self.act1(self.norm1(self.conv1(x, self.cache["conv1"])))
            self.cache["conv2"] = x[:, :, -cache_span:, :, :].clone()
            if self.clip_idx == 0:
                self.clip_idx += 1
                return None
            x = self.act2(self.norm2(self.conv2(x, self.cache["conv2"])))
            self.clip_idx += 1
            return self._project(x)

    return _LQProj(in_dim, out_dim, layer_num)


def download_weights(repo_id: str, weights_dir: Path | None = None) -> Path:
    """Download (or reuse an existing local copy of) `WEIGHT_FILES` for
    `repo_id` from the Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(WEIGHT_FILES),
            local_dir=str(weights_dir) if weights_dir else None,
        )
    )


# TCDecoder is built wider than its default and takes the concatenated
# latent + space-to-depth LQ conditioning as input channels (16 + 768).
TCDECODER_CHANNELS = (512, 256, 128, 128)
TCDECODER_LATENT_CHANNELS = 16 + 768
# Number of DiT parameters kept resident on the GPU; 0 streams them from CPU
# per forward pass, trading speed for the VRAM headroom a 12 GB card needs.
NUM_PERSISTENT_PARAMS = 0


def _load_prompt_context(weights_dir: Path, device: str, dtype):
    """Return FlashVSR's fixed prompt-context tensor, downloading and caching
    it under `weights_dir` on first use."""
    import torch

    path = weights_dir / PROMPT_CONTEXT_FILE
    if not path.exists():
        from urllib.request import urlopen

        with urlopen(PROMPT_CONTEXT_URL) as response:
            path.write_bytes(response.read())
    return torch.load(path, map_location="cpu").to(device=device, dtype=dtype)


def load(repo_id: str, weights_dir: Path | None, device: str):
    """Load the FlashVSR tiny-long streaming pipeline for `repo_id` onto
    `device`.

    Mirrors `init_pipeline()` in FlashVSR's
    `infer_flashvsr_tiny_long_video.py`: load the DiT weights through
    `diffsynth.ModelManager`, attach the vendored LQ-projection module and
    TCDecoder, then move everything to `device`, offload DiT params, and
    prime cross-attention KV caches. The Wan VAE is not loaded: this
    pipeline decodes with the TCDecoder instead.
    """
    import torch
    from diffsynth import FlashVSRTinyLongPipeline, ModelManager

    from vide.models._flashvsr_tcdecoder import build_tcdecoder

    dtype = torch.bfloat16
    weights = download_weights(repo_id, weights_dir)

    manager = ModelManager(torch_dtype=dtype, device="cpu")
    manager.load_models(
        [str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")]
    )
    pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device=device)

    lq_proj = _build_lq_proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    lq_ckpt = weights / "LQ_proj_in.ckpt"
    if lq_ckpt.exists():
        lq_proj.load_state_dict(torch.load(lq_ckpt, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in = lq_proj

    tcdecoder = build_tcdecoder(
        new_channels=TCDECODER_CHANNELS,
        device=device,
        dtype=dtype,
        new_latent_channels=TCDECODER_LATENT_CHANNELS,
    )
    tc_ckpt = weights / "TCDecoder.ckpt"
    if tc_ckpt.exists():
        tcdecoder.load_state_dict(torch.load(tc_ckpt, map_location="cpu"), strict=False)
    pipe.TCDecoder = tcdecoder

    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=NUM_PERSISTENT_PARAMS)
    pipe.init_cross_kv(context_tensor=_load_prompt_context(weights, device, dtype))
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def predict(
    pipe,
    lq_video,
    *,
    num_frames: int,
    height: int,
    width: int,
    seed: int = 0,
    sparse_ratio: float = DEFAULT_SPARSE_RATIO,
    kv_ratio: float = DEFAULT_KV_RATIO,
    local_range: int = DEFAULT_LOCAL_RANGE,
    color_fix: bool = True,
):
    """Run one FlashVSR pass over `lq_video`; returns a `(C, F, H, W)` tensor
    in [-1, 1]."""
    topk_ratio = sparse_ratio * REFERENCE_PIXELS / (height * width)
    return pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=seed,
        tiled=False,
        LQ_video=lq_video,
        num_frames=num_frames,
        height=height,
        width=width,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=topk_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
        color_fix=color_fix,
    )
