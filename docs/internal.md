# Internal design notes, per command

One entry per command: the non-obvious decisions only. If a command is a
plain wrapper with no surprises, one line is plenty. What a command *does*
belongs in its `docs/commands/*.rst` page, not here.

## split-scenes

Plain PySceneDetect wrapper — `detect()` with `ContentDetector`, then
`split_video_ffmpeg()`. Our only contributions are the default threshold
(27.0) and the default output dir (`./<name>_scenes`). No cuts detected
means exit early, don't create an empty dir.

## merge-scenes

FFmpeg concat demuxer with `-c copy`: instant and lossless, which is why
inputs must share codec/resolution/framerate (always true for clips
`split-scenes` cut from one source). The only real logic is escaping `'` in
paths for the demuxer's file-list format (`'` → `'\''`). The list is a
`NamedTemporaryFile(delete=False)` removed in a `finally` — `delete=True`
would race with ffmpeg opening it on some platforms.

## extract-audio

Probes the codec and maps it to a container that holds it
(`CODEC_EXTENSIONS`) for the default output name, falling back to `.mka`
— Matroska holds anything. Stream-copies when the target container can
hold the source codec, re-encodes (ffmpeg's default codec for the
container) when it can't — copying AAC into `.mp3` is impossible, and
failing on it just pushes the codec/container matrix onto the user.
A video with no audio stream is an error, not a silent no-op.

## remove-audio

`-c copy -an`: every non-audio stream (video, subtitles) is copied
untouched. Default output keeps the input's extension, `_noaudio` suffix.

## convert-depth-video

Per-frame depth estimation via a `transformers` pipeline (Depth Anything V2
`-hf` checkpoint by default — the non-`-hf` repos lack `config.json` and
don't load). Decisions:

- Model-specific code (checkpoint name, pipeline load, inference) lives in
  `vide/models/depth_anything_v2.py`; the command keeps only video I/O and
  colormapping.
- `torch`/`transformers`/`cv2` import inside functions; see the
  lazy-import convention in `ARCHITECTURE.md`.
- Linux CI installs CPU torch wheels (`tool.uv.sources` in `pyproject.toml`)
  — the default CUDA wheels are ~2.5 GB and CI has no GPU.
- Output is encoded by piping raw BGR frames into an ffmpeg subprocess as
  H.264/yuv420p with `+faststart` — `cv2.VideoWriter`'s mp4v codec isn't
  web-playable.
- Device pick follows the CUDA → MPS → CPU convention in `ARCHITECTURE.md`.
- Depth is min-max normalized per frame before colormapping, so absolute
  depth scale is not preserved across frames.

## upscale-video

4x super-resolution via FlashVSR
(https://github.com/OpenImagingLab/FlashVSR), a one-step diffusion VSR
model. Decisions:

- Model-specific code (weight download, pipeline load, inference, and the
  FlashVSR-specific frame prep/postprocess) lives in
  `vide/models/flashvsr.py`, same split as `convert-depth-video` /
  `depth_anything_v2.py`; the command keeps only video I/O, ffmpeg
  encoding, and option wiring. FlashVSR has no `transformers.pipeline`-
  style entry point to wrap: it ships as a fork of the `diffsynth` package
  plus two small modules the upstream repo keeps in its example scripts
  (`utils/`) rather than the installable package — the LQ-projection module
  and the streaming TCDecoder. Both are vendored (`_build_lq_proj`, adapted
  from `Buffer_LQ4x_Proj`; `vide/models/_flashvsr_tcdecoder.py`, adapted
  from `TCDecoder.py`; both Apache-2.0) since they aren't importable any
  other way; everything else (DiT, sparse attention) comes from `diffsynth`
  itself, imported lazily and never vendored.
- We use the **tiny-long streaming** pipeline (`FlashVSRTinyLongPipeline`),
  not the whole-clip "full" pipeline. Full decodes the entire clip at once
  through the Wan2.1 VAE, which does not fit a real-length clip on a
  consumer GPU; tiny-long denoises in short windows and decodes each window
  incrementally with the lightweight TCDecoder, moving finished frames to
  CPU, so clip length is bounded by disk, not VRAM. This mirrors how Wan2GP
  runs FlashVSR on consumer cards. The Wan VAE is therefore not loaded at
  all.
- Output resolution *is* VRAM-bound (attention activations scale with frame
  area). On 12 GB the ceiling is ~768x1280; `load()` offloads DiT params
  (`num_persistent_param_in_dit=0`) and the command sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to claw back headroom.
  The intended workflow is downscale-then-upscale (see the command docs).
- The vendored LQ-projection implements `stream_forward` / `clear_cache`,
  not just a whole-clip `forward`: the streaming pipelines drive it one
  window at a time, threading a causal-conv cache across calls. The
  whole-clip `forward` is kept as the equivalent non-streaming path.
- FlashVSR conditions on a fixed empty-prompt text-encoder context that
  ships only in the GitHub repo's example assets (`posi_prompt.pth`), not
  the Hugging Face weights. `_load_prompt_context()` fetches it from GitHub
  and caches it beside the weights, then passes it to `init_cross_kv` so we
  don't depend on the upstream script's hardcoded relative path.
- The vendored TCDecoder keeps only the paths the tiny-long pipeline
  exercises: the sequential streaming decode. The parallel (non-streaming)
  decode, the in-`__init__` checkpoint load and its `TGrow` weight patching,
  the encoder-side `TPool`, and the diffusers wrapper are all dropped
  (the checkpoint is loaded externally with `strict=False`, as upstream's
  tiny-long script also does). `einops.rearrange` is rewritten with
  `view`/`permute`, as in `_build_lq_proj`, so neither vendored module adds
  a dependency.
- `diffsynth` is not a `vide` dependency and is not on PyPI under that
  name with FlashVSR's pipelines included — it must be installed from the
  FlashVSR repo per its README (also needs the Block-Sparse-Attention CUDA
  extension). `flashvsr.load()`/`flashvsr.predict()` assume it's
  importable; there is no fallback.
- No `--device` flag, per `ARCHITECTURE.md`'s CUDA → MPS → CPU convention
  — `models.pick_device()` picks automatically. In practice FlashVSR's
  sparse-attention kernels are CUDA-only, so non-CUDA devices will fail
  inside `diffsynth`, not in this file; that's called out in the command's
  docs rather than special-cased in code.
- Frame prep mirrors upstream's `prepare_input_tensor`: bicubic upscale by
  `--scale`, center-crop to a multiple of 128, repeat the last frame 4x,
  then truncate to the largest valid `8k+1` clip length. Below 5 source
  frames that truncation collapses to a single chunk with nothing to
  output, so `cli()` rejects short clips with a clear error instead of a
  cryptic `torch.cat` failure inside `_build_lq_proj`. The prepared LQ clip
  is kept on the CPU; the streaming pipeline moves each window to the device
  itself, so uploading the whole clip would only waste VRAM.
- `vide/commands/upscale_video.py`'s `MODELS` dict maps the CLI's
  `--model` choice to a model module's default weights repo id — routing
  for a second model architecture later, without a class hierarchy for a
  set of one.
- Tests split the same way as the source: `tests/test_flashvsr.py` covers
  the model module directly (pure tensor-prep/postprocess helpers and the
  vendored `_build_lq_proj`, including its streaming path, run for real on
  CPU); `tests/test_flashvsr_tcdecoder.py` runs the vendored TCDecoder for
  real on CPU; both need no CUDA or `diffsynth`. `load()`/`predict()` mock
  the `diffsynth` module via `sys.modules` (same idiom as
  `convert-depth-video`'s `transformers.pipeline` mock), plus
  `huggingface_hub.snapshot_download`, `build_tcdecoder`, and
  `_load_prompt_context` — the genuinely GPU/network-bound externals.
  `tests/test_upscale_video.py` covers the CLI only, reusing the same
  mocking fixtures.
- `encoder.stdin.write` is wrapped in `try/except BrokenPipeError`: when
  ffmpeg exits early (e.g. an unwritable output path), the pipe can break
  mid-write depending on frame-count/buffering timing, and `encoder.wait()`
  right after already reports the real failure.

## extract-frame

Input-side `-ss` seek (fast — ffmpeg jumps to the nearest keyframe instead
of decoding from the start), `-frames:v 1`. `--time -1` can't use `-ss`
(it doesn't seek from the end), so it uses `-sseof -1` plus image2's
`-update 1`: decode the final second, each frame overwriting the output
file, and the last frame is what remains. Timestamps are checked against
the probed container duration so a past-the-end `--time` fails with a
clear error instead of ffmpeg's "output file is empty" warning.
