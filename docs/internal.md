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

Whole-video 4x super-resolution via FlashVSR
(https://github.com/OpenImagingLab/FlashVSR), a one-step diffusion VSR
model. Decisions:

- Model-specific code (weight download, pipeline load, inference, and the
  FlashVSR-specific frame prep/postprocess) lives in
  `vide/models/flashvsr.py`, same split as `convert-depth-video` /
  `depth_anything_v2.py`; the command keeps only video I/O, ffmpeg
  encoding, and option wiring. FlashVSR has no `transformers.pipeline`-
  style entry point to wrap: it ships as a fork of the `diffsynth` package
  plus a small LQ-projection module the upstream repo keeps in its example
  scripts (`utils/utils.py`) rather than the installable package. That
  module is vendored in `flashvsr.py`
  (`_build_lq_proj`, adapted from `Buffer_LQ4x_Proj`, Apache-2.0) since it
  isn't importable any other way; everything else (DiT, VAE, sparse
  attention) comes from `diffsynth` itself, imported lazily and never
  vendored.
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
- Whole video is processed in one pipeline call (no long-video streaming
  window), matching upstream's basic `infer_flashvsr_full.py` example
  rather than its `_long_video` variant.
- Frame prep mirrors upstream's `prepare_input_tensor`: bicubic upscale by
  `--scale`, center-crop to a multiple of 128, repeat the last frame 4x,
  then truncate to the largest valid `8k+1` clip length. Below 5 source
  frames that truncation collapses to a single chunk with nothing to
  output, so `cli()` rejects short clips with a clear error instead of a
  cryptic `torch.cat` failure inside `_build_lq_proj`.
- `vide/commands/upscale_video.py`'s `MODELS` dict maps the CLI's
  `--model` choice to a model module's default weights repo id — routing
  for a second model architecture later, without a class hierarchy for a
  set of one.
- Tests split the same way as the source: `tests/test_flashvsr.py` covers
  the model module directly (pure tensor-prep/postprocess helpers and the
  vendored `_build_lq_proj` run for real on CPU — no CUDA or `diffsynth`
  needed to verify their math; `load()`/`predict()` mock the `diffsynth`
  module via `sys.modules`, same idiom as `convert-depth-video`'s
  `transformers.pipeline` mock, plus `huggingface_hub.snapshot_download`).
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
