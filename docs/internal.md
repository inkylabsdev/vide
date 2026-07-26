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

Frame-by-frame super-resolution via Replicate. Decisions:

- Two models are supported: `real-esrgan` (default) and `anime4k`. Each lives
  in its own `vide/models/` module so the dispatch in the command is a simple
  `if/else` import; adding a third model means adding one file and one branch.
- The first frame is decoded, upscaled, and decoded again *before* the ffmpeg
  encoder is started, so the output dimensions are known in advance. The
  `VideoCapture` is then rewound to frame 0 before the main loop.
- Heavy deps (`cv2`, `numpy`, `ffmpeg`, `replicate`) are imported lazily
  inside the command function; see the lazy-import convention in
  `ARCHITECTURE.md`.
- Output is H.264/yuv420p with `+faststart`, same as `convert-depth-video`.
- Replicate returns a `FileOutput`-like object; `.read()` gives raw image
  bytes.  The fake in tests returns a `io.BytesIO` with the same interface.

## extract-frame

Input-side `-ss` seek (fast — ffmpeg jumps to the nearest keyframe instead
of decoding from the start), `-frames:v 1`. `--time -1` can't use `-ss`
(it doesn't seek from the end), so it uses `-sseof -1` plus image2's
`-update 1`: decode the final second, each frame overwriting the output
file, and the last frame is what remains. Timestamps are checked against
the probed container duration so a past-the-end `--time` fails with a
clear error instead of ffmpeg's "output file is empty" warning.
