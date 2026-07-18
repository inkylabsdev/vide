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

Stream copy only, no re-encode path. The one decision is the default
output extension: probe the codec and map it to a container that holds it
(`CODEC_EXTENSIONS`), falling back to `.mka` — Matroska holds anything.
A video with no audio stream is an error, not a silent no-op.

## remove-audio

`-c copy -an`: every non-audio stream (video, subtitles) is copied
untouched. Default output keeps the input's extension, `_noaudio` suffix.

## convert-depth-video

Per-frame depth estimation via a `transformers` pipeline (Depth Anything V2
`-hf` checkpoint by default — the non-`-hf` repos lack `config.json` and
don't load). Decisions:

- `torch`/`transformers`/`cv2` import inside the command function; see the
  lazy-import convention in `ARCHITECTURE.md`.
- Linux CI installs CPU torch wheels (`tool.uv.sources` in `pyproject.toml`)
  — the default CUDA wheels are ~2.5 GB and CI has no GPU.
- Output is encoded by piping raw BGR frames into an ffmpeg subprocess as
  H.264/yuv420p with `+faststart` — `cv2.VideoWriter`'s mp4v codec isn't
  web-playable.
- Device pick follows the CUDA → MPS → CPU convention in `ARCHITECTURE.md`.
- Depth is min-max normalized per frame before colormapping, so absolute
  depth scale is not preserved across frames.
