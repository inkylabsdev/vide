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

## extract-frame

Input-side `-ss` seek (fast — ffmpeg jumps to the nearest keyframe instead
of decoding from the start), `-frames:v 1`. `--time -1` can't use `-ss`
(it doesn't seek from the end), so it uses `-sseof -1` plus image2's
`-update 1`: decode the final second, each frame overwriting the output
file, and the last frame is what remains. Timestamps are checked against
the probed container duration so a past-the-end `--time` fails with a
clear error instead of ffmpeg's "output file is empty" warning.

## transcribe-srt

Ported from `dashed/whisperx-subtitles-replicate`, trimmed to the
readable-cue core (no translation, diarization, MMS/neural sentence
segmentation). Decisions:

- The split mirrors that repo: model code (WhisperX load / transcribe /
  align) lives in `vide/models/whisperx.py`; the pure subtitle logic
  (sentence + clause splitting, cue merge/split, timing normalization, SRT
  render) lives in `vide/subtitles.py`, importing only stdlib + pysbd so it
  is unit-testable without torch. `vide.subtitles` is the first non-command,
  non-model module in the package — 750 lines of pure logic did not belong
  in the command file.
- WhisperX is an *optional runtime dependency*, not a hard one — like the
  ffmpeg binary (`ARCHITECTURE.md`). It drags in torch/CTranslate2/pyannote,
  so it is imported lazily and the command turns a missing import into a
  friendly "pip install whisperx". `pysbd` *is* a hard dep — it is tiny and
  pure-Python, and keeping it real lets the subtitle tests exercise the true
  segmenter.
- Because WhisperX is never installed in CI, the model wrapper is covered by
  injecting a fake `whisperx` module into `sys.modules` (its lazy imports
  resolve to the fake); the command monkeypatches the wrapper functions so
  the real subtitle pipeline still runs against canned segments. Same spirit
  as `convert-depth-video` mocking the transformers pipeline.
- CTranslate2 supports CUDA or CPU only (no MPS), and float16 is CUDA-only,
  so `_asr_device` maps the picked device down to cuda/float16 or cpu/int8 —
  the one place this command departs from the CUDA → MPS → CPU convention.
  Alignment (a torch wav2vec2 model) still uses the picked device directly.
- The timing model is the source's: text-shaping passes anchor cues to raw
  word start/end times, then a single `normalize_cues` pass enforces every
  timing invariant (ordering, non-overlap, reading-speed/CPS, min/max
  duration, lead-out past the last spoken word). Two defensive branches the
  pipeline can't reach with real input (`_split_to_fit` on an unsplittable
  single word, `_balance_two`'s function-word penalty) are covered by direct
  unit tests rather than left dead.
