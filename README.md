# vide

[![Tests](https://github.com/inkylabsdev/vide/actions/workflows/test.yml/badge.svg)](https://github.com/inkylabsdev/vide/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/vide.svg)](https://pypi.org/project/vide/)
[![Python](https://img.shields.io/pypi/pyversions/vide.svg)](https://pypi.org/project/vide/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A video editing utility for the command line. `vide` wraps
[PySceneDetect](https://www.scenedetect.com/) and [FFmpeg](https://ffmpeg.org/)
behind a small, composable CLI: split a video into scene clips, edit or drop
the ones you don't want, and merge the rest back together — all without
re-encoding on the merge side.

## Requirements

- Python 3.12+
- The `ffmpeg` binary on your `PATH` (`brew install ffmpeg`, `apt install ffmpeg`, …)

## Installation

```console
$ uv tool install vide-python
# or
$ pip install vide-python
```

## Usage

### Split a video into scenes

Detects scene cuts and writes one clip per scene:

```console
$ vide split-scenes input.mp4 --output /tmp/clips
$ ls /tmp/clips
input-Scene-001.mp4  input-Scene-002.mp4  input-Scene-003.mp4
```

| Option | Description |
| --- | --- |
| `-o, --output DIRECTORY` | Where to write clips. Defaults to `./<video name>_scenes`. |
| `--threshold FLOAT` | Scene cut sensitivity (default `27.0`). Lower detects more cuts. |

### Merge clips into one video

Concatenates clips in the order given using FFmpeg's concat demuxer with
stream copy — instant and lossless, no re-encoding:

```console
$ vide merge-scenes /tmp/clips/*.mp4 --output final.mp4
```

| Option | Description |
| --- | --- |
| `-o, --output FILE` | Path of the merged video (default `merged.mp4`). |

Because streams are copied, all inputs must share the same codec, resolution,
and framerate — always true for clips produced by `vide split-scenes` from the
same source.

## Documentation

Full documentation lives in [`docs/`](docs/). Build it locally with:

```console
$ make sphinx
$ open docs/_build/html/index.html
```

## Development

```console
$ git clone https://github.com/inkylabsdev/vide.git
$ cd vide
$ uv sync
$ make test
```

The test suite runs real `vide` invocations against videos generated on the
fly with FFmpeg, and enforces 100% coverage.

Adding a command is a single file: drop a module into `vide/commands/`
exposing a module-level `cli` [click](https://click.palletsprojects.com/)
command, and it is registered automatically.

## License

[MIT](LICENSE)
