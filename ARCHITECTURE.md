# Architecture

There barely is one, on purpose. `vide` is a thin CLI over tools that already
solved the hard problems — PySceneDetect, FFmpeg, Hugging Face — and the whole
job of this repo is to not get in their way.

## The shape

```
vide/
├── cli.py       # click group + auto-registration loop (that's it)
└── commands/    # one file per command, auto-discovered
```

One package, one entry point, one file per command. No `core/`, no `utils/`,
no service layer. If you're looking for the abstraction, there isn't one —
each command is a self-contained script that happens to share a `click` group.

## How commands are wired

`cli.py` iterates `vide.commands.*` with `pkgutil`, imports each module, and
registers anything that exposes a module-level `cli` that is a
`click.Command`. That's the entire plugin system: ~5 lines of stdlib.

To add a command, drop one file into `vide/commands/` with a `cli` command in
it. Done. No registry to update, no `__init__.py` to edit, no entry-points
dance in `pyproject.toml`.

Per-command design notes live in [`docs/internal.md`](docs/internal.md) —
this file stays about the shape, not the inventory.

## Deliberate choices

- **Heavy deps import lazily, inside the command function.** Every command
  module is imported at startup to build the CLI, so anything slow at module
  top taxes `vide --help`. Cheap deps can stay at module top. This is a
  convention, not enforced — keep honoring it.
- **Commands that run neural-network pipelines pick their device
  automatically: CUDA → MPS → CPU.** Practical, not principled — the user's
  best available hardware is almost always the right answer, so there's no
  `--device` flag to document, test, and get wrong.
- **FFmpeg is a runtime requirement, not a Python dep.** `ffmpeg-python` just
  builds argv; the `ffmpeg` binary must be on `PATH`. Bundling it would be
  more code and more wheels for zero gain.

## Tests

`tests/` runs the real CLI against tiny videos generated on the fly with
FFmpeg — no fixtures checked in, no mocking of ffmpeg. Coverage is gated at
100% (`--cov-fail-under=100` in `pyproject.toml`), which is only sustainable
because the codebase is this small. Keep it that way.

## Docs

- Always come with a command doc docs/commands/ and a short implementation notes in docs/internal.md.

## Build & release

`hatchling` builds, `uv` publishes, `Makefile` has the four targets you'd
guess (`test`, `sphinx`, `build`, `publish`). Docs are Sphinx in `docs/`, one
`.rst` page per command.
