# longcat-video-avatar-1.5 (Cog package)

This Cog package runs the official LongCat-Video-Avatar-1.5 inference workflow
(from `meituan-longcat/LongCat-Video`) instead of a local ffmpeg stub.

`predict.py` will:

1. Download the official LongCat-Video source archive at a pinned commit.
2. Download model weights for:
   - `meituan-longcat/LongCat-Video`
   - `meituan-longcat/LongCat-Video-Avatar-1.5`
3. Run `run_demo_avatar_single_audio_to_video.py` with `torchrun` for `ai2v`.

## Start the server

```console
$ cd models/longcat-video-avatar-1.5
$ cog serve
```

The `vide animate-avatar` CLI assumes this HTTP server is running
(default: `http://127.0.0.1:5000`).
