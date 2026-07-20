"""Upscale a video with a temporally consistent AI video super-resolution
model.

FlashVSR by default. Heavy dependencies are imported lazily so that
loading the vide CLI stays fast.
"""

from pathlib import Path

import click

from vide import models
from vide.models import flashvsr

DEFAULT_MODEL = "flashvsr"
MODELS = {
    "flashvsr": flashvsr.DEFAULT_MODEL,
}


@click.command("upscale-video")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output video path. Defaults to ./<video name>_upscaled.mp4.",
)
@click.option(
    "--model",
    type=click.Choice(sorted(MODELS)),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Video super-resolution model to use.",
)
@click.option(
    "--scale",
    type=float,
    default=flashvsr.SCALE,
    show_default=True,
    help="Upscale factor. FlashVSR is trained and recommended for 4x.",
)
@click.option(
    "--sparse-ratio",
    type=float,
    default=flashvsr.DEFAULT_SPARSE_RATIO,
    show_default=True,
    help="Sparse-attention ratio: 1.5 is faster, 2.0 is more stable.",
)
@click.option(
    "--local-range",
    type=int,
    default=flashvsr.DEFAULT_LOCAL_RANGE,
    show_default=True,
    help="Local attention window: 9 is sharper, 11 is more stable.",
)
@click.option(
    "--kv-ratio",
    type=float,
    default=flashvsr.DEFAULT_KV_RATIO,
    show_default=True,
    help="Key/value cache ratio for the streaming attention.",
)
@click.option(
    "--seed", type=int, default=0, show_default=True, help="Diffusion sampling seed."
)
@click.option(
    "--color-fix/--no-color-fix",
    default=True,
    show_default=True,
    help="Wavelet color correction of the output against the input.",
)
@click.option(
    "--weights-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Local directory to cache/read the downloaded model weights.",
)
def cli(
    video: Path,
    output: Path | None,
    model: str,
    scale: float,
    sparse_ratio: float,
    local_range: int,
    kv_ratio: float,
    seed: int,
    color_fix: bool,
    weights_dir: Path | None,
):
    """Upscale VIDEO with a temporally consistent AI video super-resolution model."""
    import os

    # Reduce CUDA fragmentation OOMs on tight (e.g. 12 GB) cards. Must be set
    # before torch initialises CUDA, hence before the import below.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cv2
    import ffmpeg
    import torch

    if output is None:
        output = Path.cwd() / f"{video.stem}_upscaled.mp4"

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < flashvsr.MIN_FRAMES:
        raise click.ClickException(
            f"{video} has too few frames to upscale "
            f"(need at least {flashvsr.MIN_FRAMES}, got {len(frames)})"
        )

    scaled_w, scaled_h, target_w, target_h = flashvsr.compute_scaled_dims(
        width, height, scale
    )
    click.echo(
        f"Preparing {len(frames)} frame(s): {width}x{height} -> {target_w}x{target_h} ..."
    )
    lq_video, num_frames = flashvsr.prepare_lq_video(
        frames, scaled_w, scaled_h, target_w, target_h
    )

    device = models.pick_device()
    click.echo(f"Loading {model} on {device} ...")
    pipe = flashvsr.load(MODELS[model], weights_dir, device)

    click.echo("Running FlashVSR ...")
    # Keep the low-quality conditioning clip on the CPU; the streaming
    # pipeline moves each window to the device itself, so uploading the whole
    # clip would just waste VRAM.
    result = flashvsr.predict(
        pipe,
        lq_video.to(dtype=torch.bfloat16),
        num_frames=num_frames,
        height=target_h,
        width=target_w,
        seed=seed,
        sparse_ratio=sparse_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
        color_fix=color_fix,
    )

    # Encode H.264/yuv420p via ffmpeg so the result plays in browsers too;
    # cv2.VideoWriter's mp4v (MPEG-4 Part 2) is not web-playable.
    encoder = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{target_w}x{target_h}",
            framerate=fps,
        )
        .output(
            str(output),
            vcodec="libx264",
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        .overwrite_output()
        .global_args("-loglevel", "error")
        .run_async(pipe_stdin=True)
    )
    written = 0
    try:
        for frame in flashvsr.tensor_to_frames(result):
            encoder.stdin.write(frame.tobytes())
            written += 1
    except BrokenPipeError:
        # ffmpeg exited early (e.g. it couldn't open the output path) and
        # closed its end of the pipe; encoder.wait() below reports why.
        pass
    else:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise click.ClickException("ffmpeg failed to encode the output video")
    click.echo(f"Wrote {written} frame(s) to {output}")
