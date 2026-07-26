"""Upscale a video using an ML super-resolution model.

Frames are extracted with OpenCV, upscaled one-by-one via the chosen model,
and re-assembled into H.264/yuv420p with ffmpeg. Heavy dependencies are
imported lazily so that loading the vide CLI stays fast.
"""

from pathlib import Path

import click

DEFAULT_MODEL = "real-esrgan"


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
    type=click.Choice(["real-esrgan", "anime4k"]),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Super-resolution model to use.",
)
def cli(video: Path, output: Path | None, model: str):
    """Upscale VIDEO using an ML super-resolution model."""
    import cv2
    import ffmpeg
    import numpy as np

    if output is None:
        output = Path.cwd() / f"{video.stem}_upscaled.mp4"

    if model == "real-esrgan":
        from vide.models import real_esrgan

        upscale_fn = real_esrgan.upscale
    else:
        from vide.models import anime4k

        upscale_fn = anime4k.upscale

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample the first frame to determine the upscaled output dimensions.
    ret, sample = cap.read()
    if not ret:
        raise click.ClickException(f"{video} has no readable frames")
    _, sample_png = cv2.imencode(".png", sample)
    sample_up = cv2.imdecode(
        np.frombuffer(upscale_fn(bytes(sample_png)), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    out_height, out_width = sample_up.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    encoder = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{out_width}x{out_height}",
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

    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, png_bytes = cv2.imencode(".png", frame)
        upscaled = cv2.imdecode(
            np.frombuffer(upscale_fn(bytes(png_bytes)), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        encoder.stdin.write(upscaled.tobytes())
        frames += 1

    cap.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise click.ClickException("ffmpeg failed to encode the output video")
    click.echo(f"Upscaled {frames} frame(s) to {output}")
