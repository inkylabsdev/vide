"""Convert a video into a colorized per-frame depth-map video.

Depth is estimated per frame with Depth Anything V2. Heavy dependencies
are imported lazily so that loading the vide CLI stays fast.
"""

from pathlib import Path

import click

from vide import models
from vide.models import depth_anything_v2


@click.command("convert-depth-video")
@click.argument(
    "video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output video path. Defaults to ./<video name>_depth.mp4.",
)
@click.option(
    "--model",
    default=depth_anything_v2.DEFAULT_MODEL,
    show_default=True,
    help="Hugging Face depth-estimation model to use.",
)
@click.option(
    "--colormap",
    type=click.Choice(["inferno", "magma", "viridis", "jet"]),
    default="inferno",
    show_default=True,
    help="OpenCV colormap used to render the depth map.",
)
def cli(video: Path, output: Path | None, model: str, colormap: str):
    """Estimate depth for every frame of VIDEO and write a colorized depth video."""
    import cv2
    import ffmpeg

    if output is None:
        output = Path.cwd() / f"{video.stem}_depth.mp4"

    device = models.pick_device()
    click.echo(f"Loading {model} on {device} ...")
    depth_estimator = depth_anything_v2.load(model, device)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Encode H.264/yuv420p via ffmpeg so the result plays in browsers too;
    # cv2.VideoWriter's mp4v (MPEG-4 Part 2) is not web-playable.
    encoder = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{width}x{height}",
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

    colormap_id = getattr(cv2, f"COLORMAP_{colormap.upper()}")
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = depth_anything_v2.predict(depth_estimator, rgb)
        normalized = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        encoder.stdin.write(cv2.applyColorMap(normalized, colormap_id).tobytes())
        frames += 1

    cap.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise click.ClickException("ffmpeg failed to encode the output video")
    click.echo(f"Wrote {frames} frame(s) to {output}")
