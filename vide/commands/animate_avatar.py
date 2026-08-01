"""Animate an avatar with a reference image and audio via a running Cog server."""

import base64
import json
import mimetypes
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

import click

DEFAULT_MODEL = "longcat-avatar"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "longcat-video-avatar-1.5"
)
DEFAULT_SERVER_URL = "http://127.0.0.1:5000"
DEFAULT_TIMEOUT = 600


def _file_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _post_json(url: str, payload: dict) -> dict:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    with request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_output_bytes(output_ref: str) -> bytes:
    if output_ref.startswith("data:"):
        _, encoded = output_ref.split(",", 1)
        return base64.b64decode(encoded)
    if output_ref.startswith("http://") or output_ref.startswith("https://"):
        with request.urlopen(output_ref) as response:
            return response.read()
    output_path = Path(output_ref)
    if output_path.exists():
        return output_path.read_bytes()
    raise click.ClickException(f"Unsupported output reference from server: {output_ref}")


def _extract_output(prediction: dict) -> str:
    output = prediction.get("output")
    if isinstance(output, list):
        if not output:
            raise click.ClickException("Server returned empty output list.")
        output = output[0]
    if not isinstance(output, str) or not output:
        raise click.ClickException("Server did not return a valid output reference.")
    return output


@click.command("animate-avatar")
@click.argument(
    "reference_image", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output video path. Defaults to ./<audio name>_avatar.mp4.",
)
@click.option(
    "--model",
    type=click.Choice([DEFAULT_MODEL]),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Avatar animation model to use.",
)
@click.option(
    "--server-url",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="Base URL of the running Cog HTTP server.",
)
@click.option(
    "--timeout",
    type=click.IntRange(min=1),
    default=DEFAULT_TIMEOUT,
    show_default=True,
    help="Seconds to wait for server-side prediction completion.",
)
def cli(
    reference_image: Path,
    audio: Path,
    output: Path | None,
    model: str,
    server_url: str,
    timeout: int,
):
    """Animate REFERENCE_IMAGE with AUDIO via the LongCat Cog HTTP server."""
    _ = model
    if output is None:
        output = Path.cwd() / f"{audio.stem}_avatar.mp4"

    if not DEFAULT_MODEL_DIR.exists():
        raise click.ClickException(
            f"Model directory not found: {DEFAULT_MODEL_DIR}. "
            "Expected a local Cog package at models/longcat-video-avatar-1.5."
        )

    create_url = f"{server_url.rstrip('/')}/predictions"
    payload = {
        "input": {
            "reference_image": _file_to_data_url(reference_image),
            "audio": _file_to_data_url(audio),
        }
    }

    try:
        prediction = _post_json(create_url, payload)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise click.ClickException(f"Failed to call Cog server: {exc}") from exc

    start = time.time()
    status = prediction.get("status")
    while status in {"starting", "processing"}:
        poll_url = prediction.get("urls", {}).get("get")
        if not poll_url:
            raise click.ClickException("Prediction is pending but no poll URL was returned.")
        if time.time() - start >= timeout:
            raise click.ClickException("Timed out waiting for prediction to complete.")
        time.sleep(1)
        try:
            prediction = _get_json(poll_url)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise click.ClickException(f"Failed to poll prediction status: {exc}") from exc
        status = prediction.get("status")

    if status and status != "succeeded":
        error = prediction.get("error") or "unknown error"
        raise click.ClickException(f"cog prediction failed: {error}")

    output_ref = _extract_output(prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = _read_output_bytes(output_ref)
    output.write_bytes(output_bytes)
    click.echo(f"Animated avatar written to {output}")
