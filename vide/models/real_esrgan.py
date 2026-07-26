"""Real-ESRGAN image upscaling via Replicate.

Heavy dependencies are imported lazily so that loading the vide CLI stays fast.
"""

REPLICATE_MODEL = "nightmareai/real-esrgan"
DEFAULT_SCALE = 4


def upscale(image_bytes: bytes, scale: int = DEFAULT_SCALE) -> bytes:
    """Upscale an image via Real-ESRGAN on Replicate; returns raw image bytes."""
    import io

    import replicate

    output = replicate.run(
        REPLICATE_MODEL,
        input={"image": io.BytesIO(image_bytes), "scale": scale},
    )
    return output.read()
