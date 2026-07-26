"""Anime4K image upscaling via Replicate.

Heavy dependencies are imported lazily so that loading the vide CLI stays fast.
"""

REPLICATE_MODEL = "phamquiluan/anime4k"


def upscale(image_bytes: bytes) -> bytes:
    """Upscale an image via Anime4K on Replicate; returns raw image bytes."""
    import io

    import replicate

    output = replicate.run(
        REPLICATE_MODEL,
        input={"image": io.BytesIO(image_bytes)},
    )
    return output.read()
