"""Depth Anything V2 per-frame depth estimation.

Wraps the Hugging Face depth-estimation pipeline. Heavy dependencies
are imported lazily so that loading the vide CLI stays fast.
"""

# The -hf repos hold the transformers-compatible checkpoints; the plain
# Depth-Anything-V2-* repos are raw research checkpoints without config.json.
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"


def load(model: str, device: str):
    """Load a depth-estimation pipeline for `model` on `device`."""
    from transformers import pipeline

    return pipeline("depth-estimation", model=model, device=device)


def estimate(estimator, rgb):
    """Estimate depth for an RGB frame array; returns a 2-D numpy array."""
    import numpy as np
    from PIL import Image

    return np.array(estimator(Image.fromarray(rgb))["depth"])
