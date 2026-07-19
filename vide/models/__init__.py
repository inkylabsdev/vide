"""Model-specific code, one module per model.

Command modules stay about video I/O and CLI concerns; anything tied to
a particular neural network (checkpoint names, pipeline loading,
inference) lives here. Heavy dependencies are imported lazily so that
loading the vide CLI stays fast.
"""


def pick_device() -> str:
    """Best available torch device, per the CUDA -> MPS -> CPU convention."""
    import torch

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
