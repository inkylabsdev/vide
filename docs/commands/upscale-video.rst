upscale-video
=============

Upscale a video with a temporally consistent AI super-resolution model
(`FlashVSR <https://github.com/OpenImagingLab/FlashVSR>`_ by default), a
real-time diffusion-based video super-resolution model.

Usage
-----

.. code-block:: console

   $ vide upscale-video VIDEO [OPTIONS]

.. code-block:: console

   $ vide upscale-video input.mp4 --output upscaled.mp4 --scale 4

Options
-------

``-o, --output FILE``
   Output video path. Defaults to ``./<video name>_upscaled.mp4``.

``--model [flashvsr]``
   Video super-resolution model to use (default: ``flashvsr``).

``--scale FLOAT``
   Upscale factor (default: ``4.0``). FlashVSR is trained and recommended
   for 4x; other factors may be unstable.

``--sparse-ratio FLOAT``
   Sparse-attention ratio (default: ``2.0``). ``1.5`` is faster, ``2.0`` is
   more stable.

``--local-range INTEGER``
   Local attention window (default: ``11``). ``9`` is sharper, ``11`` is
   more stable.

``--kv-ratio FLOAT``
   Key/value cache ratio for the streaming attention (default: ``3.0``).

``--seed INTEGER``
   Diffusion sampling seed (default: ``0``).

``--color-fix / --no-color-fix``
   Wavelet color correction of the output against the input (default:
   enabled).

``--weights-dir DIRECTORY``
   Local directory to cache/read the downloaded model weights. Defaults to
   the Hugging Face Hub cache.

Requirements
------------

FlashVSR requires an NVIDIA GPU: its diffusion transformer and
locality-constrained sparse attention only run on CUDA. It also requires
the `diffsynth` package from the FlashVSR repository itself (not the
upstream DiffSynth-Studio package on PyPI) and the
`Block-Sparse-Attention <https://github.com/mit-han-lab/Block-Sparse-Attention>`_
extension, both installed per the
`FlashVSR README <https://github.com/OpenImagingLab/FlashVSR#getting-started>`_.
Model weights (~a few GB) are downloaded from the Hugging Face Hub on first
use.

The ``ffmpeg`` binary must be available on ``PATH``; the output is encoded
as H.264 (``yuv420p``) so it plays in browsers as well as desktop players.

Output resolution is the input size scaled by ``--scale`` and then
center-cropped down to a multiple of 128 pixels per side (FlashVSR's
required alignment), so it won't be an exact multiple of the input
resolution.

A clip needs at least 5 frames to upscale.
