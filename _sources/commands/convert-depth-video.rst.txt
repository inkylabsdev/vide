convert-depth-video
===================

Estimate per-frame depth for a video and write a colorized depth-map
video, using a Hugging Face depth-estimation model
(`Depth Anything V2 <https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf>`_
by default).

Usage
-----

.. code-block:: console

   $ vide convert-depth-video VIDEO [OPTIONS]

.. code-block:: console

   $ vide convert-depth-video input.mp4 --output depth.mp4 --colormap magma

Options
-------

``-o, --output FILE``
   Output video path. Defaults to ``./<video name>_depth.mp4``.

``--model TEXT``
   Hugging Face depth-estimation model (default:
   ``depth-anything/Depth-Anything-V2-Base-hf``). Downloaded on first use.

``--colormap [inferno|magma|viridis|jet]``
   OpenCV colormap used to render the depth map (default: ``inferno``).

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``; the output is
encoded as H.264 (``yuv420p``) so it plays in browsers as well as
desktop players.

Runs on GPU when CUDA or Apple MPS is available, otherwise on CPU.
