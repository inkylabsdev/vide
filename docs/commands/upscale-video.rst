upscale-video
=============

Upscale a video using an ML super-resolution model. Each frame is upscaled
individually via the selected model and the result is re-assembled into
H.264/yuv420p with ffmpeg.

Usage
-----

.. code-block:: console

   $ vide upscale-video VIDEO [OPTIONS]

.. code-block:: console

   $ vide upscale-video input.mp4 --output hires.mp4

.. code-block:: console

   $ vide upscale-video anime.mp4 --model anime4k --output anime_hires.mp4

Options
-------

``-o, --output FILE``
   Output video path. Defaults to ``./<video name>_upscaled.mp4``.

``--model [real-esrgan|anime4k]``
   Super-resolution model to use (default: ``real-esrgan``).

   * ``real-esrgan`` — `Real-ESRGAN <https://github.com/xinntao/Real-ESRGAN>`_:
     a general-purpose super-resolution model that works well on live-action
     and mixed content.
   * ``anime4k`` — `Anime4K <https://github.com/bloc97/Anime4K>`_:
     a high-quality, real-time anime upscaling algorithm, best suited for
     animated content.

   Both models run via `Replicate <https://replicate.com>`_ and require a
   valid ``REPLICATE_API_TOKEN`` environment variable.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``; the output is encoded
as H.264 (``yuv420p``) so it plays in browsers as well as desktop players.

A ``REPLICATE_API_TOKEN`` environment variable is required for the Replicate
API calls.
