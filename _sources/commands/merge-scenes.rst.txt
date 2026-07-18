merge-scenes
============

Concatenate clips into a single video, in the order given, without
re-encoding. Equivalent to:

.. code-block:: console

   $ ffmpeg -f concat -safe 0 -i inputs.txt -c copy output.mp4

Usage
-----

.. code-block:: console

   $ vide merge-scenes CLIP... [OPTIONS]

.. code-block:: console

   $ vide merge-scenes /tmp/clips/*.mp4 --output final.mp4

Options
-------

``-o, --output FILE``
   Path of the merged video (default: ``merged.mp4``). Overwritten if it
   already exists.

Notes
-----

Because streams are copied rather than re-encoded, all clips should share
the same codec, resolution, and framerate — which is the case for clips
produced by ``vide split-scenes`` from the same source video.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``.
