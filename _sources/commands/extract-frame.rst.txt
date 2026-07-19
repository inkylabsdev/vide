extract-frame
=============

Extract a single frame from a video as an image. Equivalent to:

.. code-block:: console

   $ ffmpeg -ss 00:01.11 -i input.mp4 -frames:v 1 output.png

Usage
-----

.. code-block:: console

   $ vide extract-frame VIDEO [OPTIONS]

.. code-block:: console

   $ vide extract-frame talk.mp4 --time 00:01.11
   Extracted frame at 00:01.11 to talk_00-01.11.png

Options
-------

``-t, --time TIME``
   Timestamp of the frame to extract, in ffmpeg time syntax:
   ``[[HH:]MM:]SS[.ms]``, e.g. ``00:01.11`` or plain seconds like ``5.5``.
   The special value ``-1`` picks the last frame. Defaults to ``00:00``,
   the first frame.

``-o, --output FILE``
   Path of the extracted image; the extension picks the format (``.png``,
   ``.jpg``, …). Defaults to ``./<video name>_<time>.png``, with ``:``
   replaced by ``-`` (``_last`` for ``--time -1``). Overwritten if it
   already exists.

Notes
-----

A ``--time`` at or past the video's duration is an error.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``.
