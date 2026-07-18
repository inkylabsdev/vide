remove-audio
============

Write a copy of a video with the audio track stripped, without
re-encoding. Equivalent to:

.. code-block:: console

   $ ffmpeg -i input.mp4 -c copy -an output.mp4

Usage
-----

.. code-block:: console

   $ vide remove-audio VIDEO [OPTIONS]

.. code-block:: console

   $ vide remove-audio talk.mp4
   Wrote audio-free copy to talk_noaudio.mp4

Options
-------

``-o, --output FILE``
   Path of the audio-free video. Defaults to
   ``./<video name>_noaudio<ext>``, keeping the input's extension.
   Overwritten if it already exists.

Notes
-----

All non-audio streams (video, subtitles) are copied untouched, so the
command is instant and lossless.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``.
