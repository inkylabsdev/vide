extract-audio
=============

Extract a video's audio track into its own file, without re-encoding.
Equivalent to:

.. code-block:: console

   $ ffmpeg -i input.mp4 -map 0:a -acodec copy output.m4a

Usage
-----

.. code-block:: console

   $ vide extract-audio VIDEO [OPTIONS]

.. code-block:: console

   $ vide extract-audio talk.mp4
   Extracted aac audio to talk.m4a

Options
-------

``-o, --output FILE``
   Path of the extracted audio file. Defaults to ``./<video name>.<ext>``,
   where the extension is chosen to match the audio codec (``.m4a`` for
   AAC, ``.mp3`` for MP3, ``.opus``, ``.ogg``, ``.flac``; ``.mka`` for
   anything else). Overwritten if it already exists.

Notes
-----

When the output container can hold the source codec (always true for the
default output name), the audio stream is copied bit-for-bit — instant
and lossless. When it can't (e.g. extracting AAC audio to ``.mp3``), the
audio is re-encoded to the container's default codec instead.

A video with no audio track is an error.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``.
