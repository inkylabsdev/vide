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

The audio stream is copied bit-for-bit — instant and lossless — but that
means the output container must be able to hold the codec. The default
output name always is; if you pass ``--output`` yourself, pick an
extension that matches the codec.

A video with no audio track is an error.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``.
