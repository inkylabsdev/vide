animate-avatar
==============

Animate an avatar video from a reference image and an audio file via a running
local Cog HTTP server for LongCat Video Avatar 1.5.

Usage
-----

.. code-block:: console

   $ vide animate-avatar REFERENCE_IMAGE AUDIO [OPTIONS]

.. code-block:: console

   $ vide animate-avatar avatar.png speech.wav --output avatar.mp4

Options
-------

``-o, --output FILE``
   Output video path. Defaults to ``./<audio name>_avatar.mp4``.

``--model [longcat-avatar]``
   Avatar animation model to use (default: ``longcat-avatar``).

``--server-url TEXT``
   Base URL of the running Cog HTTP server
   (default: ``http://127.0.0.1:5000``).

``--timeout INTEGER``
   Seconds to wait for prediction completion (default: ``600``).

Requirements
------------

This command posts requests to a running Cog HTTP server and writes the output
video locally. Ensure:

* ``cog serve`` is running from ``models/longcat-video-avatar-1.5``
* ``models/longcat-video-avatar-1.5`` contains the Cog package for
  `meituan-longcat/LongCat-Video-Avatar-1.5 <https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5>`_
