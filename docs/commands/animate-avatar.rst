animate-avatar
==============

Animate an avatar video from a reference image and an audio file using a local
Cog package for LongCat Video Avatar 1.5.

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

Requirements
------------

This command runs ``cog predict`` locally in
``models/longcat-video-avatar-1.5``. Ensure:

* ``cog`` is installed and available on ``PATH``
* ``models/longcat-video-avatar-1.5`` contains a valid Cog package for
  `meituan-longcat/LongCat-Video-Avatar-1.5 <https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5>`_
