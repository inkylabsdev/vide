split-scenes
============

Detect scenes in a video and split each one into a clip, using
`PySceneDetect <https://www.scenedetect.com/>`_ for detection and
``ffmpeg`` for splitting.

Usage
-----

.. code-block:: console

   $ vide split-scenes VIDEO [OPTIONS]

.. code-block:: console

   $ vide split-scenes input.mp4 --output /tmp/clips

Clips are named ``<video name>-Scene-001.mp4``, ``<video name>-Scene-002.mp4``,
and so on.

Options
-------

``-o, --output DIRECTORY``
   Directory to write clips to. Created if it does not exist.
   Defaults to ``./<video name>_scenes``.

``--threshold FLOAT``
   Scene cut sensitivity (default: ``27.0``). Lower values detect more
   cuts; raise it if the video is being split too aggressively.

Requirements
------------

The ``ffmpeg`` binary must be available on ``PATH``.
