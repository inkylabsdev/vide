transcribe-srt
==============

Transcribe an audio or video file into an ``.srt`` subtitle file. Speech is
transcribed with `WhisperX <https://github.com/m-bain/whisperX>`_ and aligned
to word-level timestamps, then shaped into readable, reading-speed-normalized
cues (max two lines, natural line breaks, min/max on-screen durations) modelled
on EBU-TT / Netflix subtitle guidelines.

Ported from `dashed/whisperx-subtitles-replicate
<https://github.com/dashed/whisperx-subtitles-replicate>`_ (readable-cue core;
translation, diarization, and neural sentence segmentation are not included).

Usage
-----

.. code-block:: console

   $ vide transcribe-srt MEDIA [OPTIONS]

.. code-block:: console

   $ vide transcribe-srt talk.mp4
   Loading whisper 'large-v3' on cuda ...
   Wrote 42 cue(s) (en) to talk.srt

``MEDIA`` may be any audio or video file ``ffmpeg`` can read — the audio track
is decoded automatically.

Options
-------

``-o, --output FILE``
   Output subtitle path. Defaults to ``./<media name>.srt``. Overwritten if it
   already exists.

``--model TEXT``
   faster-whisper model size (``tiny``, ``base``, ``small``, ``medium``,
   ``large-v3``) or a Hugging Face model id (default: ``large-v3``). Downloaded
   on first use.

``--language TEXT``
   ISO 639-1 code of the spoken language (e.g. ``en``, ``es``, ``ja``). Omit to
   auto-detect. Word-level alignment runs for languages WhisperX ships an
   alignment model for; others keep segment-level timing.

``--batch-size INTEGER``
   Number of audio chunks transcribed in parallel (default: ``16``). Lower it
   if you run out of memory.

``--max-line-length INTEGER``
   Maximum characters per subtitle line (default: ``42``).

``--max-lines INTEGER``
   Maximum lines per subtitle cue (default: ``2``).

``--max-cps FLOAT``
   Maximum reading speed in characters per second (default: ``17.0``).

``--min-duration FLOAT``
   Minimum seconds a cue stays on screen (default: ``1.0``).

``--max-duration FLOAT``
   Maximum seconds a cue stays on screen (default: ``7.0``).

Requirements
------------

WhisperX is an optional runtime dependency (it pulls in torch, CTranslate2, and
pyannote), so it is not installed with ``vide`` by default. Install it to use
this command:

.. code-block:: console

   $ pip install whisperx

The ``ffmpeg`` binary must also be available on ``PATH``.

Transcription runs on GPU when CUDA is available, otherwise on CPU — CTranslate2
(faster-whisper's backend) does not support Apple MPS, so on a Mac it runs on
CPU with the ``int8`` compute type.
