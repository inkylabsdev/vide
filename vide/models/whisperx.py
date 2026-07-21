"""WhisperX speech-to-text with word-level alignment.

Wraps the whisperx pipeline: transcription (faster-whisper) plus optional
word-level forced alignment. Heavy dependencies are imported lazily so that
loading the vide CLI stays fast.

whisperx is an *optional* runtime requirement — like the ffmpeg binary, it is
not a hard Python dependency (it drags in torch/ctranslate2/pyannote). Install
it with ``pip install whisperx`` to use the transcribe-srt command.
"""

# faster-whisper size or a Hugging Face model id; large-v3 is the best-quality
# default. Swap for "small"/"medium" to trade accuracy for speed and memory.
DEFAULT_MODEL = "large-v3"


def _asr_device(device: str) -> tuple[str, str]:
    """Map the picked torch device to one CTranslate2 (faster-whisper's backend)
    supports. CTranslate2 runs on CUDA or CPU only — MPS is not supported — and
    float16 is a CUDA-only compute type, so CPU falls back to fast, low-memory
    int8."""
    if device == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def load(model: str, device: str):
    """Load a whisperx ASR model for `model` on the best available `device`."""
    import whisperx

    asr_device, compute_type = _asr_device(device)
    return whisperx.load_model(model, asr_device, compute_type=compute_type)


def transcribe(asr_model, media, *, batch_size: int = 16, language: str | None = None):
    """Transcribe `media` (any file ffmpeg can read, incl. video). Returns
    ``(audio, result)`` where `audio` is the decoded 16 kHz waveform (reused for
    alignment) and `result` is whisperx's dict with `segments` and the detected
    `language`."""
    import whisperx

    audio = whisperx.load_audio(str(media))
    result = asr_model.transcribe(audio, batch_size=batch_size, language=language)
    return audio, result


def align(audio, result, device: str):
    """Align `result` to word-level timestamps when whisperx ships an alignment
    model for the detected language; otherwise return it unchanged, leaving the
    segments with segment-level timing only."""
    import whisperx
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

    language = result["language"]
    if language not in DEFAULT_ALIGN_MODELS_TORCH and language not in (
        DEFAULT_ALIGN_MODELS_HF
    ):
        return result

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    return whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
