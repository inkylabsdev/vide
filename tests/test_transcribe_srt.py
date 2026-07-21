"""Tests for the transcribe-srt command and the whisperx model wrapper.

whisperx is an optional runtime dependency (torch/ctranslate2/pyannote), so it
is never imported for real here: the model wrapper is exercised against a fake
`whisperx` module injected into sys.modules, and the command monkeypatches the
wrapper functions so the real (pure) subtitle pipeline still runs.
"""

import sys
import types

from click.testing import CliRunner

from vide.cli import cli
from vide.models import whisperx as whisperx_model


def _words(*pairs):
    return [
        {"word": word, "start": start, "end": end, "score": 0.9}
        for word, start, end in pairs
    ]


SEGMENTS = [
    {
        "text": "Hello there, how are you doing today?",
        "start": 0.0,
        "end": 2.4,
        "words": _words(
            ("Hello", 0.0, 0.4), ("there,", 0.4, 0.8), ("how", 0.9, 1.1),
            ("are", 1.1, 1.3), ("you", 1.3, 1.6), ("doing", 1.6, 2.0),
            ("today?", 2.0, 2.4),
        ),
    }
]


# --------------------------------------------------------------------------- #
# Model wrapper (fake whisperx)
# --------------------------------------------------------------------------- #
def _install_fake_whisperx(monkeypatch, *, align_langs=("en",)):
    """Inject a minimal fake `whisperx` (and whisperx.alignment) into
    sys.modules so the wrapper's lazy imports resolve to it."""
    calls = {}

    class FakeASR:
        def transcribe(self, audio, batch_size, language):
            calls["transcribe"] = {"audio": audio, "batch_size": batch_size, "language": language}
            return {"segments": SEGMENTS, "language": language or "en"}

    fake = types.ModuleType("whisperx")

    def load_model(model, device, compute_type):
        calls["load_model"] = {"model": model, "device": device, "compute_type": compute_type}
        return FakeASR()

    def load_audio(path):
        calls["load_audio"] = path
        return f"waveform::{path}"

    def load_align_model(language_code, device):
        calls["load_align_model"] = {"language_code": language_code, "device": device}
        return ("model_a", "metadata")

    def align(segments, model_a, metadata, audio, device, return_char_alignments):
        calls["align"] = {"device": device, "return_char_alignments": return_char_alignments}
        return {"segments": segments, "language": "en"}

    fake.load_model = load_model
    fake.load_audio = load_audio
    fake.load_align_model = load_align_model
    fake.align = align

    alignment = types.ModuleType("whisperx.alignment")
    alignment.DEFAULT_ALIGN_MODELS_TORCH = {lang: "x" for lang in align_langs}
    alignment.DEFAULT_ALIGN_MODELS_HF = {}
    fake.alignment = alignment

    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", alignment)
    return calls


def test_asr_device_mapping():
    assert whisperx_model._asr_device("cuda") == ("cuda", "float16")
    assert whisperx_model._asr_device("mps") == ("cpu", "int8")
    assert whisperx_model._asr_device("cpu") == ("cpu", "int8")


def test_wrapper_load_maps_device(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    whisperx_model.load("large-v3", "mps")
    assert calls["load_model"] == {
        "model": "large-v3", "device": "cpu", "compute_type": "int8"
    }


def test_wrapper_transcribe_returns_audio_and_result(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch)
    asr = whisperx_model.load("large-v3", "cpu")
    audio, result = whisperx_model.transcribe(asr, "clip.mp4", batch_size=8, language="en")
    assert audio == "waveform::clip.mp4"
    assert result["language"] == "en"
    assert calls["transcribe"]["batch_size"] == 8


def test_wrapper_align_runs_for_supported_language(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch, align_langs=("en",))
    result = {"language": "en", "segments": SEGMENTS}
    out = whisperx_model.align("waveform", result, "cpu")
    assert calls["load_align_model"]["language_code"] == "en"
    assert out["segments"] == SEGMENTS


def test_wrapper_align_skips_unsupported_language(monkeypatch):
    calls = _install_fake_whisperx(monkeypatch, align_langs=("en",))
    result = {"language": "zz", "segments": SEGMENTS}
    out = whisperx_model.align("waveform", result, "cpu")
    assert out is result  # returned unchanged
    assert "load_align_model" not in calls


# --------------------------------------------------------------------------- #
# Command
# --------------------------------------------------------------------------- #
def _patch_command(monkeypatch, *, segments=SEGMENTS, language="en", load=None):
    """Patch the wrapper functions the command calls, and pin the device."""
    monkeypatch.setattr("vide.models.pick_device", lambda: "cpu")
    if load is None:
        monkeypatch.setattr("vide.models.whisperx.load", lambda model, device: "ASR")
    else:
        monkeypatch.setattr("vide.models.whisperx.load", load)
    def _transcribe(asr, media, batch_size, language):
        return "AUDIO", {"segments": segments, "language": language or "en"}

    monkeypatch.setattr("vide.models.whisperx.transcribe", _transcribe)
    monkeypatch.setattr(
        "vide.models.whisperx.align",
        lambda audio, result, device: {"segments": segments, "language": language},
    )


def test_cli_registers_transcribe_srt():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "transcribe-srt" in result.output


def test_transcribe_srt_writes_srt(tmp_path, monkeypatch):
    _patch_command(monkeypatch)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    output = tmp_path / "out.srt"

    result = CliRunner().invoke(
        cli, ["transcribe-srt", str(media), "-o", str(output), "--language", "en"]
    )

    assert result.exit_code == 0, result.output
    text = output.read_text(encoding="utf-8")
    assert text.startswith("1\n") and " --> " in text
    assert "Wrote" in result.output and "(en)" in result.output


def test_transcribe_srt_default_output(tmp_path, monkeypatch):
    _patch_command(monkeypatch)
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["transcribe-srt", str(media)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "talk.srt").exists()


def test_transcribe_srt_no_speech(tmp_path, monkeypatch):
    _patch_command(monkeypatch, segments=[])
    media = tmp_path / "silent.wav"
    media.write_bytes(b"fake")
    output = tmp_path / "out.srt"

    result = CliRunner().invoke(cli, ["transcribe-srt", str(media), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert "No speech detected" in result.output
    assert output.read_text(encoding="utf-8") == ""


def test_transcribe_srt_missing_whisperx(tmp_path, monkeypatch):
    def _raise(model, device):
        raise ModuleNotFoundError("No module named 'whisperx'")

    _patch_command(monkeypatch, load=_raise)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    result = CliRunner().invoke(cli, ["transcribe-srt", str(media)])

    assert result.exit_code != 0
    assert "whisperx is not installed" in result.output
    assert "pip install whisperx" in result.output
