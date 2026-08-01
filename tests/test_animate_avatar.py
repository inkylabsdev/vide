import base64
import json
from urllib.error import URLError

from click.testing import CliRunner

from vide.cli import cli
from vide.commands import animate_avatar


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_cli_registers_animate_avatar():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "animate-avatar" in result.output


def test_animate_avatar_calls_server_and_writes_output(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "avatar.mp4"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)

    payloads = []
    api_response = {
        "status": "succeeded",
        "output": "data:video/mp4;base64," + base64.b64encode(b"mp4-bytes").decode("ascii"),
    }

    def fake_urlopen(target):
        if isinstance(target, str):
            raise AssertionError("Unexpected string URL fetch in this test")
        payloads.append(json.loads(target.data.decode("utf-8")))
        return FakeResponse(json.dumps(api_response).encode("utf-8"))

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.request.urlopen", fake_urlopen)

    result = CliRunner().invoke(
        cli,
        ["animate-avatar", str(image), str(audio), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b"mp4-bytes"
    assert "Animated avatar written to" in result.output
    assert payloads == [
        {
            "input": {
                "reference_image": "data:image/png;base64,aW1hZ2U=",
                "audio": "data:audio/x-wav;base64,YXVkaW8=",
            }
        }
    ]


def test_animate_avatar_raises_if_model_dir_missing(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")

    monkeypatch.setattr(
        animate_avatar,
        "DEFAULT_MODEL_DIR",
        tmp_path / "models" / "longcat-video-avatar-1.5",
    )

    result = CliRunner().invoke(cli, ["animate-avatar", str(image), str(audio)])

    assert result.exit_code != 0
    assert "Model directory not found" in result.output


def test_animate_avatar_raises_if_server_unreachable(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)

    def fake_urlopen(target):
        raise URLError("connection refused")

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.request.urlopen", fake_urlopen)

    result = CliRunner().invoke(cli, ["animate-avatar", str(image), str(audio)])

    assert result.exit_code != 0
    assert "Failed to call Cog server" in result.output


def test_animate_avatar_polls_until_succeeded(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "avatar.mp4"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)
    states = iter(
        [
            {
                "status": "processing",
                "urls": {"get": "http://127.0.0.1:5000/predictions/1"},
            },
            {
                "status": "succeeded",
                "output": "data:video/mp4;base64,"
                + base64.b64encode(b"video-final").decode("ascii"),
            },
        ]
    )

    def fake_urlopen(target):
        if isinstance(target, str):
            return FakeResponse(json.dumps(next(states)).encode("utf-8"))
        return FakeResponse(json.dumps(next(states)).encode("utf-8"))

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.request.urlopen", fake_urlopen)
    monkeypatch.setattr("vide.commands.animate_avatar.time.sleep", lambda _: None)

    result = CliRunner().invoke(
        cli,
        ["animate-avatar", str(image), str(audio), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b"video-final"


def test_animate_avatar_raises_on_failed_status(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)

    def fake_urlopen(target):
        return FakeResponse(
            json.dumps({"status": "failed", "error": "boom"}).encode("utf-8")
        )

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.request.urlopen", fake_urlopen)

    result = CliRunner().invoke(cli, ["animate-avatar", str(image), str(audio)])

    assert result.exit_code != 0
    assert "cog prediction failed: boom" in result.output
