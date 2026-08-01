import subprocess

from click.testing import CliRunner

from vide.cli import cli
from vide.commands import animate_avatar


def test_cli_registers_animate_avatar():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "animate-avatar" in result.output


def test_animate_avatar_runs_cog_predict(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "avatar.mp4"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)

    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.subprocess.run", fake_run)

    result = CliRunner().invoke(
        cli,
        ["animate-avatar", str(image), str(audio), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "Animated avatar written to" in result.output
    assert calls == [
        {
            "command": [
                "cog",
                "predict",
                "-i",
                f"reference_image=@{image}",
                "-i",
                f"audio=@{audio}",
                "-o",
                f"output_video={output}",
            ],
            "cwd": model_dir,
            "capture_output": True,
            "text": True,
            "check": False,
        }
    ]


def test_animate_avatar_raises_if_cog_missing(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)

    def fake_run(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.subprocess.run", fake_run)

    result = CliRunner().invoke(cli, ["animate-avatar", str(image), str(audio)])

    assert result.exit_code != 0
    assert "`cog` was not found on PATH." in result.output


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


def test_animate_avatar_raises_on_cog_failure(tmp_path, monkeypatch):
    image = tmp_path / "avatar.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "models" / "longcat-video-avatar-1.5"
    model_dir.mkdir(parents=True)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr(animate_avatar, "DEFAULT_MODEL_DIR", model_dir)
    monkeypatch.setattr("vide.commands.animate_avatar.subprocess.run", fake_run)

    result = CliRunner().invoke(cli, ["animate-avatar", str(image), str(audio)])

    assert result.exit_code != 0
    assert "cog predict failed: boom" in result.output
