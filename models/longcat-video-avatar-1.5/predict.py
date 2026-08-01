import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from cog import BasePredictor, Input
from cog import Path as CogPath


class Predictor(BasePredictor):
    def predict(
        self,
        reference_image: CogPath = Input(  # noqa: B008
            description="Reference avatar image"
        ),
        audio: CogPath = Input(description="Speech audio track"),  # noqa: B008
    ) -> CogPath:
        output = NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = Path(output.name)
        output.close()

        command = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(reference_image),
            "-i",
            str(audio),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg failed")
        return CogPath(str(output_path))
