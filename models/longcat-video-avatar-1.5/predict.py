import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from cog import BasePredictor, Input
from cog import Path as CogPath

LONGCAT_GIT_REF = "6b3f4b8582a8bc3f20f795735f5383716c4ba794"
LONGCAT_ARCHIVE_URL = (
    f"https://codeload.github.com/meituan-longcat/LongCat-Video/zip/{LONGCAT_GIT_REF}"
)
BASE_MODEL_ID = "meituan-longcat/LongCat-Video"
AVATAR_MODEL_ID = "meituan-longcat/LongCat-Video-Avatar-1.5"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.repo_dir = self._ensure_longcat_source()
        self.weights_root = Path("/src/weights")
        self.base_weights_dir = self.weights_root / "LongCat-Video"
        self.avatar_weights_dir = self.weights_root / "LongCat-Video-Avatar-1.5"

    def predict(
        self,
        reference_image: CogPath = Input(description="Reference avatar image"),  # noqa: B008
        audio: CogPath = Input(description="Speech audio track"),  # noqa: B008
        prompt: str = Input(
            description="Prompt describing the subject and scene",
            default="A person is talking naturally.",
        ),  # noqa: B008
        resolution: str = Input(
            description="Output resolution", choices=["480p", "720p"], default="480p"
        ),  # noqa: B008
        num_segments: int = Input(
            description="Number of continuation segments", ge=1, le=20, default=1
        ),  # noqa: B008
    ) -> CogPath:
        self._ensure_weights()

        run_dir = Path(tempfile.mkdtemp(prefix="longcat-avatar-"))
        input_json = run_dir / "input.json"
        output_dir = run_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        input_json.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "cond_image": str(reference_image),
                    "cond_audio": {"person1": str(audio)},
                }
            ),
            encoding="utf-8",
        )

        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "run_demo_avatar_single_audio_to_video.py",
            "--context_parallel_size=1",
            f"--checkpoint_dir={self.avatar_weights_dir}",
            "--stage_1=ai2v",
            f"--input_json={input_json}",
            f"--output_dir={output_dir}",
            f"--resolution={resolution}",
            f"--num_segments={num_segments}",
            "--use_distill",
            "--use_int8",
            "--model_type=avatar-v1.5",
        ]

        result = subprocess.run(
            command,
            cwd=self.repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "LongCat inference failed")[-4000:]
            raise RuntimeError(details)

        videos = sorted(output_dir.rglob("*.mp4"))
        if not videos:
            raise RuntimeError("LongCat inference completed but did not produce an MP4 output")
        return CogPath(str(videos[-1]))

    def _ensure_longcat_source(self) -> Path:
        checkout_dir = Path("/src/runtime") / f"LongCat-Video-{LONGCAT_GIT_REF}"
        if checkout_dir.exists():
            return checkout_dir

        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        archive_path = checkout_dir.parent / f"{LONGCAT_GIT_REF}.zip"
        with urlopen(LONGCAT_ARCHIVE_URL) as response:
            archive_path.write_bytes(response.read())

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(checkout_dir.parent)

        archive_path.unlink(missing_ok=True)
        return checkout_dir

    def _ensure_weights(self) -> None:
        if self.base_weights_dir.exists() and self.avatar_weights_dir.exists():
            return

        from huggingface_hub import snapshot_download

        self.weights_root.mkdir(parents=True, exist_ok=True)
        if not self.base_weights_dir.exists():
            snapshot_download(repo_id=BASE_MODEL_ID, local_dir=str(self.base_weights_dir))
        if not self.avatar_weights_dir.exists():
            snapshot_download(repo_id=AVATAR_MODEL_ID, local_dir=str(self.avatar_weights_dir))
