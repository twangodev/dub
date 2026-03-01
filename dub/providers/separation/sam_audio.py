import logging
import shutil
from pathlib import Path

import httpx

from dub.models.schemas import SeparatedAudio

logger = logging.getLogger(__name__)


class SAMAudioSeparator:
    def __init__(self, sam_audio_url: str):
        self.sam_audio_url = sam_audio_url

    async def separate(self, audio_path: Path, output_dir: Path) -> SeparatedAudio:
        logger.info(f"[Separation] Separating {audio_path} via SAM Audio at {self.sam_audio_url}")

        speech_path = output_dir / "audio_speech.wav"
        background_path = output_dir / "audio_background.wav"

        try:
            async with httpx.AsyncClient(timeout=600) as client:
                with open(audio_path, "rb") as f:
                    response = await client.post(
                        f"{self.sam_audio_url}/separate",
                        files={"file": (audio_path.name, f, "audio/wav")},
                    )
                response.raise_for_status()
                data = response.json()

                # Download separated tracks
                for key, dest in [("speech_url", speech_path), ("background_url", background_path)]:
                    if key in data:
                        track_resp = await client.get(data[key])
                        track_resp.raise_for_status()
                        dest.write_bytes(track_resp.content)

                # Clean up files on the SAM Audio server
                if "speech_url" in data:
                    job_url = data["speech_url"].rsplit("/", 1)[0]
                    try:
                        await client.delete(job_url)
                    except Exception:
                        logger.debug("[Separation] Failed to clean up remote job files")

            return SeparatedAudio(speech_path=speech_path, background_path=background_path)
        except httpx.ConnectError:
            logger.warning("[Separation] SAM Audio service unavailable, copying original as speech")
            return self._stub_separate(audio_path, speech_path, background_path)

    def _stub_separate(
        self, audio_path: Path, speech_path: Path, background_path: Path
    ) -> SeparatedAudio:
        """Fallback: copy original audio as speech, create silent background."""
        shutil.copy2(audio_path, speech_path)
        # Create a minimal silent WAV as background placeholder
        self._create_silent_wav(background_path)
        return SeparatedAudio(speech_path=speech_path, background_path=background_path)

    def _create_silent_wav(self, path: Path) -> None:
        """Create a short silent WAV file."""
        import struct

        sample_rate = 44100
        duration = 1
        num_samples = sample_rate * duration
        data_size = num_samples * 2  # 16-bit mono
        with open(path, "wb") as f:
            # WAV header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(b"\x00" * data_size)
