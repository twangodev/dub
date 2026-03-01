import logging
import mimetypes
from pathlib import Path

import httpx

from dub.models.schemas import SeparatedAudio
from dub.providers.protocols import AudioSeparator

logger = logging.getLogger(__name__)


class SAMAudioSeparator(AudioSeparator):
    def __init__(self, sam_audio_url: str):
        self.sam_audio_url = sam_audio_url

    async def separate(self, input_path: Path, output_dir: Path) -> SeparatedAudio:
        logger.info(f"[Separation] Separating {input_path} via SAM Audio at {self.sam_audio_url}")

        speech_path = output_dir / "audio_speech.wav"
        background_path = output_dir / "audio_background.wav"

        mime_type = mimetypes.guess_type(str(input_path))[0] or "application/octet-stream"

        async with httpx.AsyncClient(timeout=600) as client:
            with open(input_path, "rb") as f:
                response = await client.post(
                    f"{self.sam_audio_url}/separate",
                    files={"file": (input_path.name, f, mime_type)},
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