import logging
from pathlib import Path

import httpx

from dub.models.schemas import Segment, Word
from dub.providers.protocols import STTProvider

logger = logging.getLogger(__name__)


class QwenSTT(STTProvider):
    def __init__(self, stt_url: str):
        self.stt_url = stt_url

    async def transcribe(self, audio_path: Path) -> list[Segment]:
        logger.info(f"[STT] Transcribing {audio_path} via Qwen at {self.stt_url}")

        async with httpx.AsyncClient(timeout=300) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    f"{self.stt_url}/transcribe",
                    files={"file": (audio_path.name, f, "audio/wav")},
                )
            response.raise_for_status()
            data = response.json()

        words = [
            Word(start=ts["start_time"], end=ts["end_time"], text=ts["text"])
            for ts in data.get("timestamps", [])
        ]
        full_text = data.get("text", " ".join(w.text for w in words))
        start = words[0].start if words else 0.0
        end = words[-1].end if words else 0.0

        return [Segment(start=start, end=end, text=full_text, words=words)]