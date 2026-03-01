import asyncio
import logging
from pathlib import Path

import httpx

from dub.models.schemas import Segment, Word
from dub.providers.protocols import STTProvider

logger = logging.getLogger(__name__)


class WhisperSTT(STTProvider):
    def __init__(self, whisper_url: str):
        self.whisper_url = whisper_url

    async def transcribe(self, audio_path: Path) -> list[Segment]:
        logger.info(f"[STT] Transcribing {audio_path} via Whisper at {self.whisper_url}")

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                with open(audio_path, "rb") as f:
                    response = await client.post(
                        f"{self.whisper_url}/transcribe",
                        files={"file": (audio_path.name, f, "audio/wav")},
                    )
                response.raise_for_status()
                data = response.json()

            segments = []
            for seg in data.get("segments", []):
                words = [
                    Word(start=w["start"], end=w["end"], text=w["word"])
                    for w in seg.get("words", [])
                ]
                segments.append(
                    Segment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"],
                        words=words,
                    )
                )
            return segments
        except httpx.ConnectError:
            logger.warning("[STT] Whisper service unavailable, returning stub data")
            return self._stub_segments()

    def _stub_segments(self) -> list[Segment]:
        """Fallback stub data when Whisper service is not available."""
        return [
            Segment(
                start=0.0,
                end=2.5,
                text="Hello world, this is a test.",
                words=[
                    Word(start=0.0, end=0.4, text="Hello"),
                    Word(start=0.4, end=0.8, text="world,"),
                    Word(start=0.9, end=1.1, text="this"),
                    Word(start=1.1, end=1.3, text="is"),
                    Word(start=1.3, end=1.5, text="a"),
                    Word(start=1.5, end=2.5, text="test."),
                ],
            ),
            Segment(
                start=3.0,
                end=5.5,
                text="Welcome to the video dubbing demo.",
                words=[
                    Word(start=3.0, end=3.4, text="Welcome"),
                    Word(start=3.4, end=3.6, text="to"),
                    Word(start=3.6, end=3.8, text="the"),
                    Word(start=3.8, end=4.2, text="video"),
                    Word(start=4.2, end=4.6, text="dubbing"),
                    Word(start=4.6, end=5.5, text="demo."),
                ],
            ),
        ]
