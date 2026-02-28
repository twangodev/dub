import asyncio
import io
import logging
import struct

from dub.models.schemas import TranslatedSegment

logger = logging.getLogger(__name__)


class FishAudioTTS:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def synthesize(
        self,
        text: str,
        voice_reference: bytes | None = None,
        reference_id: str | None = None,
    ) -> bytes:
        logger.info(f"[TTS] Synthesizing: {text[:50]}...")

        if not self.api_key:
            logger.warning("[TTS] No Fish Audio API key, returning stub audio")
            return self._stub_audio(text)

        try:
            from fish_audio_sdk import Session, TTSRequest, ReferenceAudio

            session = Session(self.api_key)

            request_kwargs = {"text": text}
            if reference_id:
                request_kwargs["reference_id"] = reference_id
            elif voice_reference:
                request_kwargs["references"] = [
                    ReferenceAudio(audio=voice_reference, text="")
                ]

            result = b""
            for chunk in session.tts(TTSRequest(**request_kwargs)):
                result += chunk

            return result
        except Exception as e:
            logger.error(f"[TTS] Fish Audio error: {e}, returning stub audio")
            return self._stub_audio(text)

    def _stub_audio(self, text: str) -> bytes:
        """Generate a short silent WAV as placeholder."""
        sample_rate = 44100
        duration_sec = max(0.5, len(text) * 0.06)  # ~60ms per character
        num_samples = int(sample_rate * duration_sec)
        data_size = num_samples * 2  # 16-bit mono

        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(b"\x00" * data_size)
        return buf.getvalue()
