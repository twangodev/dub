from typing import Protocol
from pathlib import Path

from dub.models.schemas import Segment, TranslatedSegment, SeparatedAudio, Word


class STTProvider(Protocol):
    async def transcribe(self, audio_path: Path) -> list[Segment]:
        """Returns segments with word-level timestamps."""
        ...


class AudioSeparator(Protocol):
    async def separate(self, input_path: Path, output_dir: Path) -> SeparatedAudio:
        """Splits input (video or audio) into speech + background tracks."""
        ...


class TranslationProvider(Protocol):
    async def translate_chunks(
        self,
        words: list[Word],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedSegment]:
        """Translates words (with timestamps) in context windows with rolling context."""
        ...


class TTSProvider(Protocol):
    async def synthesize(
        self,
        text: str,
        voice_reference: bytes | None = None,
        reference_id: str | None = None,
    ) -> bytes:
        """Generates speech audio from text. Returns WAV bytes."""
        ...
