import asyncio
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel as PydanticBaseModel

from dub.models.schemas import Segment, TranslatedSegment, Word
from dub.providers.protocols import TranslationProvider

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
# 50% of Gemini 2.5 Flash's ~1M token context at ~4 chars/token
MAX_WINDOW_CHARS = 2_000_000
CONTEXT_SEGMENTS = 10
PAUSE_THRESHOLD = 0.7  # seconds


class TranslatedBlock(PydanticBaseModel):
    start: float
    end: float
    translated_text: str


class TranslationResponse(PydanticBaseModel):
    blocks: list[TranslatedBlock]


SYSTEM_PROMPT = """\
You are a professional translator. Translate the following speech segments \
from {source_lang} to {target_lang}. Maintain natural speech patterns suitable \
for voice dubbing. Keep translations concise to fit similar time durations. \
Ensure groups of sentences are coherent with one another.

{context_section}\
For each segment, return its start time, end time, and translated text \
in the structured format requested."""


@dataclass
class Window:
    segments: list[Segment] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(seg.text) for seg in self.segments)


class GeminiTranslation(TranslationProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @staticmethod
    def _group_into_sentences(
        words: list[Word], pause_threshold: float = PAUSE_THRESHOLD
    ) -> list[Segment]:
        if not words:
            return []
        sentences: list[Segment] = []
        current: list[Word] = [words[0]]
        for word in words[1:]:
            if word.start - current[-1].end >= pause_threshold:
                sentences.append(
                    Segment(
                        start=current[0].start,
                        end=current[-1].end,
                        text=" ".join(w.text for w in current),
                    )
                )
                current = [word]
            else:
                current.append(word)
        if current:
            sentences.append(
                Segment(
                    start=current[0].start,
                    end=current[-1].end,
                    text=" ".join(w.text for w in current),
                )
            )
        return sentences

    @staticmethod
    def _build_windows(segments: list[Segment], max_chars: int) -> list[Window]:
        windows: list[Window] = []
        current = Window()
        for seg in segments:
            if current.segments and current.char_count + len(seg.text) > max_chars:
                windows.append(current)
                current = Window(segments=[seg])
            else:
                current.segments.append(seg)
        if current.segments:
            windows.append(current)
        return windows

    def _build_prompt(
        self,
        window: Window,
        context: list[TranslatedSegment],
        source_lang: str,
        target_lang: str,
    ) -> str:
        if context:
            context_lines = "\n".join(
                f"{i + 1}. [{seg.original_text}] → [{seg.translated_text}]"
                for i, seg in enumerate(context)
            )
            context_section = (
                "Previous translations (for context and coherence — "
                "do NOT include in output):\n"
                + context_lines
                + "\n\n"
            )
        else:
            context_section = ""

        segment_lines = "\n".join(
            f"{i + 1}. [{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}"
            for i, seg in enumerate(window.segments)
        )
        return (
            SYSTEM_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                context_section=context_section,
            )
            + "\n\nSegments to translate:\n"
            + segment_lines
        )

    async def translate_chunks(
        self,
        words: list[Word],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedSegment]:
        source = source_lang or "auto-detected language"
        sentences = self._group_into_sentences(words)
        logger.info(
            f"[Translation] {len(words)} words → {len(sentences)} sentences, "
            f"translating from {source} to {target_lang}"
        )

        if not self.api_key:
            logger.warning("[Translation] No Gemini API key, returning stub translations")
            return self._stub_translate(sentences, target_lang)

        try:
            from google import genai
            from google.genai import types as gtypes

            client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.warning(f"[Translation] Failed to init Gemini client: {e}, using stubs")
            return self._stub_translate(sentences, target_lang)

        translated: list[TranslatedSegment] = []
        windows = self._build_windows(sentences, MAX_WINDOW_CHARS)
        logger.info(f"[Translation] Split into {len(windows)} window(s)")

        for window in windows:
            context = translated[-CONTEXT_SEGMENTS:] if translated else []
            prompt = self._build_prompt(window, context, source, target_lang)

            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TranslationResponse,
                    ),
                )
                result = TranslationResponse.model_validate_json(response.text)
                for seg, block in zip(window.segments, result.blocks):
                    translated.append(
                        TranslatedSegment(
                            start=seg.start,
                            end=seg.end,
                            original_text=seg.text,
                            translated_text=block.translated_text,
                        )
                    )
            except Exception as e:
                logger.error(f"[Translation] Gemini API error: {e}, using original text")
                for seg in window.segments:
                    translated.append(
                        TranslatedSegment(
                            start=seg.start,
                            end=seg.end,
                            original_text=seg.text,
                            translated_text=seg.text,
                        )
                    )

        return translated

    def _stub_translate(
        self, segments: list[Segment], target_lang: str
    ) -> list[TranslatedSegment]:
        return [
            TranslatedSegment(
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text=f"[{target_lang}] {seg.text}",
            )
            for seg in segments
        ]
