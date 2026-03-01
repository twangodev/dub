import asyncio
import logging

from dub.models.schemas import Segment, TranslatedSegment
from dub.providers.protocols import TranslationProvider

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10  # Segments per translation request
SYSTEM_PROMPT = """\
You are a professional translator. Translate the following speech segments \
from {source_lang} to {target_lang}. Maintain natural speech patterns suitable \
for voice dubbing. Keep translations concise to fit similar time durations.

Return ONLY a JSON array of translated strings, one per input segment, \
in the same order. No explanations."""


class GeminiTranslation(TranslationProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def translate_chunks(
        self,
        segments: list[Segment],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedSegment]:
        source = source_lang or "auto-detected language"
        logger.info(
            f"[Translation] Translating {len(segments)} segments "
            f"from {source} to {target_lang}"
        )

        if not self.api_key:
            logger.warning("[Translation] No Gemini API key, returning stub translations")
            return self._stub_translate(segments, target_lang)

        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.warning(f"[Translation] Failed to init Gemini client: {e}, using stubs")
            return self._stub_translate(segments, target_lang)

        translated: list[TranslatedSegment] = []
        for chunk_start in range(0, len(segments), CHUNK_SIZE):
            chunk = segments[chunk_start : chunk_start + CHUNK_SIZE]
            texts = [seg.text for seg in chunk]

            prompt = (
                SYSTEM_PROMPT.format(source_lang=source, target_lang=target_lang)
                + "\n\nSegments to translate:\n"
                + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
            )

            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                import json

                # Extract JSON array from response
                response_text = response.text.strip()
                if response_text.startswith("```"):
                    response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                translations = json.loads(response_text)

                for seg, trans_text in zip(chunk, translations):
                    translated.append(
                        TranslatedSegment(
                            start=seg.start,
                            end=seg.end,
                            original_text=seg.text,
                            translated_text=str(trans_text),
                        )
                    )
            except Exception as e:
                logger.error(f"[Translation] Gemini API error: {e}, using original text")
                for seg in chunk:
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
