import asyncio
import logging

from pydantic import BaseModel as PydanticBaseModel

logger = logging.getLogger(__name__)

MODEL = "gemini-3-flash"

EVALUATION_PROMPT = """\
You are a strict professional linguist evaluating text-to-speech output quality.

The audio you are hearing is a TTS-generated speech in {target_lang}, \
produced by a voice cloned from a {source_lang} speaker.

The intended text was:
"{text}"

Evaluate the audio on the following criteria, rating each from 1 to 10:

1. **fluency** — How smoothly does the speech flow? Are there unnatural pauses, \
stuttering, or robotic cadence?
2. **naturalness** — Does it sound like a real human speaking {target_lang}?
3. **accent_score** — How native does the accent sound? 10 = indistinguishable \
from a native speaker, 1 = heavy foreign accent.
4. **clarity** — Is the pronunciation clear and intelligible?
5. **overall** — Holistic quality score. Be strict: reserve 8+ for genuinely \
impressive quality that could pass as native speech.

Be rigorous. Most TTS output with accent transfer deserves 4-7. Only truly \
exceptional output should score 8+. A score of 9-10 means you would not be able \
to distinguish this from a native speaker recording.

Provide brief reasoning for your scores."""


class FluencyScore(PydanticBaseModel):
    fluency: float
    naturalness: float
    accent_score: float
    clarity: float
    overall: float
    reasoning: str


class FluencyResponse(PydanticBaseModel):
    scores: FluencyScore


class GeminiAudioEvaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def evaluate(
        self,
        audio_bytes: bytes,
        text: str,
        target_lang: str,
        source_lang: str,
    ) -> FluencyScore:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=self.api_key)
        prompt = EVALUATION_PROMPT.format(
            target_lang=target_lang,
            source_lang=source_lang,
            text=text,
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=[
                prompt,
                gtypes.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FluencyResponse,
            ),
        )

        result = FluencyResponse.model_validate_json(response.text)
        return result.scores
