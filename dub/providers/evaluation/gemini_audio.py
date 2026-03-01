import asyncio
import logging
from typing import Literal

from pydantic import BaseModel as PydanticBaseModel

logger = logging.getLogger(__name__)

MODEL = "gemini-3-flash"

EVALUATION_PROMPT = """\
You are a demanding {target_lang} language teacher evaluating a student's \
spoken performance. The student's native language is {source_lang} and they \
are attempting to speak {target_lang}.

Listen to this audio recording of the student reading the following passage:
"{text}"

Grade the student on each criterion using a 0-100 scale (integer percentages). \
You are a notoriously tough grader — most students with detectable foreign accents \
score between 40-70%. Only a student who sounds completely indistinguishable from \
a native speaker earns above 95%.

1. **fluency** — Speech flow, pacing, rhythm. Deduct heavily for unnatural pauses, \
robotic cadence, or hesitation. A native speaker flows effortlessly at 95-100%.
2. **naturalness** — Does this sound like a real person? Synthetic artifacts, \
monotone delivery, or odd intonation patterns should drop the score significantly.
3. **accent_score** — How native is the accent? Any detectable foreign influence \
caps this at 85%. Even slight non-native vowel coloring or consonant substitution \
caps at 90%. Only perfect native pronunciation scores 95%+.
4. **clarity** — Pronunciation accuracy and intelligibility. Mispronounced phonemes, \
swallowed syllables, or unclear articulation should be penalized.
5. **overall** — Your final grade for this student. This is the grade that goes on \
their transcript. Be ruthless: a 98%+ means this student could fool a native speaker \
into thinking they grew up speaking {target_lang}. Most accent-transferred TTS \
realistically deserves 50-75%.

Think carefully before assigning scores. Provide brief but specific reasoning \
citing concrete examples from the audio (e.g., specific mispronunciations, \
unnatural prosody patterns, or robotic artifacts you noticed)."""


class FluencyScore(PydanticBaseModel):
    fluency: float
    naturalness: float
    accent_score: float
    clarity: float
    overall: float
    reasoning: str


class FluencyResponse(PydanticBaseModel):
    scores: FluencyScore


PAIRWISE_PROMPT = """\
You are a demanding {target_lang} language teacher. The student's native language \
is {source_lang}. You will hear TWO recordings of the same passage read aloud:

"{text}"

Recording A is the first audio clip. Recording B is the second audio clip.

Your job: which recording sounds MORE like a native {target_lang} speaker? \
Consider accent authenticity, natural prosody, rhythm, and overall fluency. \
Ignore minor volume or quality differences — focus solely on how native the \
speech sounds.

You MUST pick a winner — no ties allowed."""


class PairwiseResult(PydanticBaseModel):
    winner: Literal["A", "B"]
    reasoning: str


class PairwiseResponse(PydanticBaseModel):
    result: PairwiseResult


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

    async def compare_pairwise(
        self,
        audio_a: bytes,
        audio_b: bytes,
        text: str,
        target_lang: str,
        source_lang: str,
    ) -> PairwiseResult:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=self.api_key)
        prompt = PAIRWISE_PROMPT.format(
            target_lang=target_lang,
            source_lang=source_lang,
            text=text,
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=[
                prompt,
                gtypes.Part.from_bytes(data=audio_a, mime_type="audio/wav"),
                gtypes.Part.from_bytes(data=audio_b, mime_type="audio/wav"),
            ],
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PairwiseResponse,
            ),
        )

        result = PairwiseResponse.model_validate_json(response.text)
        return result.result
