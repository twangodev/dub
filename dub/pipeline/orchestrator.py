import json
import logging
from pathlib import Path

from dub.models.schemas import Segment, TranslatedSegment, Word
from dub.pipeline.context import JobContext
from dub.providers.audio.assembler import assemble_audio, mux_video
from dub.providers.audio.duration import get_wav_duration
from dub.providers.protocols import TTSProvider
from dub.providers.evaluation.gemini_audio import GeminiAudioEvaluator
from dub.providers.tts.voice_clone import (
    create_voice_clone,
    create_voice_clone_from_samples,
    delete_voice_clone,
    extract_best_voice_sample,
)

logger = logging.getLogger(__name__)

# Pause threshold for segmenting words into utterances (seconds)
PAUSE_THRESHOLD = 0.7

# Fluency evaluation parameters
EVAL_SCRIPT_TARGET_DURATION = 30.0  # seconds of text to generate
MIN_FLUENCY_SCORE = 98.0            # early-stop threshold (0-100 scale)
MAX_EVAL_ATTEMPTS = 10              # max TTS + eval retries

# Duration-fitting parameters
DURATION_TOLERANCE = 0.10  # ±10%
MAX_FIT_ATTEMPTS = 5       # binary search iterations
SPEED_MIN = 0.5            # slowest allowed
SPEED_MAX = 3.0            # fastest allowed


async def synthesize_to_fit(
    tts: TTSProvider,
    text: str,
    target_duration: float,
    reference_id: str | None = None,
    voice_reference: bytes | None = None,
) -> bytes:
    """Binary-search on TTS speed so the output fits within ±DURATION_TOLERANCE of target."""
    lo = SPEED_MIN
    hi = SPEED_MAX
    best_audio: bytes | None = None
    best_diff = float("inf")

    for attempt in range(1, MAX_FIT_ATTEMPTS + 1):
        speed = (lo + hi) / 2 if attempt > 1 else 1.0

        audio = await tts.synthesize(
            text, voice_reference=voice_reference,
            reference_id=reference_id, speed=speed,
        )
        actual = get_wav_duration(audio)
        diff = abs(actual - target_duration)

        if diff < best_diff:
            best_diff = diff
            best_audio = audio

        ratio = actual / target_duration if target_duration > 0 else 1.0
        logger.info(
            f"[Fit] target={target_duration:.2f}s, attempt {attempt} "
            f"speed={speed:.2f} → {actual:.2f}s (ratio={ratio:.2f})"
        )

        if abs(ratio - 1.0) <= DURATION_TOLERANCE:
            return audio  # within tolerance

        if actual > target_duration:
            # too long → need faster speed
            lo = speed
        else:
            # too short → need slower speed
            hi = speed

    logger.warning(
        f"[Fit] Exhausted {MAX_FIT_ATTEMPTS} attempts, "
        f"best diff={best_diff:.2f}s for target={target_duration:.2f}s"
    )
    return best_audio  # type: ignore[return-value]


def segment_into_utterances(segments: list[Segment]) -> list[Segment]:
    """Group word-level segments into utterance-level segments based on pauses."""
    if not segments:
        return []

    # Collect all words across segments
    all_words = []
    for seg in segments:
        if seg.words:
            all_words.extend(seg.words)
        else:
            # If no word-level data, treat the whole segment as one utterance
            all_words.append(seg)

    if not all_words:
        return segments

    utterances: list[Segment] = []
    current_words = [all_words[0]]

    for word in all_words[1:]:
        gap = word.start - current_words[-1].end
        if gap >= PAUSE_THRESHOLD:
            # Flush current utterance
            utterances.append(
                Segment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=" ".join(w.text for w in current_words),
                )
            )
            current_words = [word]
        else:
            current_words.append(word)

    # Flush last utterance
    if current_words:
        utterances.append(
            Segment(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=" ".join(w.text for w in current_words),
            )
        )

    return utterances


def extract_voice_sample(speech_path: Path, max_bytes: int = 500_000) -> bytes | None:
    """Extract a voice reference sample from the speech track for TTS cloning."""
    if not speech_path.exists():
        return None
    data = speech_path.read_bytes()
    # Return up to max_bytes for the voice reference
    return data[:max_bytes] if len(data) > max_bytes else data


def save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([item.model_dump() for item in data], f, indent=2, default=str, ensure_ascii=False)


def save_audio(path: Path, audio_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)


async def run_dubbing_pipeline(ctx: JobContext) -> Path:
    """Execute the full dubbing pipeline."""
    logger.info(f"[Pipeline] Starting dubbing pipeline for job {ctx.job_id}")

    # 1. Separate speech from background (SAM Audio accepts video directly)
    await ctx.emit_progress("separate_audio", "running")
    separated = await ctx.separator.separate(ctx.input_video, ctx.job_dir)
    await ctx.emit_progress("separate_audio", "complete")

    # 3. ASR — word-level timestamps
    await ctx.emit_progress("transcribe", "running")
    segments = await ctx.stt.transcribe(separated.speech_path)
    save_json(ctx.job_dir / "segments.json", segments)
    await ctx.emit_progress("transcribe", "complete")

    # 4. Translation (sentence grouping is handled inside the translator)
    await ctx.emit_progress("translate", "running")
    all_words: list[Word] = []
    for seg in segments:
        if seg.words:
            all_words.extend(seg.words)
        else:
            all_words.append(Word(start=seg.start, end=seg.end, text=seg.text))
    translated = await ctx.translator.translate_chunks(
        all_words, ctx.target_lang, ctx.source_lang
    )
    save_json(ctx.job_dir / "translated.json", translated)
    await ctx.emit_progress("translate", "complete")

    # 6. Voice cloning — create accent clone from the best speech chunk
    voice_model_id = None
    fluent_model_id = None
    voice_ref = None
    await ctx.emit_progress("voice_clone", "running")
    if ctx.fish_audio_api_key:
        try:
            voice_model_id = await create_voice_clone(
                ctx.fish_audio_api_key, separated.speech_path, all_words, ctx.job_id,
            )
        except Exception:
            logger.exception("[Pipeline] Voice clone creation failed, falling back")
            voice_model_id = None

    if voice_model_id is None:
        # Fallback: use timestamp-guided chunk extraction (no model upload)
        voice_ref = await extract_best_voice_sample(separated.speech_path, all_words)
        if voice_ref is None:
            voice_ref = extract_voice_sample(separated.speech_path)
    await ctx.emit_progress("voice_clone", "complete")

    # 6a. Generate evaluation script and evaluate with Gemini
    if voice_model_id and ctx.gemini_api_key:
        await ctx.emit_progress("fluency_eval", "running")
        try:
            # Build ~30s evaluation script from translated segments
            eval_script_parts: list[str] = []
            eval_duration = 0.0
            for seg in translated:
                seg_duration = seg.end - seg.start
                eval_script_parts.append(seg.translated_text)
                eval_duration += seg_duration
                if eval_duration >= EVAL_SCRIPT_TARGET_DURATION:
                    break
            eval_script = " ".join(eval_script_parts)
            logger.info(
                f"[Pipeline] Evaluation script: ~{eval_duration:.1f}s, "
                f"{len(eval_script)} chars"
            )

            evaluator = GeminiAudioEvaluator(ctx.gemini_api_key)
            source_lang = ctx.source_lang or "unknown"

            best_audio: bytes | None = None
            best_score = -1.0

            for attempt in range(1, MAX_EVAL_ATTEMPTS + 1):
                # Synthesize using the accent (first) clone
                eval_audio = await synthesize_to_fit(
                    ctx.tts, eval_script, EVAL_SCRIPT_TARGET_DURATION,
                    reference_id=voice_model_id,
                )

                # Evaluate with Gemini
                score = await evaluator.evaluate(
                    eval_audio, eval_script, ctx.target_lang, source_lang,
                )
                logger.info(
                    f"[Pipeline] Attempt {attempt}/{MAX_EVAL_ATTEMPTS}: "
                    f"overall={score.overall:.1f} "
                    f"(fluency={score.fluency:.1f}, naturalness={score.naturalness:.1f}, "
                    f"accent={score.accent_score:.1f}, clarity={score.clarity:.1f}) "
                    f"— {score.reasoning}"
                )
                await ctx.emit_progress(
                    "fluency_eval", "progress",
                    f"Attempt {attempt}/{MAX_EVAL_ATTEMPTS}: overall={score.overall:.1f}",
                )

                if score.overall > best_score:
                    best_score = score.overall
                    best_audio = eval_audio

                if score.overall >= MIN_FLUENCY_SCORE:
                    logger.info(
                        f"[Pipeline] Early stop: score {score.overall:.1f} >= {MIN_FLUENCY_SCORE}"
                    )
                    break

            logger.info(f"[Pipeline] Best fluency score: {best_score:.1f}")
            await ctx.emit_progress("fluency_eval", "complete", f"best={best_score:.1f}")

            # 6b. Create fluent clone from best sample
            if best_audio is not None:
                await ctx.emit_progress("fluent_clone", "running")
                try:
                    fluent_model_id = await create_voice_clone_from_samples(
                        ctx.fish_audio_api_key,
                        audio_samples=[best_audio],
                        transcripts=[eval_script],
                        job_id=ctx.job_id,
                    )
                    if fluent_model_id:
                        logger.info(f"[Pipeline] Fluent clone created: {fluent_model_id}")
                    else:
                        logger.warning("[Pipeline] Fluent clone creation returned None")
                except Exception:
                    logger.exception("[Pipeline] Fluent clone creation failed, using accent clone")
                    fluent_model_id = None
                await ctx.emit_progress("fluent_clone", "complete")

        except Exception:
            logger.exception("[Pipeline] Fluency evaluation failed, using accent clone")
            fluent_model_id = None

    # 7. TTS per segment — prefer fluent clone, fall back to accent clone
    tts_reference_id = fluent_model_id or voice_model_id
    await ctx.emit_progress("tts", "running")
    tts_dir = ctx.job_dir / "tts_segments"
    tts_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(translated):
        target_duration = seg.end - seg.start
        audio_bytes = await synthesize_to_fit(
            ctx.tts, seg.translated_text, target_duration,
            reference_id=tts_reference_id,
            voice_reference=voice_ref,
        )
        seg_path = tts_dir / f"{i:03d}.wav"
        save_audio(seg_path, audio_bytes)
        await ctx.emit_progress("tts", "progress", f"{i + 1}/{len(translated)}")

    await ctx.emit_progress("tts", "complete")

    # 8. Assemble: TTS segments + background -> dubbed audio
    await ctx.emit_progress("assemble", "running")
    dubbed_audio_path = ctx.job_dir / "audio_dubbed.wav"
    await assemble_audio(tts_dir, separated.background_path, translated, dubbed_audio_path)
    await ctx.emit_progress("assemble", "complete")

    # 9. Mux: replace audio in video
    await ctx.emit_progress("mux", "running")
    output_path = ctx.job_dir / "output.mp4"
    await mux_video(ctx.input_video, dubbed_audio_path, output_path)
    await ctx.emit_progress("mux", "complete")

    # 10. Clean up voice clone models (both accent and fluent)
    if fluent_model_id:
        await delete_voice_clone(ctx.fish_audio_api_key, fluent_model_id)
    if voice_model_id:
        await delete_voice_clone(ctx.fish_audio_api_key, voice_model_id)

    logger.info(f"[Pipeline] Dubbing pipeline complete for job {ctx.job_id}")
    return output_path
