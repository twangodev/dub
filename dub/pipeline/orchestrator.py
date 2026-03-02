import json
import logging
from dataclasses import dataclass
from pathlib import Path

from dub.models.schemas import Segment, TranslatedSegment, Word
from dub.pipeline.context import JobContext
from dub.providers.audio.assembler import assemble_audio, mux_video
from dub.providers.audio.duration import get_audio_duration
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


async def synthesize_to_fit(
    tts: TTSProvider,
    text: str,
    target_duration: float,
    reference_id: str | None = None,
    voice_reference: bytes | None = None,
    *,
    duration_tolerance: float = 0.10,
    max_fit_attempts: int = 5,
    samples_per_step: int = 5,
    speed_min: float = 0.85,
    speed_max: float = 1.3,
) -> bytes:
    """Binary-search on TTS speed so the output fits within ±duration_tolerance of target.

    At each speed, generates samples_per_step samples and uses the median
    duration to decide which direction to bisect. Keeps the single best
    sample across all attempts.
    """
    lo = speed_min
    hi = speed_max
    best_audio: bytes | None = None
    best_diff = float("inf")

    for step in range(1, max_fit_attempts + 1):
        speed = (lo + hi) / 2 if step > 1 else 1.0

        # Generate multiple samples at this speed, early-exit if one fits
        samples: list[tuple[bytes, float]] = []
        for _ in range(samples_per_step):
            audio = await tts.synthesize(
                text, voice_reference=voice_reference,
                reference_id=reference_id, speed=speed,
            )
            dur = get_audio_duration(audio)
            samples.append((audio, dur))

            ratio = dur / target_duration if target_duration > 0 else 1.0
            if abs(ratio - 1.0) <= duration_tolerance:
                logger.info(
                    f"[Fit] target={target_duration:.2f}s, step {step} "
                    f"speed={speed:.2f} → {dur:.2f}s ✓ early accept"
                )
                return audio

        # Sort by duration and pick median
        samples.sort(key=lambda s: s[1])
        median_idx = len(samples) // 2
        median_duration = samples[median_idx][1]

        # Track best sample across all steps (closest to target)
        for audio, dur in samples:
            diff = abs(dur - target_duration)
            if diff < best_diff:
                best_diff = diff
                best_audio = audio

        ratio = median_duration / target_duration if target_duration > 0 else 1.0
        logger.info(
            f"[Fit] target={target_duration:.2f}s, step {step} "
            f"speed={speed:.2f} → median {median_duration:.2f}s "
            f"(ratio={ratio:.2f}, range={samples[0][1]:.2f}-{samples[-1][1]:.2f}s)"
        )

        if abs(ratio - 1.0) <= duration_tolerance:
            return best_audio  # type: ignore[return-value]

        if median_duration > target_duration:
            lo = speed
        else:
            hi = speed

    logger.warning(
        f"[Fit] Exhausted {max_fit_attempts} steps × {samples_per_step} samples, "
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


@dataclass
class GenerationResult:
    generation: int
    model_id: str | None
    best_score: float
    champion_audio: bytes
    champion_transcript: str
    ranked_samples: list[tuple[bytes, str]]  # (audio, transcript) sorted by wins desc


async def run_round_robin(
    evaluator: GeminiAudioEvaluator,
    samples: list[tuple[bytes, str]],
    text: str,
    target_lang: str,
    source_lang: str,
) -> list[tuple[bytes, str]]:
    """Compare all sample pairs via pairwise Gemini calls, return sorted by win count (most wins first)."""
    n = len(samples)
    wins = [0] * n

    # Build all pairs
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    logger.info(f"[RoundRobin] Comparing {len(pairs)} pairs from {n} samples")

    # Run all pairwise comparisons concurrently
    async def compare(i: int, j: int) -> tuple[int, int, str]:
        result = await evaluator.compare_pairwise(
            samples[i][0], samples[j][0], text, target_lang, source_lang,
        )
        return (i, j, result.winner)

    results = await asyncio.gather(
        *(compare(i, j) for i, j in pairs),
        return_exceptions=True,
    )

    for res in results:
        if isinstance(res, Exception):
            logger.warning(f"[RoundRobin] Pairwise comparison failed: {res}")
            continue
        i, j, winner = res
        if winner == "A":
            wins[i] += 1
        else:
            wins[j] += 1

    # Sort by wins descending
    indexed = list(enumerate(wins))
    indexed.sort(key=lambda x: x[1], reverse=True)

    win_summary = ", ".join(
        f"sample {idx} wins={w}" for idx, w in indexed
    )
    logger.info(f"[RoundRobin] Results: {win_summary}")

    return [samples[idx] for idx, _ in indexed]


async def run_iterative_refinement(
    ctx: JobContext,
    initial_model_id: str,
    eval_script: str,
    eval_duration: float,
    evaluator: GeminiAudioEvaluator,
    source_lang: str,
) -> tuple[str | None, list[str]]:
    """Run iterative multi-generation voice clone refinement.

    Returns (final_model_id, intermediate_model_ids_to_delete).
    """
    generation_results: list[GenerationResult] = []
    intermediate_models: list[str] = []
    current_model_id = initial_model_id

    for gen in range(ctx.max_generations):
        logger.info(f"[Refinement] === Generation {gen} ===")
        await ctx.emit_progress(
            "fluency_eval", "progress",
            f"Generation {gen}/{ctx.max_generations}",
        )

        # Generate samples_per_generation samples
        samples: list[tuple[bytes, str]] = []
        for s in range(ctx.samples_per_generation):
            audio = await synthesize_to_fit(
                ctx.tts, eval_script, eval_duration,
                reference_id=current_model_id,
                duration_tolerance=ctx.duration_tolerance,
                max_fit_attempts=ctx.max_fit_attempts,
                samples_per_step=ctx.samples_per_step,
                speed_min=ctx.speed_min,
                speed_max=ctx.speed_max,
            )
            samples.append((audio, eval_script))

        logger.info(f"[Refinement] Gen {gen}: generated {len(samples)} samples")

        # Round-robin pairwise ranking
        ranked = await run_round_robin(
            evaluator, samples, eval_script, ctx.target_lang, source_lang,
        )

        champion_audio, champion_transcript = ranked[0]

        # Absolute score on champion for plateau detection
        score = await evaluator.evaluate(
            champion_audio, eval_script, ctx.target_lang, source_lang,
        )
        logger.info(
            f"[Refinement] Gen {gen} champion: overall={score.overall:.1f} "
            f"(fluency={score.fluency:.1f}, naturalness={score.naturalness:.1f}, "
            f"accent={score.accent_score:.1f}, clarity={score.clarity:.1f})"
        )

        gen_result = GenerationResult(
            generation=gen,
            model_id=current_model_id,
            best_score=score.overall,
            champion_audio=champion_audio,
            champion_transcript=champion_transcript,
            ranked_samples=ranked[:ctx.top_k_samples],
        )
        generation_results.append(gen_result)

        # Log cross-generation progress
        scores_str = " -> ".join(
            f"Gen {r.generation}: {r.best_score:.1f}" for r in generation_results
        )
        logger.info(f"[Refinement] Progress: {scores_str}")

        # Check stopping conditions
        if score.overall >= ctx.min_fluency_score:
            logger.info(
                f"[Refinement] Early stop: score {score.overall:.1f} >= {ctx.min_fluency_score}"
            )
            break

        if gen > 0:
            improvement = score.overall - generation_results[gen - 1].best_score
            if improvement < ctx.plateau_threshold:
                logger.info(
                    f"[Refinement] Plateau: improvement {improvement:.1f} < {ctx.plateau_threshold} (plateau)"
                )
                break

        # Build training data for next generation clone
        # Prior-gen bests (identity anchors) + current gen top-K
        training_audio: list[bytes] = []
        training_transcripts: list[str] = []

        # Identity anchors: best from each prior generation
        for prior in generation_results:
            training_audio.append(prior.champion_audio)
            training_transcripts.append(prior.champion_transcript)

        # Current gen top-K
        for audio, transcript in ranked[:ctx.top_k_samples]:
            training_audio.append(audio)
            training_transcripts.append(transcript)

        # Create next-generation clone
        next_model_id = await create_voice_clone_from_samples(
            ctx.fish_audio_api_key,
            audio_samples=training_audio,
            transcripts=training_transcripts,
            job_id=ctx.job_id,
            label=f"gen{gen + 1}",
        )

        if next_model_id is None:
            logger.warning(f"[Refinement] Gen {gen + 1} clone creation failed, stopping")
            break

        intermediate_models.append(next_model_id)
        current_model_id = next_model_id

    # Find best generation overall
    best_gen = max(generation_results, key=lambda r: r.best_score)
    logger.info(
        f"[Refinement] Best generation: {best_gen.generation} "
        f"(score={best_gen.best_score:.1f})"
    )

    # Create final fluent clone from best generation's top-K + best-of-each-gen anchors
    final_audio: list[bytes] = []
    final_transcripts: list[str] = []

    # Identity anchors from each generation
    for gr in generation_results:
        final_audio.append(gr.champion_audio)
        final_transcripts.append(gr.champion_transcript)

    # Best generation's top-K samples
    for audio, transcript in best_gen.ranked_samples:
        final_audio.append(audio)
        final_transcripts.append(transcript)

    final_model_id = await create_voice_clone_from_samples(
        ctx.fish_audio_api_key,
        audio_samples=final_audio,
        transcripts=final_transcripts,
        job_id=ctx.job_id,
        label="fluent",
    )

    if final_model_id:
        logger.info(f"[Refinement] Final fluent clone created: {final_model_id}")
    else:
        logger.warning("[Refinement] Final fluent clone creation failed")

    return final_model_id, intermediate_models


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

    # 6. Voice cloning — use user-provided reference or create accent clone
    voice_model_id = None
    fluent_model_id = None
    voice_ref = None
    user_provided_voice = False
    await ctx.emit_progress("voice_clone", "running")

    if ctx.voice_reference_id:
        # User supplied a pre-existing Fish Audio voice model — skip clone creation
        voice_model_id = ctx.voice_reference_id
        user_provided_voice = True
        logger.info(f"[Pipeline] Using user-provided voice model: {voice_model_id}")
    elif ctx.fish_audio_api_key:
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

    # 6a. Iterative multi-generation voice clone refinement
    intermediate_models: list[str] = []
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
                if eval_duration >= ctx.eval_script_target_duration:
                    break
            eval_script = " ".join(eval_script_parts)
            logger.info(
                f"[Pipeline] Evaluation script: ~{eval_duration:.1f}s, "
                f"{len(eval_script)} chars"
            )

            evaluator = GeminiAudioEvaluator(ctx.gemini_api_key)
            source_lang = ctx.source_lang or "unknown"

            fluent_model_id, intermediate_models = await run_iterative_refinement(
                ctx, voice_model_id, eval_script, eval_duration,
                evaluator, source_lang,
            )
            await ctx.emit_progress("fluency_eval", "complete")

        except Exception:
            logger.exception("[Pipeline] Iterative refinement failed, using accent clone")
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
            duration_tolerance=ctx.duration_tolerance,
            max_fit_attempts=ctx.max_fit_attempts,
            samples_per_step=ctx.samples_per_step,
            speed_min=ctx.speed_min,
            speed_max=ctx.speed_max,
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

    # 10. Clean up voice clone models (accent, intermediate, and fluent)
    #     Skip deleting the voice model if user provided it — it's theirs, not ours.
    for mid in intermediate_models:
        await delete_voice_clone(ctx.fish_audio_api_key, mid)
    if fluent_model_id:
        await delete_voice_clone(ctx.fish_audio_api_key, fluent_model_id)
    if voice_model_id and not user_provided_voice:
        await delete_voice_clone(ctx.fish_audio_api_key, voice_model_id)

    logger.info(f"[Pipeline] Dubbing pipeline complete for job {ctx.job_id}")
    return output_path
