import json
import logging
from pathlib import Path

from dub.models.schemas import Segment, TranslatedSegment
from dub.pipeline.context import JobContext
from dub.providers.audio.extractor import extract_audio
from dub.providers.audio.speed import time_stretch
from dub.providers.audio.assembler import assemble_audio, mux_video

logger = logging.getLogger(__name__)

# Pause threshold for segmenting words into utterances (seconds)
PAUSE_THRESHOLD = 0.7


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
        json.dump([item.model_dump() for item in data], f, indent=2, default=str)


def save_audio(path: Path, audio_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)


async def run_dubbing_pipeline(ctx: JobContext) -> Path:
    """Execute the full dubbing pipeline."""
    logger.info(f"[Pipeline] Starting dubbing pipeline for job {ctx.job_id}")

    # 1. Extract audio from video
    await ctx.emit_progress("extract_audio", "running")
    await extract_audio(ctx.input_video, ctx.job_dir / "audio_original.wav")
    await ctx.emit_progress("extract_audio", "complete")

    # 2. Separate speech from background
    await ctx.emit_progress("separate_audio", "running")
    if ctx.separator is not None:
        separated = await ctx.separator.separate(ctx.audio_original, ctx.job_dir)
    else:
        # No separation — use original audio as speech, create silent background
        import shutil
        speech_path = ctx.job_dir / "audio_speech.wav"
        shutil.copy2(ctx.audio_original, speech_path)
        from dub.models.schemas import SeparatedAudio
        separated = SeparatedAudio(
            speech_path=speech_path,
            background_path=ctx.job_dir / "audio_background.wav",
        )
    await ctx.emit_progress("separate_audio", "complete")

    # 3. ASR — word-level timestamps
    await ctx.emit_progress("transcribe", "running")
    segments = await ctx.stt.transcribe(separated.speech_path)
    save_json(ctx.job_dir / "segments.json", segments)
    await ctx.emit_progress("transcribe", "complete")

    # 4. Segment into utterances
    await ctx.emit_progress("segment", "running")
    utterances = segment_into_utterances(segments)
    await ctx.emit_progress("segment", "complete")

    # 5. Translation
    await ctx.emit_progress("translate", "running")
    translated = await ctx.translator.translate_chunks(
        utterances, ctx.target_lang, ctx.source_lang
    )
    save_json(ctx.job_dir / "translated.json", translated)
    await ctx.emit_progress("translate", "complete")

    # 6. Extract voice reference for cloning
    voice_ref = extract_voice_sample(separated.speech_path)

    # 7. TTS per segment
    await ctx.emit_progress("tts", "running")
    tts_dir = ctx.job_dir / "tts_segments"
    tts_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(translated):
        audio_bytes = await ctx.tts.synthesize(
            seg.translated_text, voice_reference=voice_ref
        )
        seg_path = tts_dir / f"{i:03d}.wav"
        save_audio(seg_path, audio_bytes)

        # Speed-adjust to fit original time slot
        target_duration = seg.end - seg.start
        if target_duration > 0:
            await time_stretch(seg_path, target_duration)

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

    logger.info(f"[Pipeline] Dubbing pipeline complete for job {ctx.job_id}")
    return output_path
