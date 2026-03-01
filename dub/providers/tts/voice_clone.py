import asyncio
import logging
import tempfile
from pathlib import Path

import ffmpeg
from fishaudio import AsyncFishAudio

from dub.models.schemas import Word

logger = logging.getLogger(__name__)

# Chunk selection parameters
TARGET_DURATION = 30.0  # ideal chunk length in seconds
MIN_DURATION = 10.0
MAX_DURATION = 45.0
GAP_THRESHOLD = 0.5  # max gap between words to stay in the same run

# Model training poll parameters
POLL_INTERVAL = 2.0  # seconds between status checks
POLL_TIMEOUT = 120.0  # max seconds to wait for training


def select_best_speech_chunk(
    words: list[Word],
) -> tuple[float, float, str] | None:
    """Pick the best continuous speech chunk from word-level timestamps.

    Groups words into continuous runs (gap < GAP_THRESHOLD), then selects
    the run closest to TARGET_DURATION (between MIN and MAX).
    Falls back to the longest run if none meets MIN_DURATION.

    Returns (start_time, end_time, transcript_text) or None.
    """
    if not words:
        return None

    # Build continuous runs
    runs: list[list[Word]] = []
    current_run: list[Word] = [words[0]]

    for word in words[1:]:
        gap = word.start - current_run[-1].end
        if gap >= GAP_THRESHOLD:
            runs.append(current_run)
            current_run = [word]
        else:
            current_run.append(word)
    runs.append(current_run)

    # Filter to runs that meet minimum duration and cap at max
    candidates: list[tuple[float, float, str]] = []
    for run in runs:
        start = run[0].start
        end = run[-1].end
        duration = end - start
        if duration >= MIN_DURATION:
            # Cap at MAX_DURATION
            if duration > MAX_DURATION:
                end = start + MAX_DURATION
                # Trim words to fit
                text = " ".join(w.text for w in run if w.end <= end)
            else:
                text = " ".join(w.text for w in run)
            candidates.append((start, end, text))

    if candidates:
        # Pick the one closest to target duration
        best = min(candidates, key=lambda c: abs((c[1] - c[0]) - TARGET_DURATION))
        logger.info(
            f"[VoiceClone] Selected chunk: {best[0]:.1f}s - {best[1]:.1f}s "
            f"({best[1] - best[0]:.1f}s)"
        )
        return best

    # Fallback: use the longest run regardless of minimum
    if runs:
        longest = max(runs, key=lambda r: r[-1].end - r[0].start)
        start = longest[0].start
        end = longest[-1].end
        text = " ".join(w.text for w in longest)
        logger.warning(
            f"[VoiceClone] No chunk meets {MIN_DURATION}s minimum, "
            f"using longest run: {start:.1f}s - {end:.1f}s ({end - start:.1f}s)"
        )
        return (start, end, text)

    return None


async def _extract_audio_chunk(
    speech_path: Path, start: float, end: float,
) -> bytes:
    """Extract an audio slice from the speech WAV via FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        stream = (
            ffmpeg.input(str(speech_path), ss=start, to=end)
            .output(str(tmp_path), acodec="pcm_s16le", ar=44100, ac=1)
            .overwrite_output()
        )
        await asyncio.to_thread(stream.run, capture_stdout=True, capture_stderr=True)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


async def extract_best_voice_sample(
    speech_path: Path,
    words: list[Word],
) -> bytes | None:
    """Extract the best speech chunk as raw audio bytes for inline voice reference.

    Uses the same timestamp-guided selection as voice clone creation,
    but returns the audio bytes directly instead of uploading to Fish Audio.
    """
    chunk = select_best_speech_chunk(words)
    if chunk is None:
        return None

    start, end, _ = chunk
    try:
        return await _extract_audio_chunk(speech_path, start, end)
    except Exception as e:
        logger.error(f"[VoiceClone] Fallback audio extraction failed: {e}")
        return None


async def create_voice_clone(
    api_key: str,
    speech_path: Path,
    words: list[Word],
    job_id: str,
) -> str | None:
    """Create a persistent voice model on Fish Audio from the best speech chunk.

    Returns the model_id on success, None on any failure (graceful fallback).
    """
    # 1. Select the best chunk
    chunk = select_best_speech_chunk(words)
    if chunk is None:
        logger.warning("[VoiceClone] No suitable speech chunk found")
        return None

    start, end, transcript = chunk

    # 2. Extract audio slice
    try:
        chunk_bytes = await _extract_audio_chunk(speech_path, start, end)
    except Exception as e:
        logger.error(f"[VoiceClone] Audio extraction failed: {e}")
        return None

    # 3. Create model on Fish Audio
    client = AsyncFishAudio(api_key=api_key)
    try:
        model = await client.voices.create(
            title=f"dub-clone-{job_id}",
            voices=[chunk_bytes],
            texts=[transcript],
        )
        model_id = model.id
        logger.info(f"[VoiceClone] Model created: {model_id}, waiting for training...")
    except Exception as e:
        logger.error(f"[VoiceClone] Model creation failed: {e}")
        return None

    # 4. Poll until trained
    elapsed = 0.0
    while elapsed < POLL_TIMEOUT:
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        try:
            voice = await client.voices.get(model_id)
            if voice.state == "trained":
                logger.info(f"[VoiceClone] Model {model_id} trained successfully")
                return model_id
            if voice.state == "failed":
                logger.error(f"[VoiceClone] Model {model_id} training failed")
                await delete_voice_clone(api_key, model_id)
                return None
        except Exception as e:
            logger.warning(f"[VoiceClone] Poll error: {e}")

    logger.error(f"[VoiceClone] Model {model_id} training timed out after {POLL_TIMEOUT}s")
    await delete_voice_clone(api_key, model_id)
    return None


async def create_voice_clone_from_samples(
    api_key: str,
    audio_samples: list[bytes],
    transcripts: list[str],
    job_id: str,
    label: str = "fluent",
) -> str | None:
    """Create a voice model on Fish Audio from pre-generated audio samples.

    Returns the model_id on success, None on any failure.
    """
    client = AsyncFishAudio(api_key=api_key)
    try:
        model = await client.voices.create(
            title=f"dub-{label}-{job_id}",
            voices=audio_samples,
            texts=transcripts,
        )
        model_id = model.id
        logger.info(f"[VoiceClone] Fluent model created: {model_id}, waiting for training...")
    except Exception as e:
        logger.error(f"[VoiceClone] Fluent model creation failed: {e}")
        return None

    # Poll until trained
    elapsed = 0.0
    while elapsed < POLL_TIMEOUT:
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        try:
            voice = await client.voices.get(model_id)
            if voice.state == "trained":
                logger.info(f"[VoiceClone] Fluent model {model_id} trained successfully")
                return model_id
            if voice.state == "failed":
                logger.error(f"[VoiceClone] Fluent model {model_id} training failed")
                await delete_voice_clone(api_key, model_id)
                return None
        except Exception as e:
            logger.warning(f"[VoiceClone] Poll error: {e}")

    logger.error(f"[VoiceClone] Fluent model {model_id} training timed out after {POLL_TIMEOUT}s")
    await delete_voice_clone(api_key, model_id)
    return None


async def delete_voice_clone(api_key: str, model_id: str) -> None:
    """Delete a voice model from Fish Audio. Best-effort, catches all exceptions."""
    try:
        client = AsyncFishAudio(api_key=api_key)
        await client.voices.delete(model_id)
        logger.info(f"[VoiceClone] Deleted model {model_id}")
    except Exception as e:
        logger.warning(f"[VoiceClone] Failed to delete model {model_id}: {e}")
