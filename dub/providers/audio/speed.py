import asyncio
import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


async def time_stretch(audio_path: Path, target_duration: float) -> None:
    """Time-stretch audio file to fit target duration using FFmpeg's atempo filter."""
    logger.info(f"[Speed] Adjusting {audio_path} to {target_duration:.2f}s")

    # Get current duration using ffmpeg.probe()
    try:
        probe_result = await asyncio.to_thread(ffmpeg.probe, str(audio_path))
    except ffmpeg.Error as e:
        logger.warning(f"[Speed] ffprobe failed, skipping speed adjust: {e.stderr.decode()}")
        return

    current_duration = float(probe_result["format"]["duration"])
    if current_duration <= 0 or target_duration <= 0:
        return

    ratio = current_duration / target_duration
    # atempo filter accepts values between 0.5 and 100.0
    ratio = max(0.5, min(ratio, 100.0))

    if abs(ratio - 1.0) < 0.05:
        return  # Close enough, skip processing

    # Chain atempo filters for ratios outside 0.5-2.0 range
    filters: list[tuple[str, float]] = []
    remaining = ratio
    while remaining > 2.0:
        filters.append(("atempo", 2.0))
        remaining /= 2.0
    while remaining < 0.5:
        filters.append(("atempo", 0.5))
        remaining /= 0.5
    filters.append(("atempo", round(remaining, 4)))

    tmp_path = audio_path.with_suffix(".tmp.wav")

    # Build the filter chain
    stream = ffmpeg.input(str(audio_path))
    for filter_name, filter_val in filters:
        stream = stream.filter(filter_name, filter_val)
    stream = stream.output(str(tmp_path)).overwrite_output()

    try:
        await asyncio.to_thread(stream.run, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        logger.warning(f"[Speed] FFmpeg speed adjust failed: {e.stderr.decode()}")
        tmp_path.unlink(missing_ok=True)
        return

    tmp_path.replace(audio_path)
    logger.info(f"[Speed] Adjusted {audio_path} ({ratio:.2f}x)")