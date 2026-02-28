import asyncio
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


async def time_stretch(audio_path: Path, target_duration: float) -> None:
    """Time-stretch audio file to fit target duration using FFmpeg's atempo filter."""
    logger.info(f"[Speed] Adjusting {audio_path} to {target_duration:.2f}s")

    # Get current duration
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.warning(f"[Speed] ffprobe failed, skipping speed adjust: {stderr.decode()}")
        return

    current_duration = float(stdout.decode().strip())
    if current_duration <= 0 or target_duration <= 0:
        return

    ratio = current_duration / target_duration
    # atempo filter accepts values between 0.5 and 100.0
    ratio = max(0.5, min(ratio, 100.0))

    if abs(ratio - 1.0) < 0.05:
        return  # Close enough, skip processing

    # Chain atempo filters for ratios outside 0.5-2.0 range
    filters = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    filter_str = ",".join(filters)

    tmp_path = audio_path.with_suffix(".tmp.wav")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-filter:a", filter_str,
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.warning(f"[Speed] FFmpeg speed adjust failed: {stderr.decode()}")
        tmp_path.unlink(missing_ok=True)
        return

    tmp_path.replace(audio_path)
    logger.info(f"[Speed] Adjusted {audio_path} ({ratio:.2f}x)")
