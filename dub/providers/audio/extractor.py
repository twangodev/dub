import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video using FFmpeg."""
    logger.info(f"[Extract] Extracting audio from {video_path} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg extract failed: {stderr.decode()}")

    logger.info(f"[Extract] Audio extracted to {output_path}")
    return output_path
