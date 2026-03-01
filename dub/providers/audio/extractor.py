import asyncio
import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


async def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video using FFmpeg."""
    logger.info(f"[Extract] Extracting audio from {video_path} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = (
        ffmpeg
        .input(str(video_path))
        .output(
            str(output_path),
            vn=None,
            acodec="pcm_s16le",
            ar=44100,
            ac=1,
        )
        .overwrite_output()
    )

    try:
        await asyncio.to_thread(stream.run, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg extract failed: {e.stderr.decode()}") from e

    logger.info(f"[Extract] Audio extracted to {output_path}")
    return output_path