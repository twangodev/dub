import asyncio
import logging
from pathlib import Path

import ffmpeg

from dub.models.schemas import TranslatedSegment

logger = logging.getLogger(__name__)


async def assemble_audio(
    tts_dir: Path,
    background_path: Path,
    segments: list[TranslatedSegment],
    output_path: Path,
) -> None:
    """Combine TTS segments with background audio using FFmpeg.

    Builds a complex filter that places each TTS segment at its correct
    timestamp and mixes everything with the background track.
    """
    logger.info(f"[Assemble] Combining {len(segments)} TTS segments with background")

    # Collect existing TTS segment files
    inputs: list[tuple[int, Path, TranslatedSegment]] = []
    for i, seg in enumerate(segments):
        tts_file = tts_dir / f"{i:03d}.wav"
        if tts_file.exists():
            inputs.append((i, tts_file, seg))
        else:
            logger.warning(f"[Assemble] Missing TTS segment {tts_file}")

    if not inputs:
        logger.warning("[Assemble] No TTS segments found, copying background as output")
        import shutil
        shutil.copy2(background_path, output_path)
        return

    # Build ffmpeg-python filter graph
    background = ffmpeg.input(str(background_path))
    mix_streams = [background.audio]

    for idx, (i, tts_path, seg) in enumerate(inputs):
        delay_ms = int(seg.start * 1000)
        tts_input = ffmpeg.input(str(tts_path))
        delayed = tts_input.filter("adelay", f"{delay_ms}|{delay_ms}")
        mix_streams.append(delayed)

    num_streams = len(mix_streams)
    mixed = ffmpeg.filter(mix_streams, "amix", inputs=num_streams, duration="longest", normalize=0)

    stream = ffmpeg.output(mixed, str(output_path)).overwrite_output()

    try:
        await asyncio.to_thread(stream.run, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg assemble failed: {e.stderr.decode()}") from e

    logger.info(f"[Assemble] Assembled audio saved to {output_path}")


async def mux_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Replace audio in video with dubbed audio using FFmpeg."""
    logger.info(f"[Mux] Replacing audio in {video_path}")

    video = ffmpeg.input(str(video_path))
    audio = ffmpeg.input(str(audio_path))

    stream = (
        ffmpeg
        .output(
            video.video,
            audio.audio,
            str(output_path),
            vcodec="copy",
            shortest=None,
        )
        .overwrite_output()
    )

    try:
        await asyncio.to_thread(stream.run, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg mux failed: {e.stderr.decode()}") from e

    logger.info(f"[Mux] Output video saved to {output_path}")