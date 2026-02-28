import asyncio
import logging
from pathlib import Path

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

    # Build FFmpeg command with complex filter
    # Input 0 = background, Input 1..N = TTS segments
    cmd = ["ffmpeg", "-y", "-i", str(background_path)]
    for _, tts_path, _ in inputs:
        cmd.extend(["-i", str(tts_path)])

    # Build filter: delay each TTS segment to its start time, then mix all
    filter_parts = []
    mix_inputs = ["[0:a]"]  # background is first mix input

    for idx, (i, _, seg) in enumerate(inputs):
        input_num = idx + 1  # 0 is background
        delay_ms = int(seg.start * 1000)
        label = f"delayed{idx}"
        filter_parts.append(f"[{input_num}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        mix_inputs.append(f"[{label}]")

    num_streams = len(mix_inputs)
    mix_str = "".join(mix_inputs)
    filter_parts.append(f"{mix_str}amix=inputs={num_streams}:duration=longest:normalize=0")

    filter_complex = ";".join(filter_parts)
    cmd.extend(["-filter_complex", filter_complex, str(output_path)])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg assemble failed: {stderr.decode()}")

    logger.info(f"[Assemble] Assembled audio saved to {output_path}")


async def mux_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Replace audio in video with dubbed audio using FFmpeg."""
    logger.info(f"[Mux] Replacing audio in {video_path}")

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg mux failed: {stderr.decode()}")

    logger.info(f"[Mux] Output video saved to {output_path}")