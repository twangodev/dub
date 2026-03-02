import logging
from dataclasses import dataclass
from pathlib import Path

from redis.asyncio import Redis

from dub.models.schemas import ProgressEvent
from dub.providers.protocols import AudioSeparator, STTProvider, TranslationProvider, TTSProvider

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    job_id: str
    job_dir: Path
    input_video: Path
    target_lang: str
    stt: STTProvider
    separator: AudioSeparator
    translator: TranslationProvider
    tts: TTSProvider
    source_lang: str | None = None
    fish_audio_api_key: str = ""
    gemini_api_key: str = ""

    # Per-job voice reference (skips automatic voice clone creation)
    voice_reference_id: str | None = None

    # Iterative refinement
    max_generations: int = 4
    samples_per_generation: int = 10
    top_k_samples: int = 5
    plateau_threshold: float = 2.0
    min_fluency_score: float = 98.0
    eval_script_target_duration: float = 30.0

    # Duration fitting
    duration_tolerance: float = 0.10
    max_fit_attempts: int = 5
    samples_per_step: int = 5
    speed_min: float = 0.85
    speed_max: float = 1.3

    # Redis connection for stream progress
    _redis: Redis | None = None

    async def emit_progress(self, stage: str, status: str, detail: str | None = None) -> None:
        event = ProgressEvent(stage=stage, status=status, detail=detail)
        logger.info(f"[Job {self.job_id}] {stage}: {status}" + (f" ({detail})" if detail else ""))

        if self._redis is not None:
            try:
                stream_key = f"job:{self.job_id}:progress"
                await self._redis.xadd(stream_key, {"data": event.model_dump_json()})
            except Exception as e:
                logger.error(f"[Job {self.job_id}] Failed to publish progress: {e}")

    async def set_redis(self, redis: Redis) -> None:
        self._redis = redis
