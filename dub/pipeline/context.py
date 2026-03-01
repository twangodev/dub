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
