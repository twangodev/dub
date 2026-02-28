import json
import logging
from pathlib import Path

from redis.asyncio import Redis

from dub.config import settings
from dub.pipeline.context import JobContext
from dub.pipeline.orchestrator import run_dubbing_pipeline
from dub.providers.factory import create_stt, create_separator, create_translator, create_tts
from dub.tasks.broker import broker

logger = logging.getLogger(__name__)


async def update_job_status(
    job_id: str, status: str, error: str | None = None, output_path: str | None = None
) -> None:
    """Update job status in Redis."""
    redis = Redis.from_url(settings.redis_url)
    try:
        job_data = await redis.get(f"job:{job_id}")
        if job_data:
            job = json.loads(job_data)
        else:
            job = {"job_id": job_id}

        job["status"] = status
        if error:
            job["error"] = error
        if output_path:
            job["output_path"] = output_path

        await redis.set(f"job:{job_id}", json.dumps(job))

        # Publish completion/failure event
        event = {"stage": "pipeline", "status": status}
        if error:
            event["detail"] = error
        if output_path:
            event["output_url"] = f"/api/jobs/{job_id}/output"
        await redis.publish(f"job:{job_id}:progress", json.dumps(event))
    finally:
        await redis.aclose()


@broker.task
async def run_dubbing_job(
    job_id: str, input_path: str, target_lang: str, source_lang: str | None = None
) -> str:
    logger.info(f"[Task] Starting dubbing job {job_id}")

    job_dir = Path(settings.data_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ctx = JobContext(
        job_id=job_id,
        job_dir=job_dir,
        input_video=Path(input_path),
        target_lang=target_lang,
        source_lang=source_lang,
        stt=create_stt(settings),
        separator=create_separator(settings),
        translator=create_translator(settings),
        tts=create_tts(settings),
    )

    # Set up Redis for progress emission
    redis = Redis.from_url(settings.redis_url)
    await ctx.set_redis(redis)

    try:
        await update_job_status(job_id, "running")
        output = await run_dubbing_pipeline(ctx)
        await update_job_status(job_id, "completed", output_path=str(output))
        logger.info(f"[Task] Job {job_id} completed: {output}")
        return str(output)
    except Exception as e:
        logger.exception(f"[Task] Job {job_id} failed")
        await update_job_status(job_id, "failed", error=str(e))
        raise
    finally:
        await redis.aclose()
