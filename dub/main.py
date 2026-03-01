import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from redis.asyncio import Redis

from dub.api.routes import router
from dub.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes


async def _cleanup_expired_jobs() -> None:
    """Periodically remove disk data for jobs whose Redis keys have expired."""
    data_dir = Path(settings.data_dir)
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            if not data_dir.exists():
                continue
            redis = Redis.from_url(settings.redis_url)
            try:
                for job_dir in data_dir.iterdir():
                    if not job_dir.is_dir():
                        continue
                    job_id = job_dir.name
                    exists = await redis.exists(f"job:{job_id}")
                    if not exists:
                        logger.info(f"[Cleanup] Removing expired job directory: {job_id}")
                        shutil.rmtree(job_dir)
                        await redis.srem("jobs", job_id)
            finally:
                await redis.aclose()
        except Exception:
            logger.exception("[Cleanup] Error during expired job cleanup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_expired_jobs())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Dub - AI Video Dubbing", version="0.1.0", lifespan=lifespan)

app.include_router(router)
