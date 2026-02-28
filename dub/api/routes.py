import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from redis.asyncio import Redis
from sse_starlette.sse import EventSourceResponse

from dub.config import settings
from dub.tasks.dubbing import run_dubbing_job

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _get_redis() -> Redis:
    return Redis.from_url(settings.redis_url)


@router.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_lang: str = Form(...),
    source_lang: str | None = Form(None),
):
    """Upload a video and start a dubbing job."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(settings.data_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    input_path = job_dir / "input.mp4"
    content = await file.read()
    input_path.write_bytes(content)

    # Store job metadata in Redis
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "target_lang": target_lang,
        "source_lang": source_lang,
        "stage": None,
        "detail": None,
        "error": None,
        "output_path": None,
    }
    redis = _get_redis()
    try:
        await redis.set(f"job:{job_id}", json.dumps(job_data))
        await redis.sadd("jobs", job_id)
    finally:
        await redis.aclose()

    # Dispatch task
    await run_dubbing_job.kiq(job_id, str(input_path), target_lang, source_lang)

    logger.info(f"[API] Created job {job_id} -> {target_lang}")
    return {"job_id": job_id}


@router.get("/jobs")
async def list_jobs():
    """List all jobs with status."""
    redis = _get_redis()
    try:
        job_ids = await redis.smembers("jobs")
        jobs = []
        for jid in sorted(job_ids):
            jid_str = jid.decode() if isinstance(jid, bytes) else jid
            data = await redis.get(f"job:{jid_str}")
            if data:
                jobs.append(json.loads(data))
        return {"jobs": jobs}
    finally:
        await redis.aclose()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details."""
    redis = _get_redis()
    try:
        data = await redis.get(f"job:{job_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Job not found")
        return json.loads(data)
    finally:
        await redis.aclose()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    """SSE stream of progress events for a job."""
    redis = _get_redis()

    # Verify job exists
    data = await redis.get(f"job:{job_id}")
    if not data:
        await redis.aclose()
        raise HTTPException(status_code=404, detail="Job not found")
    await redis.aclose()

    async def event_generator():
        sub_redis = _get_redis()
        pubsub = sub_redis.pubsub()
        try:
            await pubsub.subscribe(f"job:{job_id}:progress")

            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    yield {"event": "progress", "data": data}

                    # Check if pipeline is done
                    event = json.loads(data)
                    if event.get("stage") == "pipeline" and event.get("status") in (
                        "completed",
                        "failed",
                    ):
                        yield {"event": "done", "data": data}
                        break
                else:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            await pubsub.unsubscribe(f"job:{job_id}:progress")
            await pubsub.aclose()
            await sub_redis.aclose()

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}/output")
async def get_output(job_id: str):
    """Download the dubbed video."""
    output_path = Path(settings.data_dir) / job_id / "output.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not ready")
    return FileResponse(
        str(output_path),
        media_type="video/mp4",
        filename=f"dubbed_{job_id}.mp4",
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Cancel/delete a job."""
    redis = _get_redis()
    try:
        await redis.delete(f"job:{job_id}")
        await redis.srem("jobs", job_id)
    finally:
        await redis.aclose()

    # Clean up files
    import shutil

    job_dir = Path(settings.data_dir) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    return {"status": "deleted"}
