import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
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
async def job_events(job_id: str, request: Request):
    """SSE stream of progress events with replay support.

    Clients reconnecting with Last-Event-ID will receive only missed events.
    """
    redis = _get_redis()

    # Verify job exists
    data = await redis.get(f"job:{job_id}")
    if not data:
        await redis.aclose()
        raise HTTPException(status_code=404, detail="Job not found")
    await redis.aclose()

    last_event_id = request.headers.get("Last-Event-ID")

    async def event_generator():
        stream_key = f"job:{job_id}:progress"
        r = _get_redis()
        try:
            # Phase 1: Replay — send all events after last_event_id (or from start)
            replay_start = last_event_id if last_event_id else "0-0"
            # XRANGE is inclusive, use "(" prefix to make it exclusive when replaying
            range_start = f"({replay_start}" if last_event_id else "0-0"
            entries = await r.xrange(stream_key, min=range_start)

            terminal_seen = False
            for entry_id, fields in entries:
                eid = entry_id if isinstance(entry_id, str) else entry_id.decode()
                raw = fields.get("data") or fields.get(b"data", b"")
                data = raw if isinstance(raw, str) else raw.decode()
                yield {"event": "progress", "data": data, "id": eid}

                event = json.loads(data)
                if event.get("stage") == "pipeline" and event.get("status") in (
                    "completed",
                    "failed",
                ):
                    yield {"event": "done", "data": data, "id": eid}
                    terminal_seen = True
                    break

            if terminal_seen:
                return

            # Phase 2: Live-tail with XREAD BLOCK
            # Start reading after the last replayed entry, or from the beginning
            last_id = entries[-1][0] if entries else (last_event_id or "0-0")
            if isinstance(last_id, bytes):
                last_id = last_id.decode()

            while True:
                if await request.is_disconnected():
                    break

                result = await r.xread(
                    {stream_key: last_id}, block=5000, count=10
                )

                if not result:
                    # Timeout — send keepalive
                    yield {"event": "ping", "data": ""}
                    continue

                for _stream_name, messages in result:
                    for entry_id, fields in messages:
                        eid = entry_id if isinstance(entry_id, str) else entry_id.decode()
                        raw = fields.get("data") or fields.get(b"data", b"")
                        data = raw if isinstance(raw, str) else raw.decode()
                        yield {"event": "progress", "data": data, "id": eid}

                        event = json.loads(data)
                        if event.get("stage") == "pipeline" and event.get("status") in (
                            "completed",
                            "failed",
                        ):
                            yield {"event": "done", "data": data, "id": eid}
                            return

                        last_id = eid
        finally:
            await r.aclose()

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
        await redis.delete(f"job:{job_id}:progress")
        await redis.srem("jobs", job_id)
    finally:
        await redis.aclose()

    # Clean up files
    import shutil

    job_dir = Path(settings.data_dir) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    return {"status": "deleted"}
