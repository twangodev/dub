"""FastAPI wrapper around SAM-Audio for the dub pipeline."""

import mimetypes
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

load_dotenv()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Patch ImageBindRanker so it doesn't require the imagebind package.
# Lite mode removes the visual_ranker immediately after loading, so it's never used.
from sam_audio.ranking.imagebind import ImageBindRanker as _IBR
_IBR.__init__ = lambda self, cfg: super(_IBR, self).__init__()

# Patch BaseModel._from_pretrained for newer huggingface_hub (>=0.37) which no longer
# passes `proxies` and `resume_download` to _from_pretrained.
from sam_audio.model.base import BaseModel as _BM
_orig_fp = _BM._from_pretrained.__func__

@classmethod
def _compat_fp(cls, *, proxies=None, resume_download=False, **kw):
    return _orig_fp(cls, proxies=proxies, resume_download=resume_download, **kw)

_BM._from_pretrained = _compat_fp

from sam_audio_infer import SamAudioInfer

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

model: SamAudioInfer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading SAM-Audio model (large)...")
    model = SamAudioInfer.from_pretrained(
        "large",
        dtype="bfloat16",
        enable_text_ranker=False,
        enable_span_predictor=False,
    )
    print("Model loaded.")
    yield


app = FastAPI(lifespan=lifespan)

SPEAKER_PROMPT = (
    "Describe the main speaker's voice in this video in 2-5 words for an audio "
    "separation model. Examples: 'Man speaking', 'Woman speaking', 'Young boy speaking', "
    "'Deep male voice speaking'. Reply with ONLY the short description, nothing else."
)


def _describe_speaker(file_bytes: bytes, mime_type: str) -> str:
    response = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=types.Content(
            parts=[
                types.Part(inline_data=types.Blob(data=file_bytes, mime_type=mime_type)),
                types.Part(text=SPEAKER_PROMPT),
            ]
        ),
    )
    return response.text.strip()


@app.post("/separate")
async def separate(request: Request, file: UploadFile):
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir()

    # Write uploaded file to disk
    file_bytes = await file.read()
    input_path = job_dir / (file.filename or "input")
    input_path.write_bytes(file_bytes)

    # Ask Gemini to describe the speaker from the video/audio
    mime_type = file.content_type
    if not mime_type or mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(file.filename or "video.mp4")[0] or "video/mp4"
    description = _describe_speaker(file_bytes, mime_type)
    print(f"  [{job_id}] Gemini description: {description}")

    # Run separation
    result = model.separate(str(input_path), description=description)

    speech_path = job_dir / "speech.wav"
    background_path = job_dir / "background.wav"
    result.save(str(speech_path), str(background_path))

    base_url = str(request.base_url).rstrip("/")
    return {
        "speech_url": f"{base_url}/files/{job_id}/speech.wav",
        "background_url": f"{base_url}/files/{job_id}/background.wav",
    }


@app.get("/files/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    path = OUTPUT_DIR / job_id / filename
    return FileResponse(path, media_type="audio/wav")


@app.delete("/files/{job_id}")
async def delete_job(job_id: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    shutil.rmtree(job_dir)
    return {"status": "deleted"}
