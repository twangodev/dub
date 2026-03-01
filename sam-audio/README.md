# sam-audio-inference

FastAPI server wrapping Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) for speech/background separation. Uses Gemini to auto-detect the speaker voice for better separation quality.

## Running with Docker

**Prerequisites:** NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

```bash
docker run --gpus all -p 8000:8000 \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e HF_TOKEN=your_huggingface_token \
  -v sam-audio-cache:/root/.cache \
  ghcr.io/twangodev/sam-audio-inference:latest
```

### Required Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for speaker detection |
| `HF_TOKEN` | HuggingFace token with access to [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large) |

### Volumes

Mount `/root/.cache` to persist model weights (~6 GB) across container restarts.

## API

### `POST /separate`

Upload an audio/video file and get separated speech and background tracks.

```bash
curl -X POST http://localhost:8000/separate -F "file=@video.mp4"
```

```json
{
  "speech_url": "http://localhost:8000/files/<job_id>/speech.wav",
  "background_url": "http://localhost:8000/files/<job_id>/background.wav"
}
```

### `GET /files/{job_id}/{filename}`

Download a separated audio file.

### `DELETE /files/{job_id}`

Clean up output files for a job.
