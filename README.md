# dub

Turn any video into any language while preserving the original speaker's voice.

## How It Works

1. Extract audio from video (FFmpeg)
2. Separate speech from background audio (SAM Audio)
3. Transcribe speech with word-level timestamps (Whisper)
4. Segment words into utterances by detecting pauses
5. Translate each utterance to the target language (Gemini)
6. Synthesize translated speech with the original speaker's cloned voice (Fish Audio)
7. Time-stretch each clip to fit the original utterance's duration
8. Mix synthesized speech with the preserved background track
9. Mux the dubbed audio back into the original video

**Architecture:**

FastAPI receives uploads and enqueues jobs via TaskIQ (Redis-backed). Workers pick up jobs and run the pipeline, publishing stage-by-stage progress over Redis pub/sub. Clients subscribe via SSE for real-time updates. All GPU inference (Whisper, SAM Audio) runs on external services — this project is purely an orchestrator.

All GPU inference (Whisper, SAM Audio) runs on external services. This project is an orchestrator — no GPU needed. Providers are hot-swappable via env vars and fall back to stubs when services are unavailable.

## Local Development

Copy the env template and fill in API keys:

```bash
cp .env.example .env
```

Start Redis:

```bash
docker compose up -d redis
```

Start the API and worker in separate terminals:

```bash
poe api
poe worker
```

## Docker

```bash
docker compose up
```
