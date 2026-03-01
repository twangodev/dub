# dub

Orchestration layer that turns any video into any language while preserving the original speaker's voice.

## Pipeline

1. Extract audio (FFmpeg)
2. Separate speech from background (SAM Audio)
3. Transcribe with word-level timestamps (QwenASR + Qwen3-ForcedAligner)
4. Segment into utterances
5. Translate (Gemini)
6. Synthesize with cloned voice (Fish Audio)
7. Time-stretch to fit original timing
8. Mix with background track
9. Mux back into video

## Setup

```bash
cp .env.example .env
docker compose up -d redis
poe api
poe worker
```