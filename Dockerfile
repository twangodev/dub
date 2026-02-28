FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y ffmpeg rubberband-cli && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .

FROM base AS api
CMD ["uv", "run", "uvicorn", "dub.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
CMD ["uv", "run", "taskiq", "worker", "dub.worker:broker"]
